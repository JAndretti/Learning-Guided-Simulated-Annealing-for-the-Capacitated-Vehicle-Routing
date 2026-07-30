"""End-to-end driver for the A-bis init x proposal factorial.

Runs the whole study defined in
docs/specs/2026-07-30-abis-stronger-baseline-design.md in order, reading each
stage's tuned constants out of the previous stage's CSV so nothing has to be
substituted by hand:

    smoke    tiny 8-cell run; checks the score cells are not paying lgsa's
             per-step feature cost (the failure that is otherwise invisible)
    tuneA    alpha1 / alpha2, on the tuning slice
    tuneB    INIT_TEMP per non-learned proposal, random init, alphas frozen
    tuneCW   INIT_TEMP / STOP_TEMP per non-learned proposal, CW init
    report   the reported factorial: primary + strongest-baseline columns
    analyse  tables, paired bootstrap, equal-time table, anytime figure

Every stage is resumable, so an interrupted run continues where it stopped.
Nothing here reads the reported slice before the constants are frozen.

Typical use on a fresh machine:

    # sanity first
    uv run eval/run_abis.py --stages smoke --out_dir res/abis_rtx5070

    # then the study, with a longer ladder
    uv run eval/run_abis.py --out_dir res/abis_rtx5070 \
        --budgets 1000,10000,100000 --cheap_extra_budgets 200000 \
        --tune_budget 10000

`--cheap_extra_budgets` gives the non-learned proposals steps beyond lgsa's
largest budget. They are cheaper per step, so at matched steps they finish
sooner than lgsa and their anytime curves never span its wall-clock, leaving the
equal-time comparison blank at the top budget. The extra rungs fill it in.
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_PY = os.path.join(HERE, "run.py")

ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
INIT_TEMP_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
# CW cells need a colder schedule than STOP_TEMP=0.01 allows, so the cold rungs
# lower both ends together, keeping the INIT/STOP ratio at 100.
CW_COLD_PAIRS = [(0.003, 0.00003), (0.03, 0.0003)]
NON_LEARNED = ["blind", "score1", "score2"]
CW_INIT = "Clark_and_Wright_reversal"


def sh(args: list[str]) -> None:
    """Invoke eval/run.py with the current interpreter."""
    cmd = [sys.executable, RUN_PY] + args
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def factorial(out_dir: str, csv_name: str, **kw) -> None:
    args = ["--dataset", "factorial", "--out_dir", out_dir, "--csv_name", csv_name, "--resume"]
    for k, v in kw.items():
        args += [f"--{k}", str(v)]
    sh(args)


def _read(out_dir: str, csv_name: str) -> pd.DataFrame:
    path = os.path.join(out_dir, f"{csv_name}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run the earlier stages first")
    return pd.read_csv(path)


def _warn_edge(label: str, value: float, grid: list[float]) -> None:
    if value in (grid[0], grid[-1]):
        print(
            f"  !! {label} winner {value:g} sits on a grid edge. The spec's rule is to "
            f"extend that edge by a decade and re-run before freezing.",
            flush=True,
        )


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def stage_smoke(a: argparse.Namespace) -> None:
    factorial(
        a.out_dir, "smoke", n_instances=64, offset=0,
        inits=f"random,{CW_INIT}", proposals="blind,score1,score2,lgsa",
        budgets=200, seeds=1234, alpha1=3, alpha2=1, init_temp=1.0,
    )
    d = _read(a.out_dir, "smoke")
    t = d.groupby("proposal").sa_time.mean()
    print("\nper-step cost check (mean sa_time):", flush=True)
    print(t.reindex(["blind", "score1", "score2", "lgsa"]).to_string(), flush=True)
    ok = t["blind"] < t["score1"] and t["score2"] < t["lgsa"]
    print(f"ordering blind < score1 and score2 < lgsa: {ok}", flush=True)
    if not ok:
        raise SystemExit(
            "FAIL: the score cells are not cheaper per step than lgsa, which means "
            "SCORE_FEATURE_FLAGS is not being applied and they are building all 13 "
            "feature groups. Every quality/time conclusion downstream would be void."
        )
    rnd = d[d.init == "random"]
    print(f"random-init cells all improved: {bool((rnd.final_cost < rnd.init_cost).all())}", flush=True)
    print("NOTE: CW+blind returning the construction cost unchanged is expected at "
          "200 steps, not a failure -- see the plan's Task 3.", flush=True)


def stage_tune_a(a: argparse.Namespace) -> None:
    grid = ",".join(f"{v:g}" for v in ALPHA_GRID)
    common = dict(
        n_instances=a.tune_instances, offset=a.offset_tune, inits="random",
        budgets=a.tune_budget, seeds=a.seed, init_temp=1.0,
    )
    factorial(a.out_dir, "tuning", proposals="score1", alpha1=grid, alpha2=1, **common)
    factorial(a.out_dir, "tuning", proposals="score2", alpha1=grid, alpha2=grid, **common)
    # blind reference at the same setting: without it the score1 numbers cannot be
    # read as better or worse than uniform.
    factorial(a.out_dir, "tuning", proposals="blind", alpha1=1, alpha2=1, **common)


def winners_a(a: argparse.Namespace) -> tuple[float, float]:
    d = _read(a.out_dir, "tuning")
    d = d[(d.budget == a.tune_budget) & (d.init == "random")]
    s1 = d[d.proposal == "score1"]
    s2 = d[d.proposal == "score2"]
    b1 = s1.loc[s1.final_cost.idxmin()]
    b2 = s2.loc[s2.final_cost.idxmin()]
    print(f"\nstage A winners: score1 alpha1={b1.alpha1:g} ({b1.final_cost:.4f}), "
          f"score2 alpha1={b2.alpha1:g} alpha2={b2.alpha2:g} ({b2.final_cost:.4f})",
          flush=True)
    if "blind" in set(d.proposal):
        bl = d[d.proposal == "blind"].final_cost.min()
        print(f"  blind reference {bl:.4f}  (score2 gain {b2.final_cost - bl:+.4f})", flush=True)
    _warn_edge("score2 alpha1", b2.alpha1, ALPHA_GRID)
    _warn_edge("score2 alpha2", b2.alpha2, ALPHA_GRID)
    # score2's joint winner sets alpha1 for both variants: the spec freezes one
    # alpha1, and score2 is the variant the headline comparison rests on.
    return float(b2.alpha1), float(b2.alpha2)


def stage_tune_b(a: argparse.Namespace, alpha1: float, alpha2: float) -> None:
    factorial(
        a.out_dir, "tuning", n_instances=a.tune_instances, offset=a.offset_tune,
        inits="random", proposals=",".join(NON_LEARNED), budgets=a.tune_budget,
        seeds=a.seed, alpha1=alpha1, alpha2=alpha2,
        init_temp=",".join(f"{v:g}" for v in INIT_TEMP_GRID),
    )


def stage_tune_cw(a: argparse.Namespace, alpha1: float, alpha2: float) -> None:
    factorial(
        a.out_dir, "tuning_cw", n_instances=a.tune_instances, offset=a.offset_tune,
        inits=CW_INIT, proposals=",".join(NON_LEARNED), budgets=a.tune_budget,
        seeds=a.seed, alpha1=alpha1, alpha2=alpha2,
        init_temp=",".join(f"{v:g}" for v in INIT_TEMP_GRID), stop_temp=0.01,
    )
    for it, st in CW_COLD_PAIRS:
        factorial(
            a.out_dir, "tuning_cw", n_instances=a.tune_instances, offset=a.offset_tune,
            inits=CW_INIT, proposals=",".join(NON_LEARNED), budgets=a.tune_budget,
            seeds=a.seed, alpha1=alpha1, alpha2=alpha2, init_temp=it, stop_temp=st,
        )


def winners_schedules(a: argparse.Namespace) -> dict:
    """Best (INIT_TEMP, STOP_TEMP) per (init, proposal), from the tuning slices."""
    out = {}
    rnd = _read(a.out_dir, "tuning")
    rnd = rnd[(rnd.budget == a.tune_budget) & (rnd.init == "random")]
    for p in NON_LEARNED:
        g = rnd[rnd.proposal == p]
        b = g.loc[g.final_cost.idxmin()]
        out[("random", p)] = (float(b.init_temp), float(b.stop_temp))
        _warn_edge(f"random/{p} INIT_TEMP", float(b.init_temp), INIT_TEMP_GRID)

    cw = _read(a.out_dir, "tuning_cw")
    cw = cw[cw.budget == a.tune_budget]
    cold = sorted({it for it, _ in CW_COLD_PAIRS} | set(INIT_TEMP_GRID))
    for p in NON_LEARNED:
        g = cw[cw.proposal == p]
        b = g.loc[g.final_cost.idxmin()]
        out[(CW_INIT, p)] = (float(b.init_temp), float(b.stop_temp))
        _warn_edge(f"{CW_INIT}/{p} INIT_TEMP", float(b.init_temp), cold)

    ckpt = float(rnd.stop_temp.iloc[0])
    out[("random", "lgsa")] = (1.0, ckpt)
    out[(CW_INIT, "lgsa")] = (1.0, ckpt)
    print("\ntuned schedules (INIT_TEMP / STOP_TEMP):", flush=True)
    for (i, p), (it, st) in sorted(out.items()):
        print(f"  {i:26s} {p:7s} {it:g} / {st:g}", flush=True)
    return out


def stage_report(a: argparse.Namespace, alpha1: float, alpha2: float, sched: dict) -> None:
    budgets = a.budgets
    cheap = ",".join(x for x in [budgets, a.cheap_extra_budgets] if x)

    # primary column: one shared schedule for every cell
    factorial(
        a.out_dir, "factorial", n_instances=a.n_instances, offset=a.offset,
        inits=f"random,{CW_INIT}", proposals="lgsa", budgets=budgets,
        seeds=a.seeds, init_temp=1.0,
    )
    factorial(
        a.out_dir, "factorial", n_instances=a.n_instances, offset=a.offset,
        inits=f"random,{CW_INIT}", proposals=",".join(NON_LEARNED), budgets=cheap,
        seeds=a.seeds, alpha1=alpha1, alpha2=alpha2, init_temp=1.0,
    )

    # strongest-baseline column: each non-learned cell at its own schedule
    for (init, p), (it, st) in sched.items():
        if p == "lgsa" or (it == 1.0 and st == sched[("random", "lgsa")][1]):
            continue  # already covered by the primary pass
        factorial(
            a.out_dir, "factorial", n_instances=a.n_instances, offset=a.offset,
            inits=init, proposals=p, budgets=cheap, seeds=a.seeds,
            alpha1=alpha1, alpha2=alpha2, init_temp=it, stop_temp=st,
        )


def stage_analyse(a: argparse.Namespace, sched: dict) -> None:
    analysis = os.path.join(HERE, "abis_analysis.py")
    base = [sys.executable, analysis, "--res_dir", a.out_dir,
            "--n_instances", str(a.n_instances), "--offset", str(a.offset)]
    print("\n$ primary column", flush=True)
    subprocess.run(base + ["--init_temp", "1.0", "--suffix", "_primary"], check=True)
    spec = ",".join(
        f"{i}:{p}:{it:g}:{st:g}" for (i, p), (it, st) in sched.items()
    )
    print("\n$ strongest-baseline column", flush=True)
    subprocess.run(base + ["--schedules", spec, "--suffix", "_strongest"], check=True)


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", type=str, default="res/abis",
                   help="Tag this per machine; costs differ across devices and "
                        "must never be pooled.")
    p.add_argument("--stages", type=str, default="tuneA,tuneB,tuneCW,report,analyse",
                   help="Comma-separated subset of "
                        "smoke,tuneA,tuneB,tuneCW,report,analyse")
    p.add_argument("--budgets", type=str, default="1000,5000,10000")
    p.add_argument("--cheap_extra_budgets", type=str, default="",
                   help="Extra step budgets for the non-learned proposals only, "
                        "so their anytime curves span lgsa's largest wall-clock.")
    p.add_argument("--tune_budget", type=int, default=5000)
    p.add_argument("--n_instances", type=int, default=1000)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--tune_instances", type=int, default=500)
    p.add_argument("--offset_tune", type=int, default=1000)
    p.add_argument("--seeds", type=str, default="1234,1235,1236")
    p.add_argument("--seed", type=int, default=1234, help="Seed for tuning runs.")
    a = p.parse_args()

    if a.offset_tune < a.offset + a.n_instances:
        raise SystemExit(
            f"tuning slice [{a.offset_tune}:...] overlaps the reported slice "
            f"[{a.offset}:{a.offset + a.n_instances}] -- constants would be tuned "
            f"on reported instances"
        )

    stages = [s for s in a.stages.split(",") if s]
    os.makedirs(a.out_dir, exist_ok=True)

    if "smoke" in stages:
        stage_smoke(a)
    if "tuneA" in stages:
        stage_tune_a(a)
    alpha1 = alpha2 = None
    if any(s in stages for s in ("tuneB", "tuneCW", "report")):
        alpha1, alpha2 = winners_a(a)
    if "tuneB" in stages:
        stage_tune_b(a, alpha1, alpha2)
    if "tuneCW" in stages:
        stage_tune_cw(a, alpha1, alpha2)
    sched = None
    if any(s in stages for s in ("report", "analyse")):
        sched = winners_schedules(a)
    if "report" in stages:
        stage_report(a, alpha1, alpha2, sched)
    if "analyse" in stages:
        stage_analyse(a, sched)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()

"""Tables and figures for the A-bis init x proposal factorial.

Produces, from res/abis/factorial.csv and its per-instance cost vectors:
  1. the cost table (cells x budgets), mean over seeds +- inter-seed spread
  2. the paired per-instance bootstrap against a reference cell
  3. the anytime cost-vs-total-time plot

The two variance sources are reported separately and never pooled: the
inter-seed spread answers the plan's verification point 4, the paired bootstrap
answers plan sections C.3 and D.5.

Usage:
    uv run eval/abis_analysis.py
    uv run eval/abis_analysis.py --reference random/lgsa --n_boot 10000
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROPOSAL_ORDER = ["blind", "score1", "score2", "lgsa"]


def load(csv_path: str, n_instances: int, offset: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[(df.n_instances == n_instances) & (df.offset == offset)].copy()
    if df.empty:
        raise SystemExit(f"No rows in {csv_path} for n_instances={n_instances}, offset={offset}")
    df["cell"] = df.init + "/" + df.proposal
    return df


def per_instance_matrix(df: pd.DataFrame, costs_dir: str) -> np.ndarray:
    """Stack one run's cost vector per row, in df order. -> [n_runs, n_instances]"""
    return np.stack([np.load(os.path.join(costs_dir, f)) for f in df.costs_file])


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds and inter-seed spread (max - min), per cell x budget."""
    g = df.groupby(["cell", "budget"]).final_cost
    out = pd.DataFrame({"mean": g.mean(), "spread": g.max() - g.min(), "n_seeds": g.count()})
    return out.reset_index()


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Paired CI for mean(a) - mean(b) by resampling shared instances.

    a, b are [n_instances] seed-averaged per-instance costs for two cells on the
    SAME instances. Resampling instance indices once and applying them to both
    preserves the pairing, which is what makes the CI tight.
    """
    d = a - b
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_table(
    df: pd.DataFrame, costs_dir: str, reference: str, n_boot: int, seed: int
) -> pd.DataFrame:
    """Every cell vs the reference, per budget, seed-averaged then bootstrapped."""
    rng = np.random.default_rng(seed)
    rows = []
    for budget, g in df.groupby("budget"):
        by_cell = {
            cell: per_instance_matrix(sub, costs_dir).mean(axis=0)
            for cell, sub in g.groupby("cell")
        }
        if reference not in by_cell:
            continue
        ref = by_cell[reference]
        for cell, vec in by_cell.items():
            delta, lo, hi = paired_bootstrap(vec, ref, n_boot, rng)
            rows.append({
                "budget": budget, "cell": cell, "mean_cost": float(vec.mean()),
                "delta_vs_ref": delta, "ci_lo": lo, "ci_hi": hi,
                "significant": not (lo <= 0.0 <= hi),
            })
    return pd.DataFrame(rows).sort_values(["budget", "delta_vs_ref"])


# Okabe-Ito in the order already validated for this repo's figures
# (eval/bench_plots.py); the 4-slot subset re-validated at worst adjacent
# deltaE 11.0 deutan / 24.2 normal-vision. Slot 4 (#E69F00) sits below 3:1
# contrast on white, so every series also carries a direct label and its own
# marker -- identity is never colour-alone, and the figure survives greyscale.
PROPOSAL_COLOUR = {
    "blind": "#0072B2",
    "score1": "#D55E00",
    "score2": "#009E73",
    "lgsa": "#E69F00",
}
PROPOSAL_MARKER = {"blind": "o", "score1": "s", "score2": "^", "lgsa": "D"}
INIT_STYLE = {"random": "-", "Clark_and_Wright": "--", "Clark_and_Wright_reversal": "--"}
INK, INK_MUTED, GRID = "#1a1a1a", "#5c5c5c", "#e6e6e6"


def anytime_plot(df: pd.DataFrame, out_path: str, zoom_pad: float = 0.05) -> None:
    """Mean cost vs mean total wall-clock, two panels.

    total_time = setup_time + sa_time, so a Clarke-Wright start carries its
    construction cost -- exactly the cost a random start does not pay.

    Two panels rather than one: the random-init cells span 16-33 while five of
    the eight cells live inside a 0.4-unit band, so a single axis renders every
    comparison that matters as one flat line. The right panel is the same data
    on a y-range fitted to the dense band -- small multiples, not a second
    y-scale.

    Colour encodes the proposal, linestyle the initialisation, so the eight
    cells need only four hues. Every line is also direct-labelled.
    """
    g = df.groupby(["cell", "budget"]).agg(
        cost=("final_cost", "mean"), t=("total_time", "mean")
    ).reset_index()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 7,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    lo = g.cost.min()
    hi = min(g.cost.max(), lo + 0.6)
    dense = (lo - zoom_pad, hi + zoom_pad)

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(6.9, 2.7), layout="constrained"
    )

    for ax, ylim, title in (
        (ax_full, None, "full range"),
        (ax_zoom, dense, f"detail: {dense[0]:.2f}-{dense[1]:.2f}"),
    ):
        ax.grid(True, which="major", linewidth=0.4, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for cell, sub in g.groupby("cell"):
            sub = sub.sort_values("t")
            init, proposal = cell.split("/", 1)
            colour = PROPOSAL_COLOUR.get(proposal, INK)
            ax.plot(
                sub.t, sub.cost,
                color=colour,
                marker=PROPOSAL_MARKER.get(proposal, "o"),
                linestyle=INIT_STYLE.get(init, "-"),
                markersize=3.0, linewidth=1.0,
                markeredgecolor="white", markeredgewidth=0.4,
                zorder=3,
            )
            # Direct label at the right end, in whichever panel the line is
            # legible: the full panel labels only cells that resolve there, and
            # cells inside the dense band are labelled in the detail panel
            # instead. Without this split, five labels collide at the foot of
            # the full panel. Every series is labelled exactly once, which is
            # the required relief for the low-contrast slot.
            last = sub.iloc[-1]
            in_dense = dense[0] <= last.cost <= dense[1]
            if ylim is None:
                show = not in_dense
            else:
                show = in_dense
            if show:
                short = ("CW" if init.startswith("Clark") else "rnd") + "/" + proposal
                ax.annotate(
                    short, (last.t, last.cost),
                    textcoords="offset points", xytext=(3, 1),
                    fontsize=5, color=colour, zorder=4,
                )
        ax.set_xscale("log")
        ax.set_xlabel("wall-clock (s, setup + SA)")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=7, color=INK_MUTED)
    ax_full.set_ylabel("mean cost")

    # Legends: identity is carried by colour+marker (proposal) and linestyle
    # (init), each named, so neither depends on colour alone.
    from matplotlib.lines import Line2D
    prop_handles = [
        Line2D([], [], color=PROPOSAL_COLOUR[p], marker=PROPOSAL_MARKER[p],
               linestyle="-", markersize=3.0, linewidth=1.0, label=p)
        for p in PROPOSAL_ORDER if p in set(g.cell.str.split("/").str[1])
    ]
    init_handles = [
        Line2D([], [], color=INK_MUTED, linestyle="-", linewidth=1.0, label="random init"),
        Line2D([], [], color=INK_MUTED, linestyle="--", linewidth=1.0, label="Clarke-Wright init"),
    ]
    ax_full.legend(handles=prop_handles, fontsize=6, frameon=False, loc="upper right")
    ax_zoom.legend(handles=init_handles, fontsize=6, frameon=False, loc="upper right")

    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", dpi=300)
    plt.close(fig)


def equal_time_table(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Each cell's cost interpolated at the reference cell's wall-clock per budget.

    The equal-steps table flatters whichever proposal is cheapest per step; this
    reads the same runs at matched wall-clock instead, by linear interpolation of
    cost against log-time within each cell's own anytime curve. Cells whose curve
    does not span the reference time are reported as NaN rather than extrapolated.
    """
    g = df.groupby(["cell", "budget"]).agg(
        cost=("final_cost", "mean"), t=("total_time", "mean")
    ).reset_index()
    ref = g[g.cell == reference].sort_values("t")
    rows = []
    for budget in sorted(df.budget.unique()):
        rt = ref[ref.budget == budget]
        if rt.empty:
            continue
        t_ref = float(rt.t.iloc[0])
        for cell, sub in g.groupby("cell"):
            sub = sub.sort_values("t")
            if t_ref < sub.t.min() or t_ref > sub.t.max():
                cost = float("nan")
            else:
                cost = float(np.interp(np.log(t_ref), np.log(sub.t), sub.cost))
            rows.append({
                "ref_budget": budget, "t_ref_s": t_ref, "cell": cell,
                "cost_at_t_ref": cost,
            })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="A-bis factorial analysis")
    p.add_argument("--res_dir", type=str, default="res/abis")
    p.add_argument("--csv_name", type=str, default="factorial")
    p.add_argument("--n_instances", type=int, default=1000)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--reference", type=str, default="random/lgsa")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--boot_seed", type=int, default=0)
    p.add_argument(
        "--init_temp", type=float, default=1.0,
        help="Which reported column to analyse: 1.0 = primary (shared schedule). "
             "Pass a stage-B value to analyse the strongest-baseline column.",
    )
    p.add_argument("--suffix", type=str, default="", help="Suffix for output filenames.")
    p.add_argument(
        "--schedules", type=str, default="",
        help="Per-cell schedule selection for the strongest-baseline column, as "
             "'init:proposal:init_temp:stop_temp' entries separated by commas. "
             "Each (init, proposal) is taken at its own tuned schedule instead of "
             "one shared --init_temp. Values must come from the tuning slice, "
             "never from the reported set.",
    )
    args = p.parse_args()

    csv_path = os.path.join(args.res_dir, f"{args.csv_name}.csv")
    costs_dir = os.path.join(args.res_dir, "costs")
    df = load(csv_path, args.n_instances, args.offset)

    if args.schedules:
        wanted = {}
        for entry in args.schedules.split(","):
            init, proposal, it, st = entry.split(":")
            wanted[(init, proposal)] = (float(it), float(st))
        missing = set()
        parts = []
        for (init, proposal), (it, st) in wanted.items():
            sub = df[
                (df.init == init) & (df.proposal == proposal)
                & np.isclose(df.init_temp, it) & np.isclose(df.stop_temp, st)
            ]
            if sub.empty:
                missing.add((init, proposal, it, st))
            parts.append(sub)
        if missing:
            raise SystemExit(f"No rows for requested schedules: {sorted(missing)}")
        df = pd.concat(parts)
    else:
        df = df[(df.init_temp == args.init_temp) | (df.proposal == "lgsa")]

    tab = cost_table(df)
    wide = tab.pivot(index="cell", columns="budget", values="mean")
    spread = tab.pivot(index="cell", columns="budget", values="spread")
    print("=== mean cost over seeds ===")
    print(wide.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n=== inter-seed spread (max - min) ===")
    print(spread.to_string(float_format=lambda v: f"{v:.4f}"))

    boot = bootstrap_table(df, costs_dir, args.reference, args.n_boot, args.boot_seed)
    print(f"\n=== paired per-instance bootstrap vs {args.reference} ===")
    print("(negative delta = cell is BETTER than the reference)")
    print(boot.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    eq = equal_time_table(df, args.reference)
    print(f"\n=== equal wall-clock: cost at {args.reference}'s time per budget ===")
    print(eq.pivot(index="cell", columns="ref_budget", values="cost_at_t_ref").to_string(
        float_format=lambda v: f"{v:.4f}"))

    sfx = args.suffix
    out_tab = os.path.join(args.res_dir, f"cost_table{sfx}.csv")
    out_boot = os.path.join(args.res_dir, f"bootstrap{sfx}.csv")
    out_eq = os.path.join(args.res_dir, f"equal_time{sfx}.csv")
    tab.to_csv(out_tab, index=False)
    boot.to_csv(out_boot, index=False)
    eq.to_csv(out_eq, index=False)
    anytime_plot(df, os.path.join(args.res_dir, f"anytime{sfx}"))
    print(f"\nWrote {out_tab}, {out_boot}, {out_eq}, {args.res_dir}/anytime{sfx}.[pdf|png]")


if __name__ == "__main__":
    main()

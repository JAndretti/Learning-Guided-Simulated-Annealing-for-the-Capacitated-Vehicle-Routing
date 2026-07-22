"""
Anytime frontier: LG-SA vs blind SA, one independent fully-annealed run per budget.

Unlike ``trace_curve.py``, which samples a single long annealing run and whose
intermediate points are therefore mid-anneal states, every point here is the
final cost of a complete run whose temperature schedule was compressed to that
budget. Points are directly comparable to each other and to the headline table.

Input:  res/<FOLDER>/bench/scaling_frontier.csv  (bench_scaling.py --baseline)
Output: res/<FOLDER>/bench/frontier_<dim>.{pdf,png}

Usage
-----
    uv run eval/bench_scaling.py --FOLDER BEST --dims 100 \
        --steps 100,300,1000,3000,10000,30000,100000 --batch_size 10000 \
        --dtype float16 --DATA_SOURCE neuopt --baseline --tag frontier
    uv run eval/plot_frontier.py --FOLDER BEST --dim 100 --hgs 15.563
"""

import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_MODEL = "#0072B2"
C_BASE = "#D55E00"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--FOLDER", type=str, default="BEST")
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--hgs", type=float, default=None, help="Reference cost for a horizontal line")
    return p


def main() -> None:
    args = build_parser().parse_args()
    src = f"res/{args.FOLDER}/bench/scaling_frontier.csv"
    d = pd.read_csv(src)
    d = d[d["dim"] == args.dim].sort_values("steps")
    if d.empty:
        raise SystemExit(f"no rows for dim={args.dim} in {src}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.3))
    series = [
        ("cost_baseline", "sa_time_baseline", C_BASE, "SA"),
        ("cost", "sa_time", C_MODEL, "LG-SA"),
    ]

    for ax, xcol, xlabel in (
        (axes[0], "steps", "Step budget $T$ (complete run)"),
        (axes[1], None, "Wall-clock time (s, 10,000 instances)"),
    ):
        ax.grid(True, which="major", linewidth=0.4, color="0.9", zorder=0)
        ax.set_axisbelow(True)
        if args.hgs is not None:
            ax.axhline(args.hgs, color="0.45", lw=0.8, ls=(0, (1, 2)), zorder=1)
            ax.annotate(
                "HGS",
                xy=(0.995, args.hgs),
                xycoords=("axes fraction", "data"),
                xytext=(0, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=6.5,
                color="0.45",
            )
        for ycol, tcol, colour, label in series:
            if ycol not in d:
                continue
            x = d[xcol] if xcol else d[tcol]
            ax.plot(
                x, d[ycol], color=colour, lw=1.3, marker="o", ms=3, label=label, zorder=3
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean cost")

    axes[0].legend(frameon=False, loc="upper right")
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.annotate(
            tag,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-30, 9),
            textcoords="offset points",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout(pad=0.4)
    out = f"res/{args.FOLDER}/bench/frontier_{args.dim}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"-> {out}.pdf / .png")


if __name__ == "__main__":
    main()

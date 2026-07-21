"""
Figures and LaTeX tables for the bench_*.py sweeps.

Reads only the CSVs written by bench_scaling / bench_dims_cpu / bench_batch /
bench_precision, so a figure can be restyled without re-running anything. Every
section is skipped silently if its CSV is absent.

Figures are camera-ready: no titles (the caption lives in the .tex), Times-like
serif matching the AAAI body text, single-column width by default, vector PDF
with embedded Type-42 fonts.

Usage
-----
    uv run eval/bench_plots.py --FOLDER BEST
    uv run eval/bench_plots.py --FOLDER BEST --only scaling,batch
    uv run eval/bench_plots.py --FOLDER BEST --width 3.3

Outputs (under res/<FOLDER>/bench/):
    scaling_cost.pdf/.png     cost vs. steps, one curve per dimension
    scaling_cost_rel.pdf/.png same, each dimension normalised by its own shortest budget
    scaling_time.pdf/.png     runtime vs. steps, log-log, one curve per dimension
    dims_cpu.pdf/.png         sequential runtime vs. N, one curve per budget
    batch_scaling.pdf/.png    total time and time per instance vs. batch size
    tables.tex                all three tables, ready to \\input
"""

import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    FuncFormatter,
    LogFormatterSciNotation,
    NullFormatter,
)

# ============================================================================
# STYLE
# ============================================================================

# Okabe-Ito in an order whose adjacent pairs all clear the CVD separation floor
# (validated: worst adjacent pair deltaE 9.6 deutan / 20.0 normal-vision).
# Later slots sit below 3:1 contrast on white, which is why every series also
# carries a direct label and its own marker — identity is never colour-alone,
# and the figures survive greyscale printing.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

C_TOTAL = PALETTE[0]
C_PER_INST = PALETTE[1]

INK = "#1a1a1a"  # primary text
INK_MUTED = "#5c5c5c"  # annotations, secondary labels
GRID = "#e6e6e6"

# AAAI is two-column: 3.3in fills one column, 6.9in spans both.
COL_WIDTH = 3.3
FULL_WIDTH = 6.9

# Nimbus Roman is a metric-compatible Times clone; STIX matches it for maths, so
# axis labels sit in the same face as the body text of the paper.
RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.4,
    "ytick.minor.size": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.0,
    "lines.markeredgewidth": 0.0,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    # Type 42 (TrueType) keeps text selectable and searchable in the submitted PDF;
    # Type 3 is rejected by several camera-ready checkers.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def new_fig(width: float = COL_WIDTH, height: float = 2.1, **kw):
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(width, height), **kw)
    return fig, ax


def style_axes(ax) -> None:
    ax.grid(True, which="major", linewidth=0.4, color=GRID, zorder=0)
    ax.set_axisbelow(True)


def log_axis(ax, which: str) -> None:
    """Log scale with 10^k major labels only — minor decades stay unlabelled."""
    if which == "x":
        ax.set_xscale("log")
        axis = ax.xaxis
    else:
        ax.set_yscale("log")
        axis = ax.yaxis
    axis.set_major_formatter(LogFormatterSciNotation(base=10))
    axis.set_minor_formatter(NullFormatter())


def save(fig, out: str, tight: bool = True) -> None:
    # Multi-panel figures use constrained layout instead, which tight_layout warns about.
    if tight:
        fig.tight_layout(pad=0.25)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"  -> {out}.pdf / .png")


def read(base: str, name: str) -> pd.DataFrame | None:
    path = os.path.join(base, f"{name}.csv")
    if not os.path.exists(path):
        print(f"skip {name}: no {path}")
        return None
    # A resumed sweep can duplicate a cell; each consumer keeps the last row.
    return pd.read_csv(path)


def fmt_time(seconds: float) -> str:
    """1234.5 -> '$20$m $35$s' (paper style); short runs stay in seconds."""
    if seconds < 60:
        return f"${seconds:.1f}$s"
    m, s = divmod(round(seconds), 60)
    h, m = divmod(m, 60)
    return f"${h}$h ${m:02d}$m ${s:02d}$s" if h else f"${m}$m ${s:02d}$s"


def tex_num(x: int) -> str:
    """LaTeX-safe thousands separator: 10000 -> '10{,}000'."""
    return f"{x:,}".replace(",", "{,}")


def label_line_end(ax, x, y, text: str, colour: str, dx: float = 4.0) -> None:
    """Direct label at a curve's right end — the series' non-colour identity cue."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=7,
        color=colour,
        annotation_clip=False,
    )


# ============================================================================
# SCALING (cost + time vs. steps and dimension)
# ============================================================================


def _scaling_panel(ax, df, dims, ycol: str, *, relative: bool = False) -> None:
    for i, dim in enumerate(dims):
        g = df[df.dim == dim].sort_values("steps")
        y = g[ycol] / g[ycol].iloc[0] if relative else g[ycol]
        # Strongest hue to the largest N: that is the headline series, and the
        # small-N curves are the near-flat ones that can afford a lighter slot.
        colour = PALETTE[(len(dims) - 1 - i) % len(PALETTE)]
        ax.plot(
            g.steps,
            y,
            color=colour,
            marker=MARKERS[(len(dims) - 1 - i) % len(MARKERS)],
            zorder=3 + i,
        )
        label_line_end(ax, g.steps.iloc[-1], y.iloc[-1], f"$N{{=}}{dim}$", colour)
    log_axis(ax, "x")
    ax.set_xlabel("LG-SA steps")
    # Headroom on the right for the direct labels (log scale -> multiplicative).
    ax.set_xlim(right=df.steps.max() * 4.0)


def plot_scaling(df: pd.DataFrame, base: str, suffix: str = "", width: float = COL_WIDTH) -> None:
    df = df.drop_duplicates(subset=["dim", "steps", "batch_size", "dtype"], keep="last")
    dims = sorted(df.dim.unique())

    fig, ax = new_fig(width)
    style_axes(ax)
    _scaling_panel(ax, df, dims, "cost")
    ax.set_ylabel("Mean cost")
    save(fig, os.path.join(base, f"scaling_cost{suffix}"))

    # Same data, each dimension divided by its own shortest-budget cost. Absolute
    # cost grows with N, so on a shared linear axis the small-N curves look flat
    # even where they improve by several percent; the relative view makes the
    # convergence profiles comparable. Pick whichever the text argues.
    fig, ax = new_fig(width)
    style_axes(ax)
    _scaling_panel(ax, df, dims, "cost", relative=True)
    ax.set_ylabel("Cost relative to shortest budget")
    save(fig, os.path.join(base, f"scaling_cost_rel{suffix}"))

    fig, ax = new_fig(width)
    style_axes(ax)
    _scaling_panel(ax, df, dims, "sa_time")
    log_axis(ax, "y")
    ax.set_ylabel("Time (s)")
    save(fig, os.path.join(base, f"scaling_time{suffix}"))


def table_scaling(df: pd.DataFrame) -> str:
    df = df.drop_duplicates(subset=["dim", "steps", "batch_size", "dtype"], keep="last")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Scaling Analysis: Cost and Time vs. Steps and Dimension.}",
        "\\label{tab:scaling_results}",
        "\\begin{tabular}{c r c c}",
        "\\toprule",
        "\\textbf{Dim ($N$)} & \\textbf{Steps} & \\textbf{Cost} & \\textbf{Time (s)} \\\\",
        "\\midrule",
    ]
    dims = sorted(df.dim.unique())
    for j, dim in enumerate(dims):
        g = df[df.dim == dim].sort_values("steps")
        lines.append(f"\\multirow{{{len(g)}}}{{*}}{{{dim}}}")
        for _, r in g.iterrows():
            lines.append(f"  & {int(r.steps):,} & {r.cost:.3f} & {r.sa_time:.2f} \\\\")
        if j < len(dims) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ============================================================================
# SEQUENTIAL DIMENSION SWEEP
# ============================================================================


def plot_dims_cpu(df: pd.DataFrame, base: str, suffix: str = "", width: float = COL_WIDTH) -> None:
    # Repeats are independent instances of the same size: average them, and show
    # the spread as a band so the reader can see whether it matters.
    g = (
        df.groupby(["dim", "steps"])["sa_time"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "sa_time", "std": "sa_time_std"})
    )

    fig, ax = new_fig(width)
    style_axes(ax)
    budgets = sorted(g.steps.unique())
    for i, steps in enumerate(budgets):
        s = g[g.steps == steps].sort_values("dim")
        # Larger budget -> stronger hue, matching the scaling figures.
        slot = (len(budgets) - 1 - i) % len(PALETTE)
        colour = PALETTE[slot]
        ax.plot(s.dim, s.sa_time, color=colour, marker=MARKERS[slot], zorder=3 + i)
        if s.sa_time_std.notna().any():
            ax.fill_between(
                s.dim,
                s.sa_time - s.sa_time_std.fillna(0),
                s.sa_time + s.sa_time_std.fillna(0),
                color=colour,
                alpha=0.15,
                lw=0,
                zorder=2,
            )
        # Self-describing direct labels — no legend box needed for two series.
        label_line_end(
            ax, s.dim.iloc[-1], s.sa_time.iloc[-1], f"${tex_num(int(steps))}$ steps", colour
        )

    log_axis(ax, "x")
    log_axis(ax, "y")
    ax.set_xlabel("Problem dimension $N$")
    ax.set_ylabel("Time per instance (s)")
    ax.set_xlim(right=g.dim.max() * 3.6)
    save(fig, os.path.join(base, f"dims_cpu{suffix}"))

    for steps in sorted(g.steps.unique()):
        s = g[g.steps == steps]
        print(
            f"  {int(steps):,} steps: mean {s.sa_time.mean():.2f}s, "
            f"max {s.sa_time.max():.2f}s at N={int(s.loc[s.sa_time.idxmax(), 'dim'])}"
        )


# ============================================================================
# BATCH SCALING
# ============================================================================


def plot_batch(df: pd.DataFrame, base: str, suffix: str = "", width: float = COL_WIDTH) -> None:
    """
    Two stacked panels sharing the batch-size axis, *not* a dual-axis plot:
    total time and time-per-instance have different units and different shapes,
    and overlaying them on twin y-scales lets the aspect ratio invent crossings
    that carry no meaning. Stacked panels keep both curves readable and let the
    per-instance minimum — the actual finding — sit on its own scale.
    """
    df = df.drop_duplicates(subset=["dim", "steps", "batch_size", "dtype"], keep="last")
    df = df.sort_values("batch_size")

    plt.rcParams.update(RC)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(width, 3.0), sharex=True, layout="constrained"
    )
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.02, wspace=0.0)
    for ax in (ax_top, ax_bot):
        style_axes(ax)

    ax_top.plot(df.batch_size, df.sa_time, color=C_TOTAL, marker="o", zorder=3)
    ax_top.set_ylabel("Total time (s)")
    ax_top.set_ylim(bottom=0)

    ax_bot.plot(df.batch_size, df.time_per_instance_ms, color=C_PER_INST, marker="s", zorder=3)
    ax_bot.set_ylabel("Time per instance (ms)")
    ax_bot.set_xlabel("Batch size (parallel instances)")
    ax_bot.set_ylim(bottom=0)
    ax_bot.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))

    # The peak-efficiency point is the finding; label it rather than leaving the
    # reader to pick it off the axis.
    best = df.loc[df.time_per_instance_ms.idxmin()]
    ax_bot.annotate(
        f"${best.time_per_instance_ms:.1f}$ ms at $B{{=}}{tex_num(int(best.batch_size))}$",
        xy=(best.batch_size, best.time_per_instance_ms),
        xytext=(6, 7),
        textcoords="offset points",
        fontsize=7,
        color=INK_MUTED,
    )
    save(fig, os.path.join(base, f"batch_scaling{suffix}"), tight=False)


def table_batch(df: pd.DataFrame) -> str:
    df = df.drop_duplicates(subset=["dim", "steps", "batch_size", "dtype"], keep="last")
    df = df.sort_values("batch_size")
    dim = int(df.dim.iloc[0])
    steps = int(df.steps.iloc[0])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Impact of batch size on GPU execution time. \\textbf{Setup:} "
        f"$N={dim}$, ${tex_num(steps)}$ steps.}}",
        "\\label{tab:batch_scaling}",
        "\\begin{tabular}{r r c}",
        "\\toprule",
        "\\textbf{Batch Size} & \\textbf{Time (s)} & \\textbf{Time per Instance (ms)} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"{int(r.batch_size):,} & {r.sa_time:.2f} & {r.time_per_instance_ms:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ============================================================================
# PRECISION
# ============================================================================


def table_precision(df: pd.DataFrame) -> str:
    df = df.drop_duplicates(subset=["dim", "steps", "batch_size", "precision"], keep="last")
    order = {"float32": 0, "float16": 1, "bfloat16": 2}
    df = df.sort_values("precision", key=lambda s: s.map(order).fillna(9))
    ref = df[df.precision == "float32"]
    ref_cost = float(ref.cost.iloc[0]) if not ref.empty else float(df.cost.iloc[0])
    ref_time = float(ref.sa_time.iloc[0]) if not ref.empty else float(df.sa_time.iloc[0])

    dim = int(df.dim.iloc[0])
    n = int(df.batch_size.iloc[0])
    steps = int(df.steps.iloc[0])
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\caption{Effect of inference-time numerical precision on LG-SA performance. "
        f"Results on Nazari $N{{=}}{dim}$, ${tex_num(n)}$ instances, ${tex_num(steps)}$ steps. "
        "Cost differences are reported relative to the \\texttt{float32} baseline.}",
        "\\label{tab:precision_study}",
        "\\begin{tabular}{@{}l r r r r@{}}",
        "\\toprule",
        "Precision & Final Cost & $\\Delta$ Cost & Time & Speedup \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        is_ref = r.precision == "float32"
        delta = "---" if is_ref else f"${r.cost - ref_cost:+.4f}$"
        lines.append(
            f"\\texttt{{{r.precision}}} & ${r.cost:.4f}$ & {delta} & "
            f"{fmt_time(r.sa_time)} & ${ref_time / r.sa_time:.2f}\\times$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--FOLDER", type=str, default="BEST")
    p.add_argument(
        "--only",
        type=lambda s: set(s.replace(" ", "").split(",")),
        default=None,
        help="Comma-separated subset of scaling,dims_cpu,batch,precision",
    )
    p.add_argument("--tag", type=str, default="", help="CSV name suffix used by the bench scripts")
    p.add_argument(
        "--base",
        type=str,
        default=None,
        help="Directory holding the CSVs (default res/<FOLDER>/bench)",
    )
    p.add_argument(
        "--width",
        type=float,
        default=COL_WIDTH,
        help=f"Figure width in inches ({COL_WIDTH} = one AAAI column, {FULL_WIDTH} = full page)",
    )
    args = p.parse_args()

    base = args.base or os.path.join("res", args.FOLDER, "bench")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No bench directory at {base} — run the bench_*.py scripts first")
    suffix = f"_{args.tag}" if args.tag else ""

    def wanted(name: str) -> bool:
        return args.only is None or name in args.only

    tables = []

    if wanted("scaling"):
        df = read(base, f"scaling{suffix}")
        if df is not None and not df.empty:
            print("scaling:")
            plot_scaling(df, base, suffix, args.width)
            tables.append(table_scaling(df))

    if wanted("dims_cpu"):
        df = read(base, f"dims_cpu{suffix}")
        if df is not None and not df.empty:
            print("dims_cpu:")
            plot_dims_cpu(df, base, suffix, args.width)

    if wanted("batch"):
        df = read(base, f"batch{suffix}")
        if df is not None and not df.empty:
            print("batch:")
            plot_batch(df, base, suffix, args.width)
            tables.append(table_batch(df))

    if wanted("precision"):
        df = read(base, f"precision{suffix}")
        if df is not None and not df.empty:
            print("precision:")
            tables.append(table_precision(df))

    if tables:
        out = os.path.join(base, f"tables{suffix}.tex")
        with open(out, "w") as f:
            f.write("\n".join(tables))
        print(f"\nLaTeX tables -> {out}")


if __name__ == "__main__":
    main()

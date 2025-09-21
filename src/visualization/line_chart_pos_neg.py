import math
import os
import json
from typing import Sequence, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

def plot_triplet_lines_for_predicates_explicit(
    df: pd.DataFrame,
    pos_predicates: Sequence[str],          # e.g. ["?A1=A2(+)","?A1>A3(+)","..."]
    neu_predicates: Sequence[str],          # e.g. ["?A1=A2","?A1>A3","..."]
    neg_predicates: Sequence[str],          # e.g. ["?A1=A2(-)","?A1>A3(-)","..."]
    *,
    base_labels: Optional[Sequence[str]] = None,  # titles per subplot; defaults to neu_predicates
    dataset: str = "overall",
    actions: Sequence[str] = ("zero-shot",),
    llms: Optional[Sequence[str]] = None,
    aggregate: str = "mean",                # "mean" | "median"
    show_values: bool = False,              # annotate points only on first subplot
    ncols: int = 3,
    figsize: Tuple[int, int] = (16, 9),
    index_starts_at: int = 1,
    ylabel: str = "Value",
    colors: dict = None,                    # {'pos':'#...','neu':'#...','neg':'#...'}
    markers: dict = None,                   # {'pos':'o','neu':'s','neg':'D'}
    ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Grid of line charts. For each predicate triplet (pos/neu/neg columns),
    draw three lines across LLM index.

    df must contain: ['dataset','action','llm', ... given predicate columns ...]
    """
    # ---- validation
    if not (len(pos_predicates) == len(neu_predicates) == len(neg_predicates)):
        raise ValueError("pos_predicates, neu_predicates, neg_predicates must have the same length.")
    for cols in (pos_predicates, neu_predicates, neg_predicates):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in df: {missing}")

    # ---- setup
    if colors is None:
        colors = {'pos': '#4C78A8', 'neu': '#9E9E9E', 'neg': '#E45756'}
    if markers is None:
        markers = {'pos': 'o', 'neu': 's', 'neg': 'D'}
    if base_labels is None:
        base_labels = list(neu_predicates)

    sub = df.copy()
    if "dataset" in sub.columns:
        sub = sub[sub["dataset"] == dataset]
    if "action" in sub.columns:
        sub = sub[sub["action"].isin(list(actions))]

    # infer llm order if not provided
    if llms is None:
        if "action" in df.columns and (df["action"] == "zero-shot").any():
            source = df[(df["dataset"] == dataset) & (df["action"] == "zero-shot")]
        else:
            source = df[df["dataset"] == dataset]
        seen = []
        for m in source["llm"].tolist():
            if m not in seen:
                seen.append(m)
        llms = seen or list(dict.fromkeys(df["llm"].tolist()))

    xs = list(range(index_starts_at, index_starts_at + len(llms)))

    n = len(neu_predicates)
    nrows = 2 if n > 3 else 1
    ncols_eff = math.ceil(n / nrows)
    n_index_lines = max(1, ncols_eff)
    bottom_band = {1: 0.30, 2: 0.40, 3: 0.50}.get(n_index_lines, 0.45)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows + 1, ncols_eff, figure=fig, height_ratios=[1]*nrows + [bottom_band])

    axes = np.empty((nrows, ncols_eff), dtype=object)
    for r in range(nrows):
        for c in range(ncols_eff):
            axes[r, c] = fig.add_subplot(gs[r, c])
    axes_flat = axes.ravel()

    # ---- aggregation helper
    def agg_per_llm(colname: str):
        vals = []
        for m in llms:
            arr = pd.to_numeric(sub.loc[sub["llm"] == m, colname], errors="coerce").dropna()
            if arr.empty:
                vals.append(np.nan)
            else:
                vals.append(float(arr.median() if aggregate == "median" else arr.mean()))
        return vals

    # ---- plot each triplet
    for i, (pcol, ncol, mcol, label) in enumerate(zip(pos_predicates, neu_predicates, neg_predicates, base_labels)):
        ax = axes_flat[i]

        y_pos = agg_per_llm(pcol)
        y_neu = agg_per_llm(ncol)
        y_neg = agg_per_llm(mcol)

        ax.plot(xs, y_pos, marker=markers['pos'], color=colors['pos'], label="(+)")
        ax.plot(xs, y_neu, marker=markers['neu'], color=colors['neu'], label="Neutral")
        ax.plot(xs, y_neg, marker=markers['neg'], color=colors['neg'], label="(–)")

        if show_values and i == 0:
            for x, y, col in zip(xs, y_pos, [colors['pos']]*len(xs)):
                if pd.notna(y): ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 6),
                                            textcoords="offset points", ha="center", fontsize=8, color=col)
            for x, y, col in zip(xs, y_neu, [colors['neu']]*len(xs)):
                if pd.notna(y): ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 6),
                                            textcoords="offset points", ha="center", fontsize=8, color=col)
            for x, y, col in zip(xs, y_neg, [colors['neg']]*len(xs)):
                if pd.notna(y): ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 6),
                                            textcoords="offset points", ha="center", fontsize=8, color=col)

        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in xs])
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if i == 0:
            proxies = [
                Line2D([], [], color=colors['pos'], marker=markers['pos'], linestyle='-'),
                Line2D([], [], color=colors['neu'], marker=markers['neu'], linestyle='-'),
                Line2D([], [], color=colors['neg'], marker=markers['neg'], linestyle='-'),
            ]
            ax.legend(proxies, ["(+)", "(+/-)", "(–)"], frameon=False, loc="best")

    # hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # ---- bottom: index : LLM band (dynamic lines)
    legend_ax = fig.add_subplot(gs[nrows, :]); legend_ax.axis("off")
    llm_names = list(llms)
    space = 1
    # print(nrows)
    if nrows>1:
        tick_fs = 9
    else:
         tick_fs = 12
         space = 2
    # 
     # split into two lines (0–9, 10–end) with correct indices
    line1 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llm_names[:6]))
    line2 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llm_names[6:12], start=6))
    line3 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llm_names[12:], start=12))

    legend_ax.text(0.5, 0.5/space, line1, ha="center", va="center",
                   family="monospace", fontsize=tick_fs)
    legend_ax.text(0.5, 0.25/space, line2, ha="center", va="center",
                   family="monospace", fontsize=tick_fs)
    legend_ax.text(0.5, 0, line3, ha="center", va="center",
                   family="monospace", fontsize=tick_fs)
    
    # ys = np.linspace(0.75, 0.15, num=len(lines)) if lines else []
    # for y, line in zip(ys, lines):
    #     legend_ax.text(0.5, y, line, ha="center", va="center", family="monospace", fontsize=9)

    # fig.tight_layout(rect=(0.05, 0.12 + 0.02 * (n_lines - 1), 0.98, 0.96))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.96, bottom=0.04)
    fig.suptitle(f"LLMs consistency ({dataset} | {actions[0]})", fontsize=14, y=1.05)
    return fig, axes


from pathlib import Path
import re
import matplotlib.pyplot as plt

def _slug(s: str) -> str:
    """Filesystem-safe slug."""
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s)).strip("_")

def save_triplet_plots_for_all_datasets(
    *,
    df,
    pos_predicates,
    neu_predicates,
    neg_predicates,
    base_labels=None,
    actions=("classification",),
    llms=None,
    ncols=2,
    figsize=(14, 8),
    show_values=False,
    ylim=(0, 1),
    out_dir="plots/triplet_lines",
    dpi=200
):
    """
    For every dataset in df['dataset'], draw the triplet-lines grid and save as PNG.
    Returns a list of saved file paths.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    datasets = sorted(df["dataset"].dropna().unique().tolist())
    saved = []

    for ds in datasets:
        fig, axes = plot_triplet_lines_for_predicates_explicit(
            df=df,
            pos_predicates=pos_predicates,
            neu_predicates=neu_predicates,
            neg_predicates=neg_predicates,
            base_labels=base_labels,
            dataset=ds,
            actions=list(actions),
            llms=llms,
            ncols=ncols,
            figsize=figsize,
            show_values=show_values,
            ylim=ylim,
        )
        # fig.suptitle(f"Dataset: {ds}", fontsize=14, y=0.995)

        fname = f"pos-neg-{_slug(ds)}_{actions[0]}.png"
        fpath = out_path / fname
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(fpath))

    return saved


def main(config =None):

    root_dir = os.path.dirname(os.path.abspath(__name__))
    if config == None: 
        config = {
            "folder": os.path.join(root_dir, "output"),
            "out_dir": os.path.join(root_dir, "new_charts"),
            "time": "2025-09-22_00-41",
            "llms": None,
            "actions": ["zero-shot","wikidata", "fixing","classification"],
            "predicates": ["?A1=A2","?A1>A3","?A1>A4","?A1=A3+A4","?A3∅A4","?A4=A1|3"],
            "jccards":["J(A1-A2)","J(A1-A34)","J(A3-A4)","J(A4-A1|3)"]
        }   

    folder = config.get("folder", os.path.join(root_dir, "output"))
    out_dir = config.get("out_dir", os.path.join(root_dir, "new_charts"))
    time = config.get("time", "2025-09-22_00-41")

    llms_name = config.get("llms", None)
    if llms_name is None:
        llm_path = f"{root_dir}/data/llm_info.json"
        with open(llm_path, "r", encoding="utf-8") as f:
            llms_name = list(json.load(f).keys())

    df_summery = pd.read_csv(f"{folder}/summary_{time}.csv")
    summery_llms = df_summery["llm"].unique()
    llms = []
    for llm in llms_name:
        if llm in summery_llms:
            llms.append(llm)
    
    for action in ["zero-shot", "classification"]:
        paths = save_triplet_plots_for_all_datasets(
            df=df_summery,
            pos_predicates=["?A1=A2(+)","?A1>A3(+)","?A3∅A4(+)","?A1=A3+A4(+)"],
            neu_predicates=['?A1=A2','?A1>A3','?A3∅A4','?A1=A3+A4'],
            neg_predicates=["?A1=A2(-)","?A1>A3(-)","?A3∅A4(-)","?A1=A3+A4(-)"],
            base_labels=['?A1=A2','?A1>A3','?A3∅A4','?A1=A3+A4'],
            actions=[action],
            llms=llms,
            ncols=2,
            figsize=(14, 8),
            show_values=False,
            ylim=(0,1),
            out_dir=f"{out_dir}/pos_neg"
        )
        
        paths = save_triplet_plots_for_all_datasets(
            df=df_summery,
            pos_predicates=["?A1=A2(+)","?A1>A3(+)","?A3∅A4(+)","?A1=A3+A4(+)"],
            neu_predicates=['?A1=A2','?A1>A3','?A3∅A4','?A1=A3+A4'],
            neg_predicates=["?A1=A2(-)","?A1>A3(-)","?A3∅A4(-)","?A1=A3+A4(-)"],
            base_labels=['?A1=A2','?A1>A3','?A3∅A4','?A1=A3+A4'],
            actions=[action],
            llms=llms,
            ncols=2,
            figsize=(14, 8),
            show_values=False,
            ylim=(0,1),
            out_dir=f"{out_dir}/pos_neg"
        )


if __name__== "__main__":
    main()
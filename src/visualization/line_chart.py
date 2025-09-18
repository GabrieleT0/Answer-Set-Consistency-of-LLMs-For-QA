import os
from pathlib import Path
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools, matplotlib as mpl
from matplotlib.gridspec import GridSpec
from typing import Sequence, Optional, Tuple, List


def save_all_line_plots(
    *,
    df,
    predicate_cols,
    actions,
    llms,
    outdir="/line_chart",
    ncols=3,
    figsize=(16, 9),
    connect_lines=True,
    show_values=False,
    index_starts_at=1,
    show_action_palette=True,
    action_palette=None,
    dpi=200
):
    """
    Render and save a grid plot for each dataset found in df['dataset'].
    Returns a list of saved file paths.
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    datasets = sorted(df["dataset"].dropna().unique().tolist())
    saved_paths = []

    for ds in datasets:
        fig, axes, _ = plot_llms_for_predicates(
            df=df,
            predicate_cols=predicate_cols,
            dataset=ds,
            actions=actions,
            llms=llms,
            ncols=ncols,
            figsize=figsize,
            connect_lines=connect_lines,
            show_values=show_values,
            index_starts_at=index_starts_at,
            show_action_palette=show_action_palette,
            action_palette=action_palette
        )

        folder = Path(outdir)
        folder.mkdir(parents=True, exist_ok=True) 

        fig.savefig(f"{folder}/{ds}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        
# --- helpers from your snippet (unchanged) ---
def llm_order_from_df(df: pd.DataFrame, default_sort: bool = False) -> List[str]:
    if "llm" not in df.columns:
        raise KeyError("DataFrame must have column 'llm'.")
    base = df.copy()
    if "action" in base.columns and (base["action"] == "zero-shot").any():
        base = base[base["action"] == "zero-shot"]
    seen = []
    for llm in base["llm"].tolist():
        if llm not in seen:
            seen.append(llm)
    if not seen:
        seen = list(dict.fromkeys(df["llm"].tolist()))
    if default_sort:
        seen = sorted(seen)
    return seen

def filter_predicate(df: pd.DataFrame, predicate_col: str = "?A1=A2",
                     dataset: Optional[str] = None,
                     actions: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if predicate_col not in df.columns:
        raise KeyError(f"Column '{predicate_col}' not found.")
    sub = df.copy()
    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]
    if actions is not None:
        sub = sub[sub["action"].isin(list(actions))]
    sub = sub[["dataset", "action", "llm", predicate_col]].rename(columns={predicate_col: "value"})
    return sub

def values_in_llm_order(df_sub: pd.DataFrame, llms: Sequence[str], action: str) -> List[float]:
    one = df_sub[df_sub["action"] == action].set_index("llm")["value"].to_dict()
    return [one.get(llm, float("nan")) for llm in llms]

# --- NEW: grid function with index x-axis, bottom legend band, and value toggle ---
def plot_llms_for_predicates(
    df: pd.DataFrame,
    predicate_cols: Sequence[str],
    dataset: str = "overall",
    actions: Sequence[str] = ("fixing", "classification"),
    llms: Optional[Sequence[str]] = None,
    connect_lines: bool = True,
    show_values: bool = True,           # <— toggle numeric annotation on nodes
    titles: Optional[Sequence[str]] = None,
    ylabel: str = "Score",
    ncols: int = 3,
    figsize: Tuple[int, int] = (16, 9),
    show_action_palette: bool = True,                 # show one legend row for actions at top
    action_palette: Optional[dict] = None,            # {action: color}; overrides defaults
    index_starts_at: int = 1            # set to 0 if you prefer 0-based
) -> Tuple[plt.Figure, np.ndarray, dict]:
    """
    Grid plot of LLM performance for multiple predicates (one subplot per predicate).

    - X axis uses integer indices (1..N by default) for models.
    - A bottom band shows "index : LLM" mapping, split into 1–3 lines dynamically.
    - `show_values=True` annotates each node with its numeric value.
    """
    if llms is None:
        llms = llm_order_from_df(df[df["dataset"] == dataset])  # infer order

    # integer x positions (1..N by default)
    xs = list(range(index_starts_at, index_starts_at + len(llms)))

    n = len(predicate_cols)
    # prefer 2 rows when there are >3 subplots; else single row
    nrows = 2 if n > 3 else 1
    ncols_eff = math.ceil(n / nrows)

    # bottom legend band height depends on number of columns (and thus how many lines we’ll draw)
    n_index_lines = max(1, ncols_eff)
    bottom_band = {1: 0.30, 2: 0.40, 3: 0.50}.get(n_index_lines, 0.45)


    # Consistent color mapping for actions (used in all subplots)
    if action_palette is None:
        base_colors = mpl.rcParams['axes.prop_cycle'].by_key().get('color',
                        ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
        cyc = itertools.cycle(base_colors)
        action_colors = {act: next(cyc) for act in actions}
    else:
        action_colors = {act: action_palette.get(act, 'C0') for act in actions}



    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows + 1,            # +1 row for bottom index legend
        ncols_eff,            # no right gutter needed for this line-style plot
        figure=fig,
        height_ratios=[1]*nrows + [bottom_band]
    )

    axes = np.empty((nrows, ncols_eff), dtype=object)
    for r in range(nrows):
        for c in range(ncols_eff):
            axes[r, c] = fig.add_subplot(gs[r, c])
    axes_flat = axes.ravel()

    tidy_by_pred = {}
    for i, pred in enumerate(predicate_cols):
        ax = axes_flat[i]
        sub = filter_predicate(df, predicate_col=pred, dataset=dataset, actions=actions)
        if sub.empty:
            ax.set_visible(False)
            continue

        # tidy table for this predicate
        rows = []
        for act in actions:
            vals = values_in_llm_order(sub, llms, act)
            for nm, v, x in zip(llms, vals, xs):
                rows.append({"llm": nm, "action": act, "value": v, "x": x})
        tidy = pd.DataFrame(rows)
        tidy_by_pred[pred] = tidy

        for act in actions:
            part = tidy[tidy["action"] == act]
            ys = part["value"].tolist()
            col = action_colors[act]
            ax.scatter(xs, ys, color=col)      # ensure color matches global legend
            if connect_lines:
                ax.plot(xs, ys, color=col)
            if show_values and i == 0:         # << show labels ONLY on first subplot
                for x, y in zip(xs, ys):
                    if pd.notna(y):
                        ax.annotate(f"{y:.3f}", (x, y),
                                    textcoords="offset points", xytext=(0, 6),
                                    ha="center", fontsize=8, color=col)


        # x ticks show indices only (not names)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in xs], rotation=0)
        # ax.set_xlabel("Index")
        ax.set_ylabel(ylabel)
        ttl = (titles[i] if titles and i < len(titles) else f"{pred}")
        ax.set_title(ttl)
        # if len(actions) > 1:
        #     ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    
    # hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # ---------- Index : LLM legend (bottom band, dynamic line count) ----------
    llm_names = list(llms)
    legend_ax = fig.add_subplot(gs[nrows, :])  # bottom row spanning all columns
    legend_ax.axis("off")

    # decide how many lines (1..3) based on ncols_eff
    n_lines = max(1, min(3, ncols_eff))
    chunk = math.ceil(len(llm_names) / n_lines)
    lines = []
    start_idx = index_starts_at
    for i_line in range(n_lines):
        start = i_line * chunk
        end   = (i_line + 1) * chunk
        seg   = llm_names[start:end]
        if not seg:
            continue
        line = " | ".join(f"{idx:>2}: {name}"
                          for idx, name in enumerate(seg, start=start_idx + start))
        lines.append(line)

    # place the lines at vertically spaced positions in the band
    tick_fs = 9
    ys = np.linspace(0.75, 0.15, num=len(lines)) if lines else []
    for y, line in zip(ys, lines):
        legend_ax.text(0.5, y, line, ha="center", va="center",
                       family="monospace", fontsize=tick_fs)
    # -------------------------------------------------------------------------
        # ---------- Top-row predicate color legend (one line) ----------
    # ---------- Top-row ACTIONS legend (one line, colors match subplots) ----------
    if show_action_palette:
        from matplotlib.lines import Line2D
        action_proxies = [
            Line2D([], [], marker='o', linestyle='None',
                markersize=9, markerfacecolor=action_colors[a],
                markeredgecolor=action_colors[a])
            for a in actions
        ]
        # Leave a bit of top room; reuse same rect in tight_layout below
        top_rect = (0.05, 0.12 + 0.02 * (max(1, min(3, ncols_eff)) - 1), 0.98, 0.90)
        fig.tight_layout(rect=top_rect)

        fig.legend(
            action_proxies, list(actions),
            title="Actions",
            loc="lower center", bbox_to_anchor=(0.5, 0.95),
            ncol=len(actions), frameon=False,
            handlelength=1.2, handletextpad=0.6, columnspacing=1.2
        )
    else:
        top_rect = (0.05, 0.12 + 0.02 * (max(1, min(3, ncols_eff)) - 1), 0.9, 0.9)
    # -----------------------------------------------------------------------------

    # ---------------------------------------------------------------

    # final layout (leave room at bottom for the band)
    fig.tight_layout(rect=(0.05, 0.12 + 0.02 * (n_lines - 1), 0.98, 0.96))
    fig.suptitle(f"LLMs consistency ({dataset})", fontsize=14, y=1.05)

    return fig, axes, tidy_by_pred


def main(config =None):

    root_dir = os.path.dirname(os.path.abspath(__name__))
    if config == None: 
        config = {
            "folder": os.path.join(root_dir, "output"),
            "out_dir": os.path.join(root_dir, "new_charts"),
            "time": "2025-09-17_15-25",
            "llms": None,
            "actions": ["zero-shot","wikidata", "fixing","classification"],
            "predicates": ["?A1=A2","?A1>A3","?A1>A4","?A1=A3+A4","?A3∅A4","?A4=A1|3"],
            "jccards":["J(A1-A2)","J(A1-A34)","J(A3-A4)","J(A4-A1|3)"]
        }   
    predicates = config.get("predicates",["?A1=A2","?A1>A3","?A1>A4","?A1=A3+A4","?A3∅A4","?A4=A1|3"])
    actions = config.get("actions",["zero-shot","wikidata", "fixing","classification"])
    jccards_col = config.get("jccards", ["J(A1-A2)","J(A1-A34)","J(A3-A4)","J(A4-A1|3)"])
    folder = config.get("folder", os.path.join(root_dir, "output"))
    out_dir = config.get("out_dir", os.path.join(root_dir, "new_charts"))
    time = config.get("time", "2025-09-17_15-25")
    actions = config.get("actions", )
    llms = config.get("llms", None)
    if llms is None:
        llms = ['llama3.1:8b',
                'llama3.1:70b',
                'deepseek-chat',
                'deepseek-reasoner',
                'grok-3-mini',
                'gemini-2.0-flash',
                'gemini-2.5-flash',
                'gemini-2.5-pro',
                'gpt-4.1-2025-04-14',
                'gpt-4.1-mini-2025-04-14',
                'gpt-4.1-nano-2025-04-14',
                'gpt-4o',
                'o3',
                'gpt-oss:20b',
                'gpt-5-nano',
                'gpt-5-mini',
                'gpt-5']

    df_summery = pd.read_csv(f"{folder}/summary_{time}.csv")
    
    paths = save_all_line_plots(
        df=df_summery,
        predicate_cols=predicates,
        actions=actions,
        llms=llms,
        outdir=f"{out_dir}/line_chart",
        ncols=3,
        figsize=(16, 9),
        connect_lines=True,
        show_values=False
    )

    paths = save_all_line_plots(
        df=df_summery,
        predicate_cols=jccards_col,
        actions=actions,
        llms=llms,
        outdir=f"{out_dir}/line_chart_jaccard",
        ncols=3,
        figsize=(16, 9),
        connect_lines=True,
        show_values=False
)


if __name__== "__main__":
    main()
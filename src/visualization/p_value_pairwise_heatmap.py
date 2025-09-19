import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def build_square_matrix_for_pred(df, action, dataset, predicate, llms):
    # Filter rows
    sub = df[(df["action"] == action) &
             (df["dataset"] == dataset) &
             (df["predicate"] == predicate)].copy()

    # Coerce numeric on LLM columns
    value_cols = [c for c in sub.columns if c not in ("action", "dataset", "predicate", "llm")]
    sub[value_cols] = sub[value_cols].apply(pd.to_numeric, errors="coerce")

    # Base square matrix
    mat = sub.set_index("llm")[value_cols]

    # Keep only columns that are in llms, then reindex rows/cols to your given order
    cols_in = [c for c in llms if c in mat.columns]
    mat = mat[cols_in]
    mat = mat.reindex(index=llms, columns=llms)  # <-- rank/order by your `llms` list
    return mat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

def plot_heatmap_panel_2xN(
    mats: list[pd.DataFrame],          # 4 or 6 square DataFrames (same LLM order)
    titles: list[str],                 # same length as mats
    llm_names: list[str],              # order used for rows/cols
    show_confidence: bool = False,     # if True, plot 1 - p (confidence)
    cmap: str = "Reds",
    tick_fs: int = 14,                 # bigger tick font
    title_fs: int = 14,                # bigger title font
    wspace: float = 0.05,              # wider col spacing
    hspace: float = 0.15,              # wider row spacing
    cell_in: float = 0.34,             # inches per cell side (scales figure size)
    low_thr: float = 0.05,
    high_thr: float = 0.95,
    under_color: str = "white",
    over_color: str = "black",   # None -> use cmap's deepest red
):
    """
    2x2 or 2x3 heatmaps with a centered horizontal colorbar and a two-line legend below it.
    """
    # Thresholded normalization: show white < low_thr, deepest red > high_thr
    norm = Normalize(vmin=low_thr, vmax=high_thr)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_under(under_color)
    cmap_obj.set_over(cmap_obj(1.0) if over_color is None else over_color)
    # ---- Validate inputs
    k = len(mats)
    assert k in (4, 6), "Provide exactly 4 or 6 matrices."
    assert len(titles) == k, "titles must match mats length."
    n = len(llm_names)

    # Reindex each matrix to the provided llm_names order and choose what to plot
    mats = [M.reindex(index=llm_names, columns=llm_names) for M in mats]
    datas = [(1.0 - M) if show_confidence else M for M in mats]
    masks = [D.isna() for D in datas]

    heat_cols = 2 if k == 4 else 3   # 2x2 or 2x3
    if k == 4:
        tick_fs -= 2
        title_fs -= 2

    # ---- Figure size: scale with matrix dimension and number of heatmaps
    per_subplot = max(6.5, n * cell_in)
    fig_w = per_subplot * heat_cols
    fig_h = per_subplot * 2 + 1.0    # extra for cbar + legend row
    fig = plt.figure(figsize=(fig_w, fig_h))

    # ---- Grid: add 2 extra rows: one for colorbar, one for legend
    # height ratios: [row1, row2, cbar, legend]
    gs = GridSpec(4, heat_cols,
                  figure=fig,
                  height_ratios=[1, 1, 0.06, 0.11],
                  wspace=wspace, hspace=hspace)

    axes, ims = [], []
    # norm = Normalize(vmin=0.0, vmax=1.0)

    for i in range(k):
        r = i // heat_cols
        c = i % heat_cols
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        D = datas[i].copy()
        D[masks[i]] = np.nan

        # im = ax.imshow(D, norm=norm, cmap=cmap, interpolation="nearest")
        im = ax.imshow(D, norm=norm, cmap=cmap_obj, interpolation="nearest")
        ims.append(im)

        # Square cells
        try:
            ax.set_box_aspect(1.0)
        except Exception:
            ax.set_aspect("equal", adjustable="box")

        # Ticks at every index; show labels only bottom row / left column
        ax.set_xticks(range(n)); ax.set_yticks(range(n))

        if r == 1:  # bottom row
            ax.set_xticklabels([i for i in range(n)], rotation=90, fontsize=tick_fs)
        else:
            ax.set_xticklabels([])

        if c == 0:  # left column
            ax.set_yticklabels([i for i in range(n)], fontsize=tick_fs)
        else:
            ax.set_yticklabels([])

        ax.set_title(titles[i], fontsize=title_fs, pad=3)
        ax.grid(False)

        # thin borders (optional)
        for x in range(n + 1):
            ax.axvline(x - 0.5, color="lightgray", linewidth=0.4)
        for y in range(n + 1):
            ax.axhline(y - 0.5, color="lightgray", linewidth=0.4)

    # ---- Colorbar centered under heatmaps (spans all heatmap columns)
    cax = fig.add_subplot(gs[2, :])
    # cbar = fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    cbar = fig.colorbar(ims[0], cax=cax, orientation="horizontal", extend="both")
    cbar.set_label("Confidence (1 − p)" if show_confidence else "p-value (0 → low, 1 → high)",
                fontsize=tick_fs, labelpad=2)
    # cbar.ax.tick_params(labelsize=tick_fs, length=3)

    cbar.ax.xaxis.set_label_position('bottom')             # move label to top
    cbar.ax.xaxis.set_ticks_position('bottom')          # keep ticks at bottom
    cbar.ax.tick_params(axis='x', labelsize=tick_fs - 4, labelbottom=True,     # show only bottom tick labels
                        labeltop=False, length=3)
    cbar.set_ticks(np.linspace(low_thr, high_thr, 5))
    # ---- Two-line index→LLM legend directly under the colorbar
    legend_ax = fig.add_subplot(gs[3, :])
    legend_ax.axis("off")

    # split into two lines (0–9, 10–end) with correct indices
    line1 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llm_names[:6]))
    line2 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llm_names[6:12], start=6))
    line3 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llm_names[12:], start=12))

    legend_ax.text(0.5, 0.8, line1, ha="center", va="center",
                   family="monospace", fontsize=tick_fs)
    legend_ax.text(0.5, 0.4, line2, ha="center", va="center",
                   family="monospace", fontsize=tick_fs)
    legend_ax.text(0.5, 0, line3, ha="center", va="center",
                   family="monospace", fontsize=tick_fs)

    # tighter outer margins so plots fill the canvas
    fig.subplots_adjust(left=0.06, right=0.985, top=0.96, bottom=0.04)
    return fig, axes

def save_heatmap_panels_by_combo(
    df: pd.DataFrame,
    llms: list[str],
    predicates: list[str],
    out_dir: str = "panels",
    show_confidence: bool = True,    # plot 1 - p by default
    cmap: str = "Reds",
    dpi: int = 300,
    **panel_kwargs,                  # passed through to plot_heatmap_panel_2xN (e.g., low_thr=0.05, high_thr=0.95)
):
    """
    For each (action, dataset) in df, build 4/6 matrices (one per predicate) and save a 2xN panel.
    Expects df columns: ['action','dataset','predicate','llm', <LLM columns...>] with p-values.
    Returns list of saved file paths.
    """

    def _safe(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s)).strip("_")

    def _latexify_emptyset(s: str) -> str:
        return s.replace("∅", r"$\emptyset$")

    def build_square_matrix_for_pred(df_in, action, dataset, predicate, llm_order):
        sub = df_in[(df_in["action"] == action) &
                    (df_in["dataset"] == dataset) &
                    (df_in["predicate"] == predicate)].copy()
        if sub.empty:
            return pd.DataFrame(index=llm_order, columns=llm_order, dtype=float)

        value_cols = [c for c in sub.columns if c not in ("action", "dataset", "predicate", "llm")]
        sub[value_cols] = sub[value_cols].apply(pd.to_numeric, errors="coerce")

        mat = sub.set_index("llm")[value_cols]
        # keep only requested LLMs/columns, then reindex to full order
        cols_in = [c for c in llm_order if c in mat.columns]
        mat = mat[cols_in]
        mat = mat.reindex(index=llm_order, columns=llm_order)
        return mat

    os.makedirs(out_dir, exist_ok=True)
    saved = []

    combos = df[["action", "dataset"]].drop_duplicates().itertuples(index=False, name=None)
    for action, dataset in combos:
        # build per-predicate matrices (RAW p-values)
        mats = [build_square_matrix_for_pred(df, action, dataset, p, llms) for p in predicates]

        # skip if all-NaN across all mats
        if all(m.isna().all().all() for m in mats):
            continue

        titles = [f"{_latexify_emptyset(p)} | {dataset} | {action}" for p in predicates]

        # plot (uses your existing plot_heatmap_panel_2xN)
        fig, _ = plot_heatmap_panel_2xN(
            mats=mats,
            titles=titles,
            llm_names=llms,
            show_confidence=show_confidence,
            cmap=cmap,
            **panel_kwargs
        )

        ds_dir = os.path.join(out_dir, _safe(dataset))
        os.makedirs(ds_dir, exist_ok=True)
        fname = f"{_safe(dataset)}__{_safe(action)}.png"
        fpath = os.path.join(ds_dir, fname)
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(fpath)

    return saved


def main(config = None):
    root_dir = os.path.dirname(os.path.abspath(__name__))
    if config == None: 
        config = {
            "folder": os.path.join(root_dir, "output"),
            "out_dir": os.path.join(root_dir, "new_charts"),
            "time": "2025-09-17_15-25",
            "llms": None
        }   
    folder = config.get("folder", os.path.join(root_dir, "output"))
    out_dir = config.get("out_dir", os.path.join(root_dir, "new_charts"))
    time = config.get("time", "2025-09-17_15-25")
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
        
    df_p_value = pd.read_csv(f"{folder}/p_value_matrices_{time}.csv")
    predicates = ["?A1=A2","?A1>A3","?A1>A4","?A1=A3+A4","?A3∅A4","?A4=A1|3"]
    paths = save_heatmap_panels_by_combo(
        df=df_p_value,
        llms=llms,
        predicates=predicates,
        out_dir=f"{out_dir}/p_value_heatmap2X3",
        show_confidence=True,       # plot 1 - p
        cmap="Reds",
        low_thr=0.05, high_thr=0.95, over_color="black"  # if your plot function supports thresholds
    )
    # 2x2
    predicates = ["?A1=A2","?A1>A3","?A3∅A4","?A4=A1|3"]
    paths = save_heatmap_panels_by_combo(
        df=df_p_value,
        llms=llms,
        predicates=predicates,
        out_dir=f"{out_dir}/p_value_heatmap2X2",
        show_confidence=True,       # plot 1 - p
        cmap="Reds",
        low_thr=0.05, high_thr=0.95, over_color="black"  # if your plot function supports thresholds
    )


      

if __name__ == "__main__":
    main()

    
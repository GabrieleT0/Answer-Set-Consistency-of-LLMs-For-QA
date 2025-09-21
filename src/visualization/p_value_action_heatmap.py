import os
import json
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

def plot_heatmap_actions_1X2_vertical(
    p_value_matrix: pd.DataFrame,   # df with ['llm','action'] + columns (p-values)
    actions: list[str],             # exactly two, e.g. ["fixing","classification"]
    dataset: str,
    columns: list[str],             # predicate columns like ["p(A1=A2)", ...]
    llms: list[str],                # LLM order (x-axis)
    *,
    show_confidence: bool = True,  # True -> plot 1 - p
    cmap: str = "Reds",
    low_thr: float = 0.05,
    high_thr: float = 0.95,
    under_color: str = "white",     # < low_thr
    over_color: str = "black",      # > high_thr
    tick_fs: int = 11,
    title_fs: int = 14,
    cell_in: float = 0.34,          # inches per cell
    wspace: float = 0.20,           # gap between two plots
    cbar_height_ratio: float = 0.055  # shrink colorbar (row height ratio)
):
    if len(actions) != 2:
        raise ValueError("Provide exactly two actions.")

    # Ensure numeric
    sub = p_value_matrix[['llm','action'] + columns].copy()
    for c in columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # Build two matrices per action, then TRANSPOSE so:
    # rows = predicates (columns), cols = llms
    mats = []
    for act in actions:
        df_a = sub[sub['action'] == act][['llm'] + columns]
        mat = df_a.set_index('llm').reindex(index=llms, columns=columns).T  # transpose here
        mats.append(mat)

    # Display as p or 1-p
    disp = [(1.0 - M) if show_confidence else M for M in mats]

    n_rows = len(columns)   # predicates on y-axis
    n_cols = len(llms)      # llms on x-axis

    # Figure size to keep cells near-square
    sub_w = max(5.5, n_cols * cell_in)
    sub_h = max(4.8, n_rows * cell_in)
    fig_w = sub_w * 2 + 0.8
    fig_h = sub_h +0.5

    rc = {"text.usetex": False, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, fig_h))
        # 2 rows: (0) shrunken colorbar, (1) heatmaps
        gs = GridSpec(2, 2, figure=fig,
                      height_ratios=[cbar_height_ratio, 1.0],
                      wspace=wspace, hspace=0.10)

        # Thresholded normalization & cmap: bar shows only [low_thr, high_thr]
        norm = Normalize(vmin=low_thr, vmax=high_thr)
        base = plt.get_cmap(cmap)
        try:
            cmap_obj = base.with_extremes(under=under_color, over=over_color)
        except AttributeError:
            cmap_obj = base
            cmap_obj.set_under(under_color)
            cmap_obj.set_over(over_color)

        ims, axes = [], []
        for i, (act, M) in enumerate(zip(actions, disp)):
            ax = fig.add_subplot(gs[1, i])
            axes.append(ax)

            im = ax.imshow(M, norm=norm, cmap=cmap_obj, interpolation="nearest")
            ims.append(im)

            # Square cells: aspect = rows / cols (predicates / llms)
            try:
                ax.set_box_aspect(n_rows / n_cols)
            except Exception:
                ax.set_aspect('equal', adjustable='box')

            ax.set_xticks(range(n_cols))
            ax.set_yticks(range(n_rows))
            ax.set_xticklabels([i for i in range(n_cols)], rotation=0, fontsize=tick_fs)
            if i == 0:  # left column
                ax.set_yticklabels(columns, fontsize=tick_fs)
            else:
                ax.set_yticklabels([])

            # light borders
            for x in range(n_cols + 1):
                ax.axvline(x - 0.5, color="lightgray", linewidth=0.35)
            for y in range(n_rows + 1):
                ax.axhline(y - 0.5, color="lightgray", linewidth=0.35)

            ax.set_title(f"{act} | {dataset}", fontsize=title_fs, pad=6, fontweight="semibold")
            ax.grid(False)

        # Shrunken top colorbar spanning both heatmaps
        cax = fig.add_subplot(gs[0, :])

        cbar = fig.colorbar(ims[0], cax=cax, orientation="horizontal", extend="both")
        cbar.set_label(
            "Confidence (1 − p)" if show_confidence else "p-value (0 → low, 1 → high)",
            fontsize=tick_fs, labelpad=4
        )
        # Put label on top; ticks below
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.tick_params(axis='x', labelbottom=True, labeltop=False,
                            labelsize=max(tick_fs-1, 8), length=2)
        cbar.set_ticks(np.linspace(low_thr, high_thr, 5))

                    
        # Left subplot: show LLM names once
        # ax.set_yticklabels(columns, fontsize=tick_fs)

        # split into two lines (0–9, 10–end) with correct indices
        line1 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llms[:6]))
        line2 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llms[6:12], start=6))
        line3 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llms[12:], start=12))
        
        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis("off")
        legend_ax.text(0.5, 0.1, line1, ha="center", va="center",
                    family="monospace", fontsize=tick_fs)
        legend_ax.text(0.5, 0.05, line2, ha="center", va="center",
                    family="monospace", fontsize=tick_fs)
        legend_ax.text(0.5, 0, line3, ha="center", va="center",
                    family="monospace", fontsize=tick_fs)

        # Margins: extra bottom for long LLM x-labels
        fig.subplots_adjust(left=0.12, right=0.995, top=0.92, bottom=0.22)

    return fig, axes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

def plot_heatmap_actions_1xN(
    p_value_matrix: pd.DataFrame,          # df with ['llm','action'] + columns (p-values)
    actions: list[str],                    # length 2 or 3 (order = plot order left→right)
    dataset: str,
    columns: list[str],                    # predicate columns like ["p(A1=A2)", ...]
    llms: list[str],                       # row order (y-axis)
    *,
    show_confidence: bool = False,         # True -> plot 1 - p
    cmap: str = "Reds",
    low_thr: float = 0.05,
    high_thr: float = 0.95,
    under_color: str = "white",            # < low_thr
    over_color: str = "black",             # > high_thr
    tick_fs: int = 12,
    title_fs: int = 16,
    cell_in: float = 0.34,                 # inches per cell
    wspace: float = 0.0,                  # gap between subplots
    cbar_height_ratio: float = 0.10,       # relative height of colorbar row
    hide_row_labels_on_nonleft: bool = True,
    cbar_width_ratio: float = 0.06,   # <- add this
):
    """
    Plot 1×N heatmaps (N=2 or 3): rows=LLMs, cols=predicates. All share the same color scale.
    - Only the left plot shows y-axis (LLM) labels (unless disabled).
    - Colorbar is compact at the bottom with label above the bar.
    """
    k = len(actions)
    if k not in (2, 3):
        raise ValueError("actions must have length 2 or 3.")

    # Ensure numeric p-values
    sub = p_value_matrix[['llm', 'action'] + columns].copy()
    for c in columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # Build matrices (rows=LLMs, cols=predicates) — one per action in the given order
    mats = []
    for act in actions:
        df_a = sub[sub['action'] == act][['llm'] + columns]
        mat = df_a.set_index('llm').reindex(index=llms, columns=columns)
        mats.append(mat)

    # Values to display
    # disp = [(1.0 - M) if show_confidence else M for M in mats]
    disp = [(1.0 - M) if show_confidence else M for M in mats]
    n_rows, n_cols = len(llms), len(columns)

    # Figure size to keep cells square-ish
    sub_w = max(5.0, n_cols * cell_in)
    sub_h = max(5.0, n_rows * cell_in)
    fig_w = sub_w * k + 0.6 + (k-1) * 0.1      # small extra width
    fig_h = sub_h + 1.15                       # room for colorbar

    rc = {"text.usetex": False, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, fig_h))

        # GridSpec: first row = heatmaps, second row = compact colorbar
        gs = GridSpec(1, k + 1, figure=fig, width_ratios=[1.0]*k + [cbar_width_ratio],
              wspace=wspace)


        # Thresholded normalization & colormap [0.05–0.95], with under/over colors
        norm = Normalize(vmin=low_thr, vmax=high_thr)
        base = plt.get_cmap(cmap)
        try:
            cmap_obj = base.with_extremes(under=under_color, over=over_color)
        except AttributeError:
            cmap_obj = base
            cmap_obj.set_under(under_color)
            cmap_obj.set_over(over_color)

        ims, axes = [], []
        for i, (act, M) in enumerate(zip(actions, disp)):
            ax = fig.add_subplot(gs[0, i])
            axes.append(ax)

            im = ax.imshow(M, norm=norm, cmap=cmap_obj, interpolation="nearest")
            ims.append(im)

            # Square cells: aspect = rows / cols
            try:
                ax.set_box_aspect(n_rows / n_cols)
            except Exception:
                ax.set_aspect('equal', adjustable='box')

            # Ticks & labels
            ax.set_xticks(range(n_cols)); ax.set_yticks(range(n_rows))
            ax.set_xticklabels(columns, rotation=90, fontsize=tick_fs)
            if hide_row_labels_on_nonleft and i > 0:
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels(llms, fontsize=tick_fs)

            # light cell borders
            for x in range(n_cols + 1):
                ax.axvline(x - 0.5, color="lightgray", linewidth=0.35)
            for y in range(n_rows + 1):
                ax.axhline(y - 0.5, color="lightgray", linewidth=0.35)

            ax.set_title(f"{act} | {dataset}", fontsize=title_fs, pad=6, fontweight="semibold")
            ax.grid(False)

        # Compact bottom colorbar spanning all heatmaps
        cax = fig.add_subplot(gs[0, -1])   # last column = colorbar
        cbar = fig.colorbar(ims[0], cax=cax, orientation="vertical", extend="both")
        cbar.set_label(
            "Confidence (1 − p)" if show_confidence else "p-value (0 → low, 1 → high)",
            fontsize=tick_fs, labelpad=10
        )
        cbar.ax.yaxis.set_label_position('right')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.tick_params(axis='y', labelsize=tick_fs, length=3)
        cbar.set_ticks(np.linspace(low_thr, high_thr, 5))


        cbar.ax.tick_params(axis='x', labelbottom=True, labeltop=False,
                            labelsize=max(tick_fs-1, 9), length=2)
        cbar.set_ticks(np.linspace(low_thr, high_thr, 5))

        # Margins: leave room on the left for LLM names
        left_margin = 0.26 if (not hide_row_labels_on_nonleft or k >= 1) else 0.10
        fig.subplots_adjust(left=left_margin, right=0.99, top=0.93, bottom=0.14)

    return fig, axes


def save_all_dataset_plots_1X3(
    p_value_matrix: pd.DataFrame,
    actions: list[str],
    columns: list[str],
    llms: list[str],
    out_dir: str = "plots",
    show_confidence: bool = True,
    **kwargs,                     # pass through to plot_heatmap_actions_1xN
):
    """
    For each dataset in p_value_matrix, generate a 1xN heatmap panel and save it.
    """
    os.makedirs(out_dir, exist_ok=True)

    datasets = p_value_matrix['dataset'].unique()
    paths = []
    for ds in datasets:
        df_temp = p_value_matrix[p_value_matrix['dataset'] == ds]

        fig, axes = plot_heatmap_actions_1xN(
            df_temp,
            actions=actions,
            dataset=ds,
            columns=columns,
            llms=llms,
            show_confidence=show_confidence,
            **kwargs
        )

        fname = f"{ds.replace(' ', '_')}_heatmaps.png"
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(fpath)
        print(f"Saved: {fpath}")

    return paths

import os
import matplotlib.pyplot as plt

def save_all_dataset_plots_vertical(
    p_value_matrix: pd.DataFrame,
    actions: list[str],             # must be length 2
    columns: list[str],
    llms: list[str],
    out_dir: str = "plots_vertical",
    show_confidence: bool = True,
    **kwargs,                       # pass-through to plot_heatmap_actions_1X2_vertical
):
    """
    For each dataset in p_value_matrix, generate a 1x2 vertical heatmap panel and save it.
    """
    if len(actions) != 2:
        raise ValueError("plot_heatmap_actions_1X2_vertical requires exactly 2 actions")

    os.makedirs(out_dir, exist_ok=True)

    datasets = p_value_matrix['dataset'].unique()
    paths = []
    for ds in datasets:
        df_temp = p_value_matrix[p_value_matrix['dataset'] == ds]

        fig, axes = plot_heatmap_actions_1X2_vertical(
            df_temp,
            actions=actions,
            dataset=ds,
            columns=columns,
            llms=llms,
            show_confidence=show_confidence,
            **kwargs
        )

        fname = f"{ds.replace(' ', '_')}_vertical.png"
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(fpath)
        print(f"Saved: {fpath}")

    return paths




def main(config = None):
    root_dir = os.path.dirname(os.path.abspath(__name__))
    if config == None: 
        config = {
            "folder": os.path.join(root_dir, "output"),
            "out_dir": os.path.join(root_dir, "new_charts"),
            "time": "2025-09-22_00-41",
            "llms": None
        }   
        
    folder = config.get("folder", os.path.join(root_dir, "output"))
    out_dir = config.get("out_dir", os.path.join(root_dir, "new_charts"))
    time = config.get("time", "2025-09-22_00-41")

    actions = ["fixing","classification","wikidata"]
    predicates = ["?A1=A2","?A1>A3","?A1>A4","?A1=A3+A4","?A3∅A4","?A4=A1|3"]
    columns=["p(A1=A2)", "p(A1=A3+A4)", "p(A1>A3)", "p(A1>A4)", "p(A3∅A4)", "p(A4=A1|3)"]
    
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

    p_value_matrix = df_summery[["action", "dataset", "llm"] + columns]
    

    paths = save_all_dataset_plots_1X3(
        p_value_matrix,
        actions=actions,
        columns=columns,
        llms=llms,
        out_dir=f"{out_dir}/p_value_heatmap_actions_1x3",
        cmap="Reds",
        low_thr=0.05, high_thr=0.95,
        under_color="white", over_color="black",
        tick_fs=10, title_fs=12,
        cell_in=0.34, wspace=0.0,
        hide_row_labels_on_nonleft=True,
        )
    
    paths = save_all_dataset_plots_vertical(
        p_value_matrix,
        actions=["fixing", "classification"],
        columns=columns,
        llms=llms,
        out_dir=f"{out_dir}/p_value_heatmap_actions_1x2",
        cmap="Reds",
        low_thr=0.05, high_thr=0.95,
        under_color="white", over_color="black",
        tick_fs=11, title_fs=14,
        cell_in=0.34, wspace=0.22, cbar_height_ratio=0.045,
        )




if __name__ == "__main__":
    main()


import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import json
from matplotlib.gridspec import GridSpec


def draw_llm_bubble_grid(
    df: pd.DataFrame,
    predicates: list,                # list[str] -> one subplot per predicate
    llms: list,                      # sorted LLM names (display order)
    dataset: str,                    # dataset filter in df["dataset"]
    actions: list,                   # list[str] -> filter df["action"] and aggregate
    bubble_sizes: dict,              # {llm: area or "small"/"medium"/"large"}
    release_dates: dict,             # {llm: "YYYY-MM-DD" | datetime | {"release_date": "..."}}
    *,
    date_format: str = "%Y-%m-%d",
    families: dict | None = None,    # {llm: "OpenAI"/"Google"/"Meta"/"xAI"/"DeepSeek"/...}
    legend_position: str = "top",    # "top" or "bottom" for the OUTSIDE legend
    annotate: bool = True,
    annotate_rotation: int = 0,
    label_vnudges_px: int = 0,
    jitter_same_day: bool = True,
    jitter_days: float = 0.0,
    ncols: int = 3,                  # e.g., 2 -> 2x?, 3 -> 2x3 for 6 preds, etc.
    figsize: tuple = (15, 8),
    title_prefix: str = "",          # optional prefix for each subplot title
    ylabel: str = "Performance",
    bucket_sizes: dict | None = None # map for named sizes -> area
):
    """
    Create a grid of bubble scatter plots:
      - 1 subplot per predicate in `predicates`.
      - x-axis = release date (from `release_dates`)
      - y-axis = aggregated metric per (llm, predicate) over selected `actions`
      - bubble size = `bubble_sizes[llm]` (points^2) or bucket name ('small'/'medium'/'large')

    Data expectations for `df`:
      - Must contain columns: 'dataset', 'llm', 'action', and each predicate name in `predicates`.
      - Each predicate column contains numeric metric values.

    Aggregation:
      - For each (llm, predicate), we take the mean over rows that match the given dataset & actions.
    """
    if bucket_sizes is None:
        bucket_sizes = {"small": 80, "medium": 200, "large": 420}

    # --- helpers -------------------------------------------------------------
    def _to_dt(d):
        if isinstance(d, datetime):
            return d
        if isinstance(d, dict) and "release_date" in d:
            d = d["release_date"]
        return datetime.strptime(d, date_format)

    # Parse/normalize release dates for all llms
    parsed_dates = {}
    for m in llms:
        if m not in release_dates:
            raise KeyError(f"Missing release date for model: {m}")
        parsed_dates[m] = _to_dt(release_dates[m]["release_date"])

    # Normalize bubble sizes (allow direct float or named buckets)
    def _area_for(m):
        s = bubble_sizes.get(m, "medium")
        if isinstance(s, (int, float)):
            return float(s)
        return float(bucket_sizes.get(str(s), bucket_sizes["medium"]))

    # Filter df to dataset/actions once
    mask = (df["dataset"] == dataset) & (df["action"].isin(actions))
    base = df.loc[mask].copy()
    if base.empty:
        raise ValueError("No data after filtering. Check dataset/actions filter.")

    # Build values per predicate as list aligned with llms
    def _values_for_predicate(pred_col: str):
        # aggregate mean over actions for each llm
        agg = (base.groupby("llm")[pred_col]
                    .mean()
                    .reindex(llms))  # align to provided order
        if agg.isna().any():
            # fill missing llm with NaN-safe behavior (drop later)
            pass
        return agg.tolist()

    # color grouping
    def _group_key(model: str):
        return families.get(model, "Other") if families else "Series"

    # --- grid layout (dynamic: prefer 2 rows → 2x2 for 4, 2x3 for 6, etc.) ---
    n = len(predicates)

    # Prefer 2 rows when n > 3; otherwise 1 row
    nrows = 2 if n > 3 else 1
    ncols_eff = int(np.ceil(n / nrows))

    # Bottom index legend needs space proportional to how many lines we'll draw
    # We'll draw as many lines as columns (2 lines for 2x?, 3 lines for 3 columns)
    n_index_lines = max(1, ncols_eff)
    bottom_band = {1: 0.30, 2: 0.40, 3: 0.50}.get(n_index_lines, 0.45)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows + 1,                 # +1 for bottom index legend band
        ncols_eff + 1,             # +1 slim legend gutter on the right
        figure=fig,
        height_ratios=[1]*nrows + [bottom_band],
        width_ratios=[1]*ncols_eff + [0.18]
    )

    # Build subplot axes (exclude the rightmost legend gutter column)
    axes = np.empty((nrows, ncols_eff), dtype=object)
    for r in range(nrows):
        for c in range(ncols_eff):
            axes[r, c] = fig.add_subplot(gs[r, c])
    axes_flat = axes.ravel()

    # Right-side legend gutter axis (optional use)
    legend_ax = fig.add_subplot(gs[:nrows, -1])
    legend_ax.axis("off")


    legend_handles = {}
    legend_title = "" if families else "Series"

    for i, pred in enumerate(predicates):
        ax = axes_flat[i]
        vals = _values_for_predicate(pred)
        # Build grouped data
        groups = defaultdict(lambda: {"x": [], "y": [], "s": [], "labels": []})
        for m, v in zip(llms, vals):
            if pd.isna(v):
                continue
            d = parsed_dates[m]
            s = _area_for(m)
            groups[_group_key(m)]["x"].append(d)
            groups[_group_key(m)]["y"].append(float(v))
            groups[_group_key(m)]["s"].append(s)
            groups[_group_key(m)]["labels"].append(llms.index(m))

        # Jitter same-day points (across all groups in this subplot)
        if jitter_same_day:
            day_map = defaultdict(list)
            for gname, G in groups.items():
                for idx, d in enumerate(G["x"]):
                    day_map[d.date()].append((gname, idx))
            for _, items in day_map.items():
                if len(items) > 1:
                    k = len(items)
                    offsets = [((j - (k - 1)/2.0) * jitter_days) for j in range(k)]
                    for off, (gname, idx) in zip(offsets, items):
                        groups[gname]["x"][idx] = groups[gname]["x"][idx] + timedelta(days=off)

        # Plot
        texts = []
        for gname, G in groups.items():
            xs = [mdates.date2num(d) for d in G["x"]]
            sc = ax.scatter(xs, G["y"], s=G["s"], alpha=0.85,
                            label=None if gname in legend_handles else gname)
            if gname not in legend_handles:
                legend_handles[gname] = sc

            if annotate:
                for d, y, name in zip(G["x"], G["y"], G["labels"]):
                    t = ax.text(mdates.date2num(d), y, name,
                                fontsize=8, rotation=annotate_rotation,
                                ha="left", va="bottom", clip_on=False)
                    texts.append(t)

        # De-overlap labels by alternating vertical nudge
        if annotate and texts:
            fig.canvas.draw()
            texts.sort(key=lambda t: (t.get_position()[0], t.get_position()[1]))
            for j, t in enumerate(texts):
                dx = 0
                dy = ((-1) ** j) * label_vnudges_px
                xdata, ydata = t.get_position()
                xdisp, ydisp = ax.transData.transform((xdata, ydata))
                xnew, ynew = ax.transData.inverted().transform((xdisp + dx, ydisp + dy))
                t.set_position((xnew, ynew))

        # Ax formatting per subplot
        ttl = f"{title_prefix}{pred}" if title_prefix else str(pred)
        ax.set_title(ttl)
        ax.set_xlabel("Release date")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        # fig.autofmt_xdate()

    # Hide any unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    if legend_handles:
        # --- 1) Build uniform-size FAMILY proxies (filled dots), left side ---
        fam_labels = list(legend_handles.keys())

        # get colors from your plotted handles
        fam_colors = []
        for k in fam_labels:
            fc = legend_handles[k].get_facecolor()
            color = fc[0] if hasattr(fc, "__len__") and len(fc) else fc
            fam_colors.append(color)

        # uniform "medium" bubble area for all family nodes
        medium_area = float(bucket_sizes["medium"])
        fam_handles = [
            plt.scatter([], [], s=medium_area,
                        facecolors=color, edgecolors=color, linewidths=0.9)
            for color in fam_colors
        ]

        # --- 2) Bubble-size proxies (hollow), right side ---
        size_order   = ["small", "medium", "large"]   # order on the right
        size_labels  = []
        size_handles = []
        for nm in size_order:
            if nm in bucket_sizes:
                size_handles.append(
                    plt.scatter([], [], s=bucket_sizes[nm],
                                facecolors="none", edgecolors="black", linewidths=0.9)
                )
                size_labels.append(f"Size: {nm.capitalize()}")

        # --- 3) Combine: families (left) + sizes (right) in ONE ROW ---
        combined_handles = fam_handles + size_handles
        combined_labels  = fam_labels  + size_labels

        # (Optional) reserve a bit more top space so legend never overlaps subplots
        # fig.tight_layout(rect=(0.04, 0.10, 0.98, 0.90))

        # --- 4) Draw legend in one row; sizes will appear on the right ---
        if legend_position == "top":
            fig.legend(
                combined_handles, combined_labels,
                title=legend_title, frameon=False,
                loc="lower center", bbox_to_anchor=(0.5, 1.0),
                ncol=len(combined_handles),
                handlelength=1.2, handletextpad=0.6, columnspacing=1.2
            )
        else:
            fig.legend(
                combined_handles, combined_labels,
                title=legend_title, frameon=False,
                loc="upper center", bbox_to_anchor=(0.5, 0.08),
                ncol=len(combined_handles),
                handlelength=1.2, handletextpad=0.6, columnspacing=1.2
            )


    # ---------- Index : LLM legend (bottom band) ----------
    # ---------- Index : LLM legend (bottom band, dynamic line count) ----------
    llm_names = llms
    tick_fs = 9  # font size for the index legend lines

    # Split into up to three lines (0–5, 6–11, 12–end); adjust as you like
    line1 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llm_names[:6]))
    line2 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llm_names[6:12], start=6)) if len(llm_names) > 6 else ""
    line3 = " | ".join(f"{i:>2}: {name}" for i, name in enumerate(llm_names[12:], start=12)) if len(llm_names) > 12 else ""

    legend_ax = fig.add_subplot(gs[nrows, :])  # bottom row spanning all columns
    legend_ax.axis("off")

    # place the lines; tweak y positions if you use 1 or 2 lines only
    if line1:
        legend_ax.text(0.5, 0.6, line1, ha="center", va="center",
                    family="monospace", fontsize=tick_fs)
    if line2:
        legend_ax.text(0.5, 0.3, line2, ha="center", va="center",
                    family="monospace", fontsize=tick_fs)
    if line3:
        legend_ax.text(0.5, 0.01, line3, ha="center", va="center",
                    family="monospace", fontsize=tick_fs)
    # -------------------------------------------------------------------------

    date_locator   = mdates.AutoDateLocator()
    date_formatter = mdates.ConciseDateFormatter(date_locator)

    for r in range(nrows):
        for c in range(ncols_eff):
            ax_sub = axes[r, c]
            if r < nrows - 1:
                ax_sub.tick_params(axis='x', labelbottom=False)
            else:
                ax_sub.xaxis.set_major_locator(date_locator)
                ax_sub.xaxis.set_major_formatter(date_formatter)
                ax_sub.tick_params(axis='x', labelrotation=30)
                for lab in ax_sub.get_xticklabels():
                    lab.set_ha('right')


    # Give more bottom space so rotated ticks aren’t clipped
    fig.tight_layout(rect=(0.1, 0.1, 0.98, 0.99))  # leave ~10% at top for legend
    fig.suptitle(f"LLMs consistency over time & parameter size  ({dataset} | {actions[0]})", fontsize=14, y=1.1)



 # leave room for outside legend
    return fig, axes

def save_all_bubble_grids(
    *,
    df,
    predicates,
    llms,
    bubble_sizes,
    release_dates,
    families=None,
    # If None, infer from df['dataset'].unique()
    datasets=None,
    # If None, create one action-set per unique action in df['action'].
    # You can also pass a list of lists, e.g. [["zero-shot"], ["fix-llm-response"], ["zero-shot","fix-llm-response"]]
    actions=None,
    include_all_actions=False,   # if True and action_sets is None, add one extra set with all actions
    outdir="plots",
    ncols=3,
    figsize=(16, 9),
    date_format="%Y-%m-%d",
    ylabel="Score",
    title_prefix="",
    dpi=200,
):
    """
    Save a bubble-grid plot for each (dataset, action-set).
    Returns a list of saved file paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Infer datasets
    if datasets is None:
        datasets = sorted(df["dataset"].dropna().unique().tolist())

    for ds in datasets:
        for act in actions:
           
            # Draw
            fig, axes = draw_llm_bubble_grid(
                df=df,
                predicates=predicates,
                llms=llms,
                dataset=ds,
                actions=[act],
                bubble_sizes=bubble_sizes,
                release_dates=release_dates,
                families=families,
                ncols=ncols,
                figsize=figsize,
                date_format=date_format,
                title_prefix=title_prefix,
                ylabel=ylabel,
            )

            folder = Path(outdir)
            folder.mkdir(parents=True, exist_ok=True) 
            fig.savefig(f"{folder}/{ds}_{act}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            


def main(config =None):

    root_dir = os.path.dirname(os.path.abspath(__name__))
    if config == None: 
        config = {
            "folder": os.path.join(root_dir, "output"),
            "out_dir": os.path.join(root_dir, "new_charts"),
            "time": "2025-09-17_15-25",
            "llms": None,
            "datasets":["overall", "spinach", "qawiki","synthetic"],
            "actions":["zero-shot","wikidata", "fixing","classification"],
            "predicates": ["?A1=A2","?A1>A3","?A1>A4","?A1=A3+A4","?A3∅A4","?A4=A1|3"],
            "jccards":["J(A1-A2)","J(A1-A34)","J(A3-A4)","J(A4-A1|3)"]
        }   

    datasets = config.get("datasets",["overall", "spinach", "qawiki","synthetic"])
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
    

    model_size = {
        'llama3.1:8b': "small",
        'llama3.1:70b':'large', 
        'gpt-oss:20b': 'medium',
        'gpt-4o':'large',
        'gpt-4.1-2025-04-14':'large',
        'gpt-4.1-mini-2025-04-14':'medium',
        'gpt-4.1-nano-2025-04-14':'small',
        'gpt-5':'large',
        'gpt-5-mini':'medium',
        'gpt-5-nano':'small',
        'gemini-2.0-flash':'medium',
        'gemini-2.5-pro':'large',
        'gemini-2.5-flash':'medium',
        'grok-3-mini':'small',
        'deepseek-chat':'large',          # e.g., assume large (MoE active)
        'deepseek-reasoner':'large',
        'o3':'large',
    }

    families = {
        'gpt-4o':'OpenAI','gpt-4.1-2025-04-14':'OpenAI','gpt-4.1-mini-2025-04-14':'OpenAI','gpt-4.1-nano-2025-04-14':'OpenAI',
        'gpt-oss:20b':'OpenAI','gpt-5':'OpenAI','gpt-5-mini':'OpenAI','gpt-5-nano':'OpenAI','o3':'OpenAI',
        'gemini-2.0-flash':'Google','gemini-2.5-pro':'Google','gemini-2.5-flash':'Google',
        'llama3.1:8b':'Meta','llama3.1:70b':'Meta','grok-3-mini':'xAI',
        'deepseek-chat':'DeepSeek','deepseek-reasoner':'DeepSeek'
    }
    BASE_DIR = os.path.dirname(__file__)
    # print(BASE_DIR)
    with open(BASE_DIR + "/llm_info.json", "r") as f:
        llm_info = json.load(f)

    paths = save_all_bubble_grids(
        df=df_summery,
        predicates=predicates,
        llms=llms,
        bubble_sizes=model_size,
        release_dates=llm_info,
        families=families,
        datasets=datasets,       # or None to infer from df
        actions=actions,
        outdir=f"{out_dir}/bubble_scatter",
        ncols=3,
        figsize=(16,9),
        ylabel="Score",
        title_prefix=""
    )

    paths = save_all_bubble_grids(
        df=df_summery,
        predicates=predicates,
        llms=llms,
        bubble_sizes=model_size,
        release_dates=llm_info,
        families=families,
        datasets=datasets,       # or None to infer from df
        actions=actions,
        outdir=f"{out_dir}/bubble_scatter_jarccard",
        ncols=3,
        figsize=(16,9),
        ylabel="Score",
        title_prefix=""
    )


if __name__== "__main__":
    main()
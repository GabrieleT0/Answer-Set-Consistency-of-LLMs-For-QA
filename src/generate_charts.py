import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def create_action_barcharts(df, output_folder="../charts_barcharts", exclude_action="wikidata", metrics = ['J(A1-A2)', 'J(A1-A34)', 'J(A4-A1|3)', 'J(A3-A4)'], exclude_llms=None):
    """
    Generates one bar chart per dataset per action.
    X-axis: Jaccard metrics.
    Each bar: one LLM's value for that metric.
    Legend is placed below the plot in multiple columns.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics = ['J(A1-A2)', 'J(A1-A34)', 'J(A4-A1|3)', 'J(A3-A4)']
    sns.set(style="whitegrid")

    os.makedirs(output_folder, exist_ok=True)
    df = df[df['action'] != exclude_action]

    # Exclude LLMs if provided
    if exclude_llms is not None:
        df = df[~df['llm'].isin(exclude_llms)]

    datasets = df['dataset'].unique()

    for dataset in datasets:
        df_dataset = df[df['dataset'] == dataset]
        actions = df_dataset['action'].unique()

        for action in actions:
            df_action = df_dataset[df_dataset['action'] == action]
            llms = df_action['llm'].tolist()
            
            x = np.arange(len(metrics))  # positions for metrics
            n_models = len(llms)
            width = 0.8 / n_models  # auto-adjust bar width
            
            plt.figure(figsize=(14, 5))
            
            # Use a larger color palette to minimize repeats
            colors = sns.color_palette("tab20", n_colors=n_models)

            # Plot each LLM
            for i, llm in enumerate(llms):
                values = df_action[df_action['llm'] == llm][metrics].values.flatten()
                plt.bar(x + i*width, values, width=width, color=colors[i], label=llm, edgecolor='black')

            # Center X-axis labels under the groups
            total_width = width * n_models
            plt.xticks(x + total_width/2 - width/2, metrics)

            plt.ylim(0, 1)
            plt.ylabel("Jaccard Metric Value")
            plt.title(f"J-Metrics per LLM ({dataset} - {action})")
            
            # Legend below the chart in multiple columns
            n_cols = min(6, n_models)  # max 6 columns
            plt.legend(title="LLM", loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=n_cols)
            
            plt.tight_layout()
            # Save figure
            filename = f"{output_folder}/barchart_{dataset}_{action}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            print(f"Saved: {filename}")

def plot_pvalue_heatmaps_by_dataset_action(
    csv_file: str,
    cmap: str = "Reds_r",
    exclude_actions=("wikidata",),
    save: bool = True,
    out_dir: str = "../charts/action_pvalue_heatmaps",
    exclude_llms=None
):
    """
    Generate one heatmap per (dataset, action) pair.
      - Excludes rows with action in `exclude_actions`.
      - Colors only p <= 0.05 (intensity increases as p -> 0).
      - Cells with p > 0.05 are shown as white (masked).
      - Annotates only the significant cells (<= 0.05).
    Args:
        csv_file: path to CSV.
        cmap: a reversed sequential colormap string (e.g., "Reds_r","Blues_r","viridis_r").
        exclude_actions: actions to drop (default includes "wikidata").
        save: if True, save PNGs into out_dir; otherwise show them.
        out_dir: directory where PNGs are saved when save=True.
    """
    df = pd.read_csv(csv_file)
    # drop excluded actions
    df = df[~df["action"].isin(exclude_actions)]

    # Exclude LLMs if provided
    if exclude_llms is not None:
        df = df[~df['llm'].isin(exclude_llms)]

    # identify p-value columns
    pval_cols = [c for c in df.columns if c.startswith("p(")]
    if not pval_cols:
        raise ValueError("No p-value columns found (columns starting with 'p(').")

    # coerce to numeric
    df[pval_cols] = df[pval_cols].apply(pd.to_numeric, errors="coerce")

    if save:
        os.makedirs(out_dir, exist_ok=True)

    for dataset in df["dataset"].unique():
        actions = df[df["dataset"] == dataset]["action"].unique()
        for action in actions:
            subset = df[(df["dataset"] == dataset) & (df["action"] == action)]
            if subset.empty:
                continue

            # prepare matrix: rows = llm, cols = p-value features
            heat = subset.set_index("llm")[pval_cols].copy()

            # mask non-significant (p > 0.05) and NaNs
            mask = (heat > 0.05) | heat.isna()

            annot = heat.copy()
            for r in annot.index:
                for c in annot.columns:
                    val = annot.loc[r, c]
                    if pd.isna(val) or val > 0.05:
                        annot.loc[r, c] = ""         # non-significant
                    elif val <= 0.001:
                        annot.loc[r, c] = "≈0"       # very close to zero
                    else:
                        annot.loc[r, c] = f"{val:.3f}"  # keep 3 decimals

            # if there are no significant values, skip (or optionally draw an empty plot)
            if (~mask).sum().sum() == 0:
                print(f"No significant p-values (<=0.05) for {dataset} | {action} — skipping.")
                continue

            # plot size based on matrix dimensions
            figsize = (max(6, 1 * heat.shape[1] + 2), max(3, 0.4 * heat.shape[0] + 1))
            plt.figure(figsize=figsize)

            sns.heatmap(
                heat,
                mask=mask,
                annot=annot,
                fmt="",               # use provided annot strings
                cmap=cmap,            # reversed cmap so lower p -> more intense color
                vmin=0.0,
                vmax=0.05,            # map color scale to [0, 0.05]
                cbar_kws={"label": "p-value (only p ≤ 0.05 shown)"},
                linewidths=0.5,
                linecolor="lightgray",
                square=False,
            )

            plt.title(f"P-values (p ≤ 0.05 colored) — {dataset} | {action}", fontsize=10)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            if save:
                fname = f"{dataset}_{action}_pvalues.png".replace(" ", "_")
                plt.savefig(os.path.join(out_dir, fname), dpi=300)
                plt.close()
            else:
                plt.show()

def plot_square_heatmaps(csv_file, out_dir="heatmaps", cmap="Reds_r", exclude_llms=None):
    df = pd.read_csv(csv_file)

    # Identify LLM columns (everything after the 'llm' column)
    llm_cols = df.columns.tolist()
    llm_start = llm_cols.index("llm") + 1
    llm_cols = llm_cols[llm_start:]

    if exclude_llms is not None:
        df = df[~df["llm"].isin(exclude_llms)]
        llm_cols = [c for c in llm_cols if c not in exclude_llms]

    os.makedirs(out_dir, exist_ok=True)

    # Group by predicate → dataset → action
    grouped = df.groupby(["predicate", "dataset", "action"])
    for (predicate, dataset, action), subset in grouped:
        if subset.empty:
            continue

        # Build square matrix: rows = llm, cols = llm_cols
        mat = subset.set_index("llm")[llm_cols].copy()

        # Ensure numeric
        mat = mat.apply(pd.to_numeric, errors="coerce")

        # Mask: non-significant values (p > 0.05)
        mask = (mat > 0.05) | mat.isna()

        # Annotation: ≈0 for very small, 3 decimals otherwise
        annot = mat.copy()
        for r in annot.index:
            for c in annot.columns:
                val = annot.loc[r, c]
                if pd.isna(val) or val > 0.05:
                    annot.loc[r, c] = ""
                elif val <= 0.001:
                    annot.loc[r, c] = "≈0"
                else:
                    annot.loc[r, c] = f"{val:.3f}"

        # Skip if no significant values
        if (~mask).sum().sum() == 0:
            print(f"Skipping {predicate} | {dataset} | {action}: no significant values.")
            continue

        # Prepare figure
        figsize = (max(6, 0.8 * len(llm_cols)), max(6, 0.8 * len(subset)))
        plt.figure(figsize=figsize)

        sns.heatmap(
            mat,
            mask=mask,
            annot=annot,
            fmt="",
            cmap=cmap,
            vmin=0.0, vmax=0.05,
            cbar_kws={"label": "p-value (only p ≤ 0.05 shown)"},
            linewidths=0.5, linecolor="lightgray"
        )

        plt.title(f"{predicate} | {dataset} | {action}", fontsize=10)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save inside predicate folder
        pred_dir = os.path.join(out_dir, predicate)
        os.makedirs(pred_dir, exist_ok=True)

        fname = f"{dataset}_{action}_pvalues.png".replace(" ", "_")
        plt.savefig(os.path.join(pred_dir, fname), dpi=300)
        plt.close()

        print(f"Saved: {os.path.join(pred_dir, fname)}")


# Example usage
if __name__ == "__main__":

    df = pd.read_csv("../output/summary_2025-09-10_22-14.csv")
    create_action_barcharts(df, output_folder="../charts", exclude_action="wikidata", metrics=['J(A1-A2)', 'J(A1-A34)', 'J(A4-A1|3)', 'J(A3-A4)'], exclude_llms=["deepseek-r1:1.5b", "deepseek-r1:70b"])

    plot_pvalue_heatmaps_by_dataset_action("../output/summary_2025-09-10_22-14.csv",exclude_llms=["deepseek-r1:1.5b", "deepseek-r1:70b"])

    plot_square_heatmaps("../output/p_value_matrices_2025-09-10_22-14.csv", out_dir="../charts/llms_pvalue_heatmaps", exclude_llms=["deepseek-r1:1.5b", "deepseek-r1:70b"])
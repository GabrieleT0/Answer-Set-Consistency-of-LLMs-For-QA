import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp

root_dir = os.path.dirname(os.path.abspath(__file__))

# Load all three datasets and combine them
df_zs = pd.read_csv(f"{root_dir}/../../output/summary_zero-shot.csv")
df_cte = pd.read_csv(f"{root_dir}/../../output/summary_classification.csv")
df_ora = pd.read_csv(f"{root_dir}/../../output/summary_fixing.csv")
combined_df = pd.concat([df_zs, df_cte, df_ora], ignore_index=True)

def analyze_idk_consistency_tradeoff(df, consistency_col, visualize=True, verbose=True):
    """
    Analyze the trade-off between IDK rate and a specific consistency metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Combined dataframe with 18 models × 3 actions = 54 rows
        Must have columns: 'llm', 'idk', and the consistency column
    consistency_col : str
        Column name for the consistency metric (e.g., '?A1=A2', 'J_A1_ave')
    visualize : bool
        Whether to create and save visualization plots
    verbose : bool
        Whether to print detailed output
    
    Returns
    -------
    dict
        Results including:
        - 'metric': the consistency column name
        - 'n_models': number of models analyzed
        - 'mean_r': mean correlation coefficient
        - 'std_r': standard deviation of correlations
        - 'min_r', 'max_r': min and max correlation
        - 't_stat': t-statistic from one-sample t-test
        - 'p_value': two-tailed p-value
        - 'p_value_onetail': one-tailed p-value
        - 'significant': boolean, True if p < 0.05
        - 'correlations': list of dicts with per-model details
    """
    
    # Verify consistency column exists
    if consistency_col not in df.columns:
        raise ValueError(f"Column '{consistency_col}' not found in dataframe")
    
    # Compute per-model correlations
    model_correlations = []
    model_details = []
    
    for model, group in df.groupby("llm"):
        idk_vals = group['idk'].values
        consistency_vals = group[consistency_col].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(idk_vals) | np.isnan(consistency_vals))
        idk_vals = idk_vals[valid_mask]
        consistency_vals = consistency_vals[valid_mask]
        
        # Require at least 3 valid points for correlation
        if len(idk_vals) >= 3:
            r, p = pearsonr(idk_vals, consistency_vals)
            model_correlations.append(r)
            model_details.append({
                "model": model,
                "correlation": r,
                "p_value": p,
                "n_valid_actions": len(idk_vals)
            })
    
    if not model_correlations:
        raise ValueError(f"No valid correlations computed for '{consistency_col}'")
    
    # Convert to array for meta-analysis
    correlations_array = np.array(model_correlations)
    
    # Compute summary statistics
    mean_r = correlations_array.mean()
    std_r = correlations_array.std()
    min_r = correlations_array.min()
    max_r = correlations_array.max()
    
    # One-sample t-test: H0: mean(r) = 0
    t_stat, t_pval = ttest_1samp(correlations_array, 0)
    t_pval_onetail = t_pval / 2 if mean_r > 0 else 1 - (t_pval / 2)
    
    # Print verbose output
    if verbose:
        print(f"\n{'='*70}")
        print(f"Consistency Metric: {consistency_col}")
        print(f"{'='*70}")
        print(f"Models analyzed: {len(model_details)}")
        print(f"\n📊 Summary Statistics:")
        print(f"   Mean correlation (r):  {mean_r:>8.4f}")
        print(f"   Std deviation:         {std_r:>8.4f}")
        print(f"   Min correlation:       {min_r:>8.4f}")
        print(f"   Max correlation:       {max_r:>8.4f}")
        
        print(f"\n🧪 One-Sample t-test (H0: mean(r) = 0):")
        print(f"   t-statistic:           {t_stat:>8.4f}")
        print(f"   p-value (2-tailed):    {t_pval:>8.4f}")
        print(f"   p-value (1-tailed):    {t_pval_onetail:>8.4f}")
        
        if t_pval < 0.05:
            print(f"   Status:                ✅ SIGNIFICANT (α=0.05)")
        else:
            print(f"   Status:                ❌ NOT SIGNIFICANT (α=0.05)")
        
        # Show per-model breakdown
        print(f"\n📋 Per-Model Correlations (sorted by strength):")
        details_df = pd.DataFrame(model_details).sort_values("correlation", ascending=False)
        print(details_df[['model', 'correlation', 'p_value']].to_string(index=False))
    
    # Generate visualization
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of correlations
        axes[0].hist(correlations_array, bins=8, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(mean_r, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_r:.3f}')
        axes[0].axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='r=0 (null)')
        axes[0].set_xlabel('Pearson Correlation (r)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title(f'Distribution of Per-Model Correlations\n{consistency_col}', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Bar plot of correlations per model
        sorted_details = sorted(model_details, key=lambda x: x['correlation'], reverse=True)
        models_list = [d['model'] for d in sorted_details]
        corrs_list = [d['correlation'] for d in sorted_details]
        colors = ['green' if c > 0 else 'red' for c in corrs_list]
        
        axes[1].barh(models_list, corrs_list, color=colors, alpha=0.7, edgecolor='black')
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].set_xlabel('Pearson Correlation (r)', fontsize=11)
        axes[1].set_title(f'Per-Model Correlations\n{consistency_col}', fontsize=12, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save with sanitized filename
        safe_metric_name = consistency_col.replace('?', '').replace('∅', 'empty').replace('=', 'eq').replace('>', 'gt').replace('+', 'plus')
        output_file = f"{root_dir}/../../output/tradeoff/idk_consistency_tradeoff_{safe_metric_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"\n💾 Plot saved to: {output_file}")
        plt.close()
    
    return {
        "metric": consistency_col,
        "n_models": len(model_details),
        "mean_r": mean_r,
        "std_r": std_r,
        "min_r": min_r,
        "max_r": max_r,
        "t_stat": t_stat,
        "p_value": t_pval,
        "p_value_onetail": t_pval_onetail,
        "significant": t_pval < 0.05,
        "correlations": model_details
    }


# ============================================================================
# MAIN ANALYSIS: Test multiple consistency metrics
# ============================================================================

if __name__ == "__main__":
    # Define multiple consistency metrics to analyze
    consistency_metrics = [
        "?A1=A2",
        "?A1=A3+A4",
        "?A1>A3",
        "?A1>A4",
        "?A3∅A4",
        "?A4=A1|3",
        "J(A1-A2)",
        "J(A3-A4)",
        "J(A4-A1|3)"
    ]
    
    print(f"\n{'='*70}")
    print("IDK-Consistency Trade-off Analysis")
    print(f"{'='*70}")
    print(f"Analyzing {len(combined_df)} data points (18 models × 3 actions)")
    
    # Run analysis for each metric
    results = []
    for metric in consistency_metrics:
        try:
            result = analyze_idk_consistency_tradeoff(combined_df, metric, visualize=True, verbose=True)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error analyzing '{metric}': {e}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE: All Metrics")
    print(f"{'='*70}\n")
    
    if results:
        summary_data = []
        for r in results:
            summary_data.append({
                "Metric": r["metric"],
                "Models": r["n_models"],
                "Mean r": f"{r['mean_r']:>7.4f}",
                "Std r": f"{r['std_r']:>7.4f}",
                "t-stat": f"{r['t_stat']:>7.4f}",
                "p-value": f"{r['p_value']:>8.4f}",
                "Significant": "✅ YES" if r['significant'] else "   NO"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save to CSV (with numeric values, no formatting)
        csv_summary_data = []
        for r in results:
            csv_summary_data.append({
                "Metric": r["metric"],
                "Models": r["n_models"],
                "Mean_r": round(r['mean_r'], 4),
                "Std_r": round(r['std_r'], 4),
                "Min_r": round(r['min_r'], 4),
                "Max_r": round(r['max_r'], 4),
                "t_stat": round(r['t_stat'], 4),
                "p_value": round(r['p_value'], 6),
                "Significant": "Yes" if r['significant'] else "No"
            })
        
        csv_summary_df = pd.DataFrame(csv_summary_data)
        csv_output_path = f"{root_dir}/../../output/tradeoff/idk_consistency_tradeoff_summary.csv"
        csv_summary_df.to_csv(csv_output_path, index=False)
        print(f"\n💾 Summary table saved to: {csv_output_path}")
        
        # Save all per-model correlations to CSV
        all_correlations = []
        for r in results:
            for model_info in r["correlations"]:
                all_correlations.append({
                    "Metric": r["metric"],
                    "Model": model_info["model"],
                    "Correlation": round(model_info["correlation"], 4),
                    "p_value": round(model_info["p_value"], 6)
                })
        
        correlations_df = pd.DataFrame(all_correlations)
        correlations_csv_path = f"{root_dir}/../../output/tradeoff/idk_consistency_tradeoff_per_model.csv"
        correlations_df.to_csv(correlations_csv_path, index=False)
        print(f"💾 Per-model correlations saved to: {correlations_csv_path}")
        
        # Interpretation
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print(f"{'='*70}")
        
        sig_metrics = [r for r in results if r['significant']]
        print(f"\nSignificant trade-offs: {len(sig_metrics)}/{len(results)} metrics\n")
        
        if sig_metrics:
            print("✅ METRICS WITH SIGNIFICANT TRADE-OFF:")
            for r in sig_metrics:
                direction = "↑ positive" if r['mean_r'] > 0 else "↓ negative"
                print(f"   • {r['metric']:20s} (r={r['mean_r']:>7.4f}, p={r['p_value']:.4f}) [{direction}]")
        
        not_sig = [r for r in results if not r['significant']]
        if not_sig:
            print(f"\n❌ METRICS WITHOUT SIGNIFICANT TRADE-OFF:")
            for r in not_sig:
                print(f"   • {r['metric']:20s} (r={r['mean_r']:>7.4f}, p={r['p_value']:.4f})")
        
        print(f"\n{'='*70}\n")

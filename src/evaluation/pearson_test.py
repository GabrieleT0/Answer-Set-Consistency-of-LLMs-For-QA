import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

root_dir = os.path.dirname(os.path.abspath(__name__))
llm_path = f"{root_dir}/data/llm_info.json"
with open(llm_path, "r", encoding="utf-8") as f:
    llms_info = json.load(f)

df_summery_zs = pd.read_csv(f"{root_dir}/output/summary_zero-shot.csv")

# Your existing code for data preparation
IDs = [] 
x = []
y_R1= []
y_R2 = []
y_R3 = []
y_R4 = []
y_R5 = []
y_R = []

y_JR1 = []
y_JR4 = []
y_JR5 = []

for key in llms_info.keys():
    if llms_info[key]["score"]:
        IDs.append(key)    
        x.append(llms_info[key]["score"])
        r1 = float(df_summery_zs[df_summery_zs["llm"]==key]["?A1=A2"].values[0])
        y_R1.append(r1)
        r2 = float(df_summery_zs[df_summery_zs["llm"]==key]["?A1>A3"].values[0])
        r3 = float(df_summery_zs[df_summery_zs["llm"]==key]["?A1>A4"].values[0])
        r4 = float(df_summery_zs[df_summery_zs["llm"]==key]["?A3∅A4"].values[0])
        r5 = float(df_summery_zs[df_summery_zs["llm"]==key]["?A4=A1|3"].values[0])
        y_R2.append(r2)
        y_R3.append(r3)
        y_R4.append(r4)
        y_R5.append(r5)
        r = np.mean([r1,r2,r3,r4,r5])
        y_R.append(r)

        jr1 = float(df_summery_zs[df_summery_zs["llm"]==key]["J(A1-A2)"].values[0])
        jr4 = float(df_summery_zs[df_summery_zs["llm"]==key]["J(A3-A4)"].values[0])
        jr5 = float(df_summery_zs[df_summery_zs["llm"]==key]["J(A4-A1|3)"].values[0])
        y_JR1.append(jr1)
        y_JR4.append(jr4)
        y_JR5.append(jr5)

# Convert to numpy array
x = np.array(x)

# Calculate correlations for all y arrays
correlations = {}

# Define all y arrays and their names
y_arrays = {
    'R1': y_R1,
    'R2': y_R2, 
    'R3': y_R3,
    'R4': y_R4,
    'R5': y_R5,
    'R_mean': y_R,
    'JR1': y_JR1,
    'JR4': y_JR4,
    'JR5': y_JR5
}

# Calculate Pearson correlations
results = []
for name, y_array in y_arrays.items():
    r, p_value = pearsonr(x, y_array)
    results.append({
        'Metric': name,
        'Correlation_r': r,
        'P_value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    })

# Create DataFrame and save to CSV
df_correlations = pd.DataFrame(results)

# Display the results
print("Correlation Results:")
print(df_correlations.to_string(index=False))

# Save to CSV
df_correlations.to_csv(root_dir + '/output/correlation_results.csv', index=False)
print(f"\nResults saved to 'correlation_results.csv'")

# Optional: Round the values for better readability
df_correlations['Correlation_r'] = df_correlations['Correlation_r'].round(4)
df_correlations['P_value'] = df_correlations['P_value'].round(6)

print("\nRounded Results:")
print(df_correlations.to_string(index=False))

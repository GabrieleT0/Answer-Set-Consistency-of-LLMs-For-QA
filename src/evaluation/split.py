import pandas as pd
import json 
import os

def split(df, prefix, actions = None, llm_info = None, folder = None):
    if actions is None:
        actions = ["zero-shot", "classification", "fixing"]
    # Create the name-to-ID mapping
    if llm_info is None:
        root_dir = os.path.dirname(os.path.abspath(__name__))
        folder = root_dir + "/output/"
        llm_path = f"{root_dir}/data/llm_info.json"
        with open(llm_path, "r", encoding="utf-8") as f:
            llm_info = json.load(f)
    name_id_map = {key: llm_info[key]["ID"] for key in llm_info}

    df = df[df["dataset"] == "overall"]

    for act in actions:
        df_act = df[df["action"] == act].copy()  # Use .copy() to avoid SettingWithCopyWarning
        df_act["ID"] = df_act["llm"].apply(lambda x: name_id_map[x])
        df_act = df_act.sort_values(by="ID", ascending=True)

        cols = ["ID"] + [col for col in df_act.columns if col != "ID"]
        df_act = df_act[cols]
        df_act.to_csv(folder + f"{prefix}_{act}.csv", index=False)
        print(f"Saved: {prefix}_{act}.csv")

def main():
    root_dir = os.path.dirname(os.path.abspath(__name__))
    folder = root_dir + "/output/"
    # file = "summary_2025-09-23_16-04.csv"
    file = "p_value_matrices.csv"
    prefix = "p_value"
    df = pd.read_csv(folder + file)
    split(df, prefix)


if __name__=="__main__":
    main()

# Now split_dfs['action_value'] gives you the DataFrame for that action

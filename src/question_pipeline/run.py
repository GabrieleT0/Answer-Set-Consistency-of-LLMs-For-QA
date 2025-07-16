import json
import pandas as pd
from tqdm import tqdm
from agent_workflow import run_agent_workflow, load_config

def main():
    config = load_config()
    input_path = config["input_path"]
    output_path = config["output_path"]


    with open(input_path, "r") as f:
        question_data = json.load(f)

    processed_results = []

    for i in tqdm(range(len(question_data)), desc="Processing"):
        item = question_data[i]
        q1 = item.get("question")
        uid = item.get("uid", f"no_uid_{i}")

        try:
            result = run_agent_workflow(q1)
            if result:
                processed_results.append({"uid": uid, **result})
        except Exception as e:
            print(f"❌ Error at index {i}: {e}")
            continue

    df = pd.DataFrame(processed_results)
    df.to_csv(output_path + ".csv", index=False)
    df.to_csv(output_path + ".tsv", sep="\t", index=False)

    print(f"\n✅ Saved {len(df)} rows to:\n- {output_path}.csv\n- {output_path}.tsv")

if __name__ == "__main__":
    main()

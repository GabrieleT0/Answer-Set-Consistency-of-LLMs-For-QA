import json
import os

# --- CONFIGURATION ---
FOLDER_PATH = ['./equal','./minus','./relation-classification', './sup-sub']  # folders containing JSON files
KEYS_TO_CLEAR = ["3", "7",'44','65','91','105']   # keys whose values you want to cancel (empty list)

# ----------------------

def process_json_files(folder_path, keys_to_clear):
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)

        # Load JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Clear selected keys
        for key in keys_to_clear:
            if key in data:
                del data[key]

        # Save updated JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Processed: {filename}")

# Run it
for folder in FOLDER_PATH:
    process_json_files(folder, KEYS_TO_CLEAR)
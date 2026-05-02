import os
import re
import pandas as pd

BASE_DIR = os.path.join("data", "raw", "prometheus_export")
OUTPUT_DIR = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "smartscale_training_dataset.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_users_from_run_name(run_name):
    match = re.search(r"run_(\d+)_users", run_name)
    if match:
        return int(match.group(1))
    return None


all_runs = []

for run_name in os.listdir(BASE_DIR):
    run_path = os.path.join(BASE_DIR, run_name)

    if not os.path.isdir(run_path):
        continue

    combined_file = os.path.join(run_path, "combined_metrics.csv")

    if not os.path.exists(combined_file):
        print(f"Skipping {run_name}: combined_metrics.csv not found")
        continue

    print(f"Reading {combined_file}")

    df = pd.read_csv(combined_file)

    df["run_name"] = run_name
    df["users"] = extract_users_from_run_name(run_name)

    all_runs.append(df)

if not all_runs:
    raise Exception("No combined_metrics.csv files found.")

merged_df = pd.concat(all_runs, ignore_index=True)

merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])
merged_df = merged_df.sort_values(["run_name", "timestamp", "metric"])

merged_df.to_csv(OUTPUT_FILE, index=False)

print("Done.")
print(f"Saved merged dataset: {OUTPUT_FILE}")
print(f"Rows: {len(merged_df)}")
print(f"Runs: {merged_df['run_name'].nunique()}")
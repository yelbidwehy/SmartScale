import os
import re
import pandas as pd

BASE_DIR = os.path.join("data", "raw", "prometheus_export")
OUTPUT_DIR = os.path.join("data", "processed")

RAW_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "smartscale_training_dataset.csv")
CLEAN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "smartscale_training_dataset_cleaned.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "smartscale_cleaning_report.csv")

TARGET_SERVICES = [
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "shippingservice",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_users_from_run_name(run_name):
    match = re.search(r"run_(\d+)_users", run_name)
    if match:
        return int(match.group(1))
    return None


def normalize_service_name(row):
    if "destination_workload" in row and pd.notna(row["destination_workload"]):
        return row["destination_workload"]

    pod = str(row.get("pod", ""))
    for service in TARGET_SERVICES:
        if pod.startswith(service + "-"):
            return service

    return None


all_runs = []

for run_name in os.listdir(BASE_DIR):
    run_path = os.path.join(BASE_DIR, run_name)

    if not os.path.isdir(run_path):
        continue

    users = extract_users_from_run_name(run_name)
    if users is None:
        print(f"Skipping {run_name}: invalid run name")
        continue

    combined_file = os.path.join(run_path, "combined_metrics.csv")

    if not os.path.exists(combined_file):
        print(f"Skipping {run_name}: combined_metrics.csv not found")
        continue

    print(f"Reading {combined_file}")

    df = pd.read_csv(combined_file)
    df["run_name"] = run_name
    df["users"] = users

    all_runs.append(df)

if not all_runs:
    raise Exception("No valid combined_metrics.csv files found.")

merged_df = pd.concat(all_runs, ignore_index=True)

merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])
merged_df["timestamp"] = merged_df["timestamp"].dt.floor("5s")

merged_df = merged_df.sort_values(["run_name", "timestamp", "metric"])
merged_df.to_csv(RAW_OUTPUT_FILE, index=False)


# =========================
# FRONTEND LOAD METRICS
# =========================

frontend_rps = (
    merged_df[merged_df["metric"] == "rps_total"]
    .groupby(["run_name", "users", "timestamp"], as_index=False)["value"]
    .mean()
    .rename(columns={"value": "frontend_rps"})
)

frontend_requests = (
    merged_df[merged_df["metric"] == "total_requests"]
    .groupby(["run_name", "users", "timestamp"], as_index=False)["value"]
    .mean()
    .rename(columns={"value": "frontend_total_requests_1m"})
)

frontend_rps["timestamp"] = pd.to_datetime(frontend_rps["timestamp"]).dt.floor("5s")
frontend_requests["timestamp"] = pd.to_datetime(frontend_requests["timestamp"]).dt.floor("5s")


# =========================
# SERVICE METRICS
# =========================

df = merged_df.copy()
df["service"] = df.apply(normalize_service_name, axis=1)
df = df[df["service"].isin(TARGET_SERVICES)].copy()

service_metrics = df[df["metric"].isin([
    "cpu_usage_cores",
    "memory_usage_bytes",
    "latency_p95_ms",
    "latency_avg_ms",
])].copy()

service_wide = service_metrics.pivot_table(
    index=["run_name", "users", "timestamp", "service"],
    columns="metric",
    values="value",
    aggfunc="mean"
).reset_index()

cleaned_df = service_wide.merge(
    frontend_rps,
    on=["run_name", "users", "timestamp"],
    how="left"
)

cleaned_df = cleaned_df.merge(
    frontend_requests,
    on=["run_name", "users", "timestamp"],
    how="left"
)

cleaned_df = cleaned_df.sort_values(["run_name", "service", "timestamp"])

cleaned_df["frontend_rps"] = (
    cleaned_df.groupby(["run_name"])["frontend_rps"]
    .transform(lambda s: s.ffill().bfill())
    .fillna(0)
)

cleaned_df["frontend_total_requests_1m"] = (
    cleaned_df.groupby(["run_name"])["frontend_total_requests_1m"]
    .transform(lambda s: s.ffill().bfill())
    .fillna(0)
)

cleaned_df["latency_avg_was_missing"] = cleaned_df["latency_avg_ms"].isna()
cleaned_df["latency_p95_was_missing"] = cleaned_df["latency_p95_ms"].isna()

for col in ["latency_avg_ms", "latency_p95_ms"]:
    cleaned_df[col] = (
        cleaned_df
        .groupby(["run_name", "service"])[col]
        .transform(lambda s: s.ffill().bfill())
    )

numeric_cols = [
    "frontend_rps",
    "frontend_total_requests_1m",
    "cpu_usage_cores",
    "memory_usage_bytes",
    "latency_avg_ms",
    "latency_p95_ms",
]

for col in numeric_cols:
    cleaned_df[col] = cleaned_df[col].fillna(0)

cleaned_df.loc[cleaned_df["frontend_rps"] == 0, "latency_avg_ms"] = 0
cleaned_df.loc[cleaned_df["frontend_rps"] == 0, "latency_p95_ms"] = 0

cleaned_df = cleaned_df.groupby(
    ["run_name", "users", "timestamp", "service"],
    as_index=False
).agg({
    "frontend_rps": "mean",
    "frontend_total_requests_1m": "mean",
    "cpu_usage_cores": "sum",
    "memory_usage_bytes": "sum",
    "latency_avg_ms": "mean",
    "latency_p95_ms": "mean",
    "latency_avg_was_missing": "max",
    "latency_p95_was_missing": "max",
})

cleaned_df = cleaned_df.sort_values(["run_name", "timestamp", "service"])
cleaned_df.to_csv(CLEAN_OUTPUT_FILE, index=False)


# =========================
# REPORT
# =========================

report = pd.DataFrame({
    "item": [
        "raw_rows",
        "cleaned_rows",
        "runs",
        "services",
        "frontend_rps_zero_rows",
        "frontend_rps_min",
        "frontend_rps_max",
        "latency_avg_missing_before_fill",
        "latency_p95_missing_before_fill",
        "missing_values_after_cleaning",
    ],
    "value": [
        len(merged_df),
        len(cleaned_df),
        cleaned_df["run_name"].nunique(),
        cleaned_df["service"].nunique(),
        int((cleaned_df["frontend_rps"] == 0).sum()),
        cleaned_df["frontend_rps"].min(),
        cleaned_df["frontend_rps"].max(),
        int(cleaned_df["latency_avg_was_missing"].sum()),
        int(cleaned_df["latency_p95_was_missing"].sum()),
        int(cleaned_df.isna().sum().sum()),
    ]
})

report.to_csv(REPORT_FILE, index=False)

print("Done.")
print(f"Saved raw dataset: {RAW_OUTPUT_FILE}")
print(f"Saved cleaned dataset: {CLEAN_OUTPUT_FILE}")
print(f"Saved cleaning report: {REPORT_FILE}")
print(f"Cleaned rows: {len(cleaned_df)}")
print(f"Frontend RPS zero rows: {(cleaned_df['frontend_rps'] == 0).sum()}")
print(f"Frontend RPS min: {cleaned_df['frontend_rps'].min()}")
print(f"Frontend RPS max: {cleaned_df['frontend_rps'].max()}")
print(f"Missing values after cleaning: {cleaned_df.isna().sum().sum()}")
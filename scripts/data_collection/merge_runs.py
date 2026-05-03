import re
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = PROJECT_ROOT / "data" / "raw" / "prometheus_export"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RAW_OUTPUT_FILE = OUTPUT_DIR / "smartscale_training_dataset.csv"
CLEAN_OUTPUT_FILE = OUTPUT_DIR / "smartscale_training_dataset_cleaned.csv"
REPORT_FILE = OUTPUT_DIR / "smartscale_cleaning_report.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

METRIC_RENAME = {
    "frontend_rps_total": "frontend_rps",
    "frontend_total_requests": "frontend_total_requests_1m",
    "service_rps": "service_rps",
    "service_latency_p95_ms": "latency_p95_ms",
    "service_latency_avg_ms": "latency_avg_ms",
    "cpu_usage_cores": "cpu_usage_cores",
    "memory_usage_bytes": "memory_usage_bytes",
}


def extract_users_from_run_name(run_name: str):
    match = re.search(r"run_(\d+)_users", run_name)
    if match:
        return int(match.group(1))
    return None


def normalize_service_name(row):
    if "destination_workload" in row and pd.notna(row["destination_workload"]):
        service = str(row["destination_workload"])
        if service in TARGET_SERVICES:
            return service

    pod = str(row.get("pod", ""))
    for service in TARGET_SERVICES:
        if pod.startswith(service + "-"):
            return service

    return None


def load_all_runs():
    all_runs = []

    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

    for run_path in sorted(BASE_DIR.iterdir()):
        if not run_path.is_dir():
            continue

        run_name = run_path.name
        users = extract_users_from_run_name(run_name)

        if users is None:
            print(f"Skipping {run_name}: invalid run name")
            continue

        combined_file = run_path / "combined_metrics.csv"

        if not combined_file.exists():
            print(f"Skipping {run_name}: combined_metrics.csv not found")
            continue

        print(f"Reading {combined_file}")

        df = pd.read_csv(combined_file)
        df["run_name"] = run_name
        df["users"] = users

        all_runs.append(df)

    if not all_runs:
        raise RuntimeError("No valid combined_metrics.csv files found.")

    return pd.concat(all_runs, ignore_index=True)


def main():
    merged_df = load_all_runs()

    merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])
    merged_df["timestamp"] = merged_df["timestamp"].dt.floor("5s")

    merged_df = merged_df.sort_values(["run_name", "timestamp", "metric"])
    merged_df.to_csv(RAW_OUTPUT_FILE, index=False)

    # =========================
    # Frontend gateway load metrics
    # =========================
    frontend_rps = (
        merged_df[merged_df["metric"] == "frontend_rps_total"]
        .groupby(["run_name", "users", "timestamp"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "frontend_rps"})
    )

    frontend_requests = (
        merged_df[merged_df["metric"] == "frontend_total_requests"]
        .groupby(["run_name", "users", "timestamp"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "frontend_total_requests_1m"})
    )

    # =========================
    # Per-service metrics
    # =========================
    df = merged_df.copy()
    df["service"] = df.apply(normalize_service_name, axis=1)
    df = df[df["service"].isin(TARGET_SERVICES)].copy()

    service_metrics = df[df["metric"].isin([
        "service_rps",
        "service_latency_p95_ms",
        "service_latency_avg_ms",
        "cpu_usage_cores",
        "memory_usage_bytes",
    ])].copy()

    service_metrics["metric"] = service_metrics["metric"].replace(METRIC_RENAME)

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

    # Ensure expected columns exist
    expected_cols = [
        "frontend_rps",
        "frontend_total_requests_1m",
        "service_rps",
        "cpu_usage_cores",
        "memory_usage_bytes",
        "latency_avg_ms",
        "latency_p95_ms",
    ]

    for col in expected_cols:
        if col not in cleaned_df.columns:
            cleaned_df[col] = pd.NA

    # Missing flags before fill
    cleaned_df["service_rps_was_missing"] = cleaned_df["service_rps"].isna()
    cleaned_df["latency_avg_was_missing"] = cleaned_df["latency_avg_ms"].isna()
    cleaned_df["latency_p95_was_missing"] = cleaned_df["latency_p95_ms"].isna()

    # Fill frontend metrics per run
    for col in ["frontend_rps", "frontend_total_requests_1m"]:
        cleaned_df[col] = (
            cleaned_df.groupby("run_name")[col]
            .transform(lambda s: s.ffill().bfill())
            .fillna(0)
        )

    # Fill service metrics per run + service
    for col in [
        "service_rps",
        "cpu_usage_cores",
        "memory_usage_bytes",
        "latency_avg_ms",
        "latency_p95_ms",
    ]:
        cleaned_df[col] = (
            cleaned_df.groupby(["run_name", "service"])[col]
            .transform(lambda s: s.ffill().bfill())
            .fillna(0)
        )

    # No frontend load means no meaningful request latency/RPS
    zero_load_mask = cleaned_df["frontend_rps"] == 0
    cleaned_df.loc[zero_load_mask, "service_rps"] = 0
    cleaned_df.loc[zero_load_mask, "latency_avg_ms"] = 0
    cleaned_df.loc[zero_load_mask, "latency_p95_ms"] = 0

    cleaned_df = cleaned_df.groupby(
        ["run_name", "users", "timestamp", "service"],
        as_index=False
    ).agg({
        "frontend_rps": "mean",
        "frontend_total_requests_1m": "mean",
        "service_rps": "mean",
        "cpu_usage_cores": "sum",
        "memory_usage_bytes": "sum",
        "latency_avg_ms": "mean",
        "latency_p95_ms": "mean",
        "service_rps_was_missing": "max",
        "latency_avg_was_missing": "max",
        "latency_p95_was_missing": "max",
    })

    cleaned_df = cleaned_df.sort_values(["run_name", "timestamp", "service"])
    cleaned_df.to_csv(CLEAN_OUTPUT_FILE, index=False)

    report = pd.DataFrame({
        "item": [
            "raw_rows",
            "cleaned_rows",
            "runs",
            "services",
            "frontend_rps_zero_rows",
            "frontend_rps_min",
            "frontend_rps_max",
            "service_rps_missing_before_fill",
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
            int(cleaned_df["service_rps_was_missing"].sum()),
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


if __name__ == "__main__":
    main()
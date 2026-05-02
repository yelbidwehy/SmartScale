import os
import re
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "smartscale_training_dataset.csv"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "two_model_dataset"
MODEL1_DIR = OUTPUT_ROOT / "model1_rps_forecast"
MODEL2_DIR = OUTPUT_ROOT / "model2_capacity_per_service"

MODEL1_DIR.mkdir(parents=True, exist_ok=True)
MODEL2_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 12
PREDICT_STEP = 1

CPU_PER_POD = 0.5
MEMORY_PER_POD_MB = 512
SAFETY_FACTOR = 1.2
MIN_REPLICAS = 1
MAX_REPLICAS = 10

EXCLUDED_PATTERNS = [
    "istio",
    "prometheus",
    "grafana",
    "locust",
    "online-boutique-test",
    "kube",
    "otel",
    "jaeger",
    "alertmanager",
    "node-exporter",
    "metrics"
]


def extract_service_name(pod_name):
    if pd.isna(pod_name):
        return None

    pod_name = str(pod_name)

    # Remove Kubernetes deployment suffix:
    # frontend-757dfdbc6b-z2qgd -> frontend
    pod_name = re.sub(r"-[a-f0-9]{8,10}-[a-z0-9]{5}$", "", pod_name)

    # Remove StatefulSet suffix:
    # redis-cart-0 -> redis-cart
    pod_name = re.sub(r"-\d+$", "", pod_name)

    return pod_name


def is_valid_service(service):
    if service is None:
        return False

    service_lower = str(service).lower()

    for pattern in EXCLUDED_PATTERNS:
        if pattern in service_lower:
            return False

    return True


def calculate_replicas(cpu_cores, memory_bytes):
    memory_mb = memory_bytes / (1024 * 1024)

    required_cpu = cpu_cores * SAFETY_FACTOR
    required_memory = memory_mb * SAFETY_FACTOR

    replicas_by_cpu = math.ceil(required_cpu / CPU_PER_POD)
    replicas_by_memory = math.ceil(required_memory / MEMORY_PER_POD_MB)

    replicas = max(replicas_by_cpu, replicas_by_memory)
    replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, replicas))

    return replicas


def create_rps_forecast_dataset(rps_values):
    X, y = [], []

    for i in range(len(rps_values) - WINDOW_SIZE - PREDICT_STEP + 1):
        X.append(rps_values[i:i + WINDOW_SIZE])
        y.append(rps_values[i + WINDOW_SIZE + PREDICT_STEP - 1])

    return np.array(X), np.array(y)


def main():
    print("Reading merged dataset...")
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["timestamp", "metric", "value"])

    # =========================
    # Base metrics: RPS + latency
    # =========================
    base_metrics = df[df["metric"].isin([
        "rps_total",
        "latency_p95_ms",
        "latency_avg_ms"
    ])].copy()

    base_wide = (
        base_metrics
        .pivot_table(
            index=["timestamp", "run_name", "users"],
            columns="metric",
            values="value",
            aggfunc="mean"
        )
        .reset_index()
    )

    base_wide.columns.name = None
    base_wide = base_wide.sort_values(["run_name", "timestamp"])
    base_wide = base_wide.ffill().bfill().fillna(0)

    # =========================
    # Model 1: RPS Forecasting
    # =========================
    print("\nPreparing Model 1 dataset...")

    rps_df = base_wide[["timestamp", "run_name", "users", "rps_total"]].copy()
    rps_df = rps_df.sort_values(["run_name", "timestamp"])

    rps_scaler = MinMaxScaler()
    rps_df["rps_scaled"] = rps_scaler.fit_transform(rps_df[["rps_total"]])

    X_rps, y_rps = [], []

    for run_name, group in rps_df.groupby("run_name"):
        values = group["rps_scaled"].values

        X_run, y_run = create_rps_forecast_dataset(values)

        if len(X_run) > 0:
            X_rps.append(X_run)
            y_rps.append(y_run)

    X_rps = np.concatenate(X_rps, axis=0).reshape(-1, WINDOW_SIZE, 1)
    y_rps = np.concatenate(y_rps, axis=0).reshape(-1, 1)

    np.save(MODEL1_DIR / "X_rps.npy", X_rps)
    np.save(MODEL1_DIR / "y_rps.npy", y_rps)
    joblib.dump(rps_scaler, MODEL1_DIR / "rps_scaler.pkl")

    rps_df.to_csv(MODEL1_DIR / "model1_rps_dataset.csv", index=False)

    print(f"Model 1 X shape: {X_rps.shape}")
    print(f"Model 1 y shape: {y_rps.shape}")

    # =========================
    # Resource metrics per service
    # =========================
    print("\nPreparing service resource metrics...")

    resource_df = df[df["metric"].isin([
        "cpu_usage_cores",
        "memory_usage_bytes"
    ])].copy()

    resource_df["service"] = resource_df["pod"].apply(extract_service_name)
    resource_df = resource_df[resource_df["service"].apply(is_valid_service)]

    print("\nDetected services after filtering:")
    for service in sorted(resource_df["service"].dropna().unique()):
        print(f"- {service}")

    resource_wide = (
        resource_df
        .pivot_table(
            index=["timestamp", "run_name", "users", "service"],
            columns="metric",
            values="value",
            aggfunc="mean"
        )
        .reset_index()
    )

    resource_wide.columns.name = None
    resource_wide = resource_wide.fillna(0)

    if "cpu_usage_cores" not in resource_wide.columns:
        resource_wide["cpu_usage_cores"] = 0

    if "memory_usage_bytes" not in resource_wide.columns:
        resource_wide["memory_usage_bytes"] = 0

    # =========================
    # Calculate required replicas from CPU/RAM
    # =========================
    print("\nCalculating required replicas per service...")

    resource_wide["required_replicas"] = resource_wide.apply(
        lambda row: calculate_replicas(
            row["cpu_usage_cores"],
            row["memory_usage_bytes"]
        ),
        axis=1
    )

    # =========================
    # Merge with RPS + latency
    # =========================
    model2_long = resource_wide.merge(
        base_wide,
        on=["timestamp", "run_name", "users"],
        how="left"
    )

    model2_long = model2_long.fillna(0)

    # =========================
    # Create Model 2 wide dataset
    # Inputs: users, RPS, latency
    # Targets: required replicas per service
    # =========================
    print("\nCreating Model 2 replica prediction dataset...")

    model2_dataset = (
        model2_long
        .pivot_table(
            index=[
                "timestamp",
                "run_name",
                "users",
                "rps_total",
                "latency_p95_ms",
                "latency_avg_ms"
            ],
            columns="service",
            values="required_replicas",
            aggfunc="max"
        )
        .reset_index()
    )

    model2_dataset.columns.name = None
    model2_dataset = model2_dataset.fillna(MIN_REPLICAS)

    service_columns = [
        col for col in model2_dataset.columns
        if col not in [
            "timestamp",
            "run_name",
            "users",
            "rps_total",
            "latency_p95_ms",
            "latency_avg_ms"
        ]
    ]

    rename_map = {
        service: f"{service}_replicas"
        for service in service_columns
    }

    model2_dataset = model2_dataset.rename(columns=rename_map)

    feature_columns = [
        "users",
        "rps_total",
        "latency_p95_ms",
        "latency_avg_ms"
    ]

    target_columns = list(rename_map.values())

    X_capacity_raw = model2_dataset[feature_columns].values
    y_capacity_raw = model2_dataset[target_columns].values

    capacity_input_scaler = MinMaxScaler()
    capacity_output_scaler = MinMaxScaler()

    X_capacity = capacity_input_scaler.fit_transform(X_capacity_raw)
    y_capacity = capacity_output_scaler.fit_transform(y_capacity_raw)

    np.save(MODEL2_DIR / "X_capacity.npy", X_capacity)
    np.save(MODEL2_DIR / "y_capacity.npy", y_capacity)

    joblib.dump(capacity_input_scaler, MODEL2_DIR / "capacity_input_scaler.pkl")
    joblib.dump(capacity_output_scaler, MODEL2_DIR / "capacity_output_scaler.pkl")

    with open(MODEL2_DIR / "feature_columns.txt", "w") as f:
        for col in feature_columns:
            f.write(col + "\n")

    with open(MODEL2_DIR / "target_columns.txt", "w") as f:
        for col in target_columns:
            f.write(col + "\n")

    model2_dataset.to_csv(MODEL2_DIR / "model2_capacity_dataset.csv", index=False)
    model2_long.to_csv(MODEL2_DIR / "model2_long_debug.csv", index=False)

    print(f"Model 2 X shape: {X_capacity.shape}")
    print(f"Model 2 y shape: {y_capacity.shape}")
    print(f"Model 2 features: {feature_columns}")
    print(f"Model 2 targets: {target_columns}")

    print("\nDone.")
    print(f"Output folder: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
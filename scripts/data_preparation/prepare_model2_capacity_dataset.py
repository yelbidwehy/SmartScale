import pandas as pd
import numpy as np
import math
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "smartscale_training_dataset_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "model2_dataset_v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CPU_PER_POD = 0.2
MEMORY_PER_POD_MB = 256
SAFETY_FACTOR = 1.2
MIN_REPLICAS = 1
MAX_REPLICAS = 10


def calculate_replicas(cpu, memory_bytes):
    memory_mb = memory_bytes / (1024 * 1024)

    cpu_needed = cpu * SAFETY_FACTOR
    mem_needed = memory_mb * SAFETY_FACTOR

    r_cpu = math.ceil(cpu_needed / CPU_PER_POD)
    r_mem = math.ceil(mem_needed / MEMORY_PER_POD_MB)

    replicas = max(r_cpu, r_mem)
    replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, replicas))

    return replicas


def main():
    df = pd.read_csv(INPUT_FILE)

    original_rows = len(df)

    # Remove no-load rows
    df = df[df["frontend_rps"] > 0].copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["run_name", "service", "timestamp"])

    # Fill required numeric columns safely
    required_cols = [
        "frontend_rps",
        "service_rps",
        "cpu_usage_cores",
        "memory_usage_bytes",
        "latency_p95_ms",
        "latency_avg_ms"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df.groupby(["run_name", "service"])[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(0)

    # Predicted RPS: next frontend RPS per run/service
    df["predicted_rps"] = df.groupby(["run_name", "service"])["frontend_rps"].shift(-1)
    df["predicted_rps"] = df.groupby(["run_name", "service"])["predicted_rps"].transform(lambda s: s.ffill().bfill())
    df["predicted_rps"] = df["predicted_rps"].fillna(df["frontend_rps"])

    # RPS trend features
    df["rps_lag_1"] = df.groupby(["run_name", "service"])["frontend_rps"].shift(1)
    df["rps_lag_2"] = df.groupby(["run_name", "service"])["frontend_rps"].shift(2)
    df["rps_lag_3"] = df.groupby(["run_name", "service"])["frontend_rps"].shift(3)

    df["service_rps_lag_1"] = df.groupby(["run_name", "service"])["service_rps"].shift(1)

    df["rps_rolling_mean_3"] = (
        df.groupby(["run_name", "service"])["frontend_rps"]
        .rolling(3)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    trend_cols = [
        "rps_lag_1",
        "rps_lag_2",
        "rps_lag_3",
        "service_rps_lag_1",
        "rps_rolling_mean_3"
    ]

    for col in trend_cols:
        df[col] = df.groupby(["run_name", "service"])[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(df["frontend_rps"])

    PREDICTION_HORIZON_STEPS = 6  # 6 * 5s = 30 seconds ahead

    df["future_cpu_usage_cores"] = (
        df.groupby(["run_name", "service"])["cpu_usage_cores"]
        .shift(-PREDICTION_HORIZON_STEPS)
    )

    df["future_memory_usage_bytes"] = (
        df.groupby(["run_name", "service"])["memory_usage_bytes"]
        .shift(-PREDICTION_HORIZON_STEPS)
    )

    df = df.dropna(subset=["future_cpu_usage_cores", "future_memory_usage_bytes"])

    df["required_replicas"] = df.apply(
        lambda row: calculate_replicas(
            row["future_cpu_usage_cores"],
            row["future_memory_usage_bytes"]
        ),
        axis=1
    )

    # Encode service
    service_dummies = pd.get_dummies(df["service"], prefix="svc")
    df = pd.concat([df, service_dummies], axis=1)

    feature_columns = [
        "predicted_rps",
        "frontend_rps",
        "service_rps",
        "rps_lag_1",
        "rps_lag_2",
        "rps_lag_3",
        "service_rps_lag_1",
        "rps_rolling_mean_3",
        "cpu_usage_cores",
        "memory_usage_bytes",
        "latency_p95_ms",
        "latency_avg_ms"
    ] + list(service_dummies.columns)

    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    df[feature_columns] = df[feature_columns].fillna(0)

    X_raw = df[feature_columns].values
    y_raw = df["required_replicas"].values.reshape(-1, 1)

    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    X = input_scaler.fit_transform(X_raw)
    y = output_scaler.fit_transform(y_raw)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

    joblib.dump(input_scaler, OUTPUT_DIR / "input_scaler.pkl")
    joblib.dump(output_scaler, OUTPUT_DIR / "output_scaler.pkl")

    df.to_csv(OUTPUT_DIR / "dataset_debug.csv", index=False)

    with open(OUTPUT_DIR / "feature_columns.txt", "w") as f:
        for col in feature_columns:
            f.write(col + "\n")

    print("Dataset prepared successfully")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Original rows: {original_rows}")
    print(f"Rows after removing zero-load: {len(df)}")
    print(f"Removed rows: {original_rows - len(df)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features count: {len(feature_columns)}")
    print("\nRequired replicas distribution:")
    print(df["required_replicas"].value_counts().sort_index())


if __name__ == "__main__":
    main()
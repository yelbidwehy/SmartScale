import pandas as pd
import numpy as np
import os
import math
from sklearn.preprocessing import MinMaxScaler
import joblib

INPUT_FILE = "data/processed/smartscale_training_dataset_cleaned.csv"
OUTPUT_DIR = "data/processed/model2_dataset_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # =========================
    # Simulate predicted RPS
    # =========================
    df["predicted_rps"] = df["frontend_rps"].shift(-1)
    df["predicted_rps"] = df["predicted_rps"].ffill().bfill()

    # =========================
    # Target
    # =========================
    df["required_replicas"] = df.apply(
        lambda row: calculate_replicas(
            row["cpu_usage_cores"],
            row["memory_usage_bytes"]
        ),
        axis=1
    )

    # =========================
    # Encode service
    # =========================
    service_dummies = pd.get_dummies(df["service"], prefix="svc")
    df = pd.concat([df, service_dummies], axis=1)

    # =========================
    # Features
    # =========================
    feature_columns = [
        "predicted_rps",
        "cpu_usage_cores",
        "memory_usage_bytes",
        "latency_p95_ms",
        "latency_avg_ms"
    ] + list(service_dummies.columns)

    X_raw = df[feature_columns].values
    y_raw = df["required_replicas"].values.reshape(-1, 1)

    # =========================
    # Scaling
    # =========================
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    X = input_scaler.fit_transform(X_raw)
    y = output_scaler.fit_transform(y_raw)

    # =========================
    # Save
    # =========================
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    joblib.dump(input_scaler, os.path.join(OUTPUT_DIR, "input_scaler.pkl"))
    joblib.dump(output_scaler, os.path.join(OUTPUT_DIR, "output_scaler.pkl"))

    df.to_csv(os.path.join(OUTPUT_DIR, "dataset_debug.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "feature_columns.txt"), "w") as f:
        for col in feature_columns:
            f.write(col + "\n")

    print("Dataset prepared successfully")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features count: {len(feature_columns)}")
    print(df["required_replicas"].value_counts())

if __name__ == "__main__":
    main()
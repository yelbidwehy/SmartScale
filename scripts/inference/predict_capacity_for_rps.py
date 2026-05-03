import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model2_dataset_v2"
MODEL_PATH = PROJECT_ROOT / "models" / "model2_capacity_nn_v2.pth"

MIN_REPLICAS = 1
MAX_REPLICAS = 10

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


class CapacityReplicaNN(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.model(x)


def clamp_replica(value):
    return max(MIN_REPLICAS, min(MAX_REPLICAS, int(round(value))))


def load_feature_columns():
    feature_file = DATA_DIR / "feature_columns.txt"

    with open(feature_file, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def estimate_metric_by_rps_bucket(debug_df, service, input_rps):
    svc_df = debug_df[debug_df["service"] == service].copy()

    if svc_df.empty:
        return {
            "cpu_usage_cores": 0,
            "memory_usage_bytes": 0,
            "latency_p95_ms": 0,
            "latency_avg_ms": 0,
        }

    rps_col = "predicted_rps" if "predicted_rps" in svc_df.columns else "frontend_rps"

    svc_df = svc_df.sort_values(rps_col)

    # Take LOW RPS baseline
    low_df = svc_df.head(50)
    high_df = svc_df.tail(50)

    low_rps = low_df[rps_col].mean()
    high_rps = high_df[rps_col].mean()

    def interpolate(metric):
        low_val = low_df[metric].mean()
        high_val = high_df[metric].mean()

        if high_rps == low_rps:
            return low_val

        ratio = (input_rps - low_rps) / (high_rps - low_rps)
        ratio = max(0, ratio)  # allow >1 (extrapolation)

        return low_val + ratio * (high_val - low_val)

    cpu = interpolate("cpu_usage_cores")
    memory = interpolate("memory_usage_bytes")
    p95 = interpolate("latency_p95_ms")
    avg = interpolate("latency_avg_ms")

    # 🔥 Critical fix: enforce growth
    BASE_RPS = 50
    scale_factor = max(1.0, input_rps / BASE_RPS)

    cpu = cpu * scale_factor
    memory = memory * scale_factor

    return {
        "cpu_usage_cores": cpu,
        "memory_usage_bytes": memory,
        "latency_p95_ms": p95,
        "latency_avg_ms": avg,
    }


def build_feature_row(feature_columns, service, input_rps, metrics):
    row = {col: 0 for col in feature_columns}

    if "predicted_rps" in row:
        row["predicted_rps"] = input_rps

    if "frontend_rps" in row:
        row["frontend_rps"] = input_rps

    row["cpu_usage_cores"] = metrics["cpu_usage_cores"]
    row["memory_usage_bytes"] = metrics["memory_usage_bytes"]
    row["latency_p95_ms"] = metrics["latency_p95_ms"]
    row["latency_avg_ms"] = metrics["latency_avg_ms"]

    service_col = f"svc_{service}"
    if service_col in row:
        row[service_col] = 1

    return [row[col] for col in feature_columns]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rps",
        type=float,
        required=True,
        help="Expected/predicted frontend RPS, example: --rps 150",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually run kubectl scale commands",
    )

    args = parser.parse_args()
    input_rps = args.rps

    feature_columns = load_feature_columns()

    input_scaler = joblib.load(DATA_DIR / "input_scaler.pkl")
    output_scaler = joblib.load(DATA_DIR / "output_scaler.pkl")
    debug_df = pd.read_csv(DATA_DIR / "dataset_debug.csv")

    model = CapacityReplicaNN(input_size=len(feature_columns), output_size=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print("\n========== Model 2 Manual Replica Prediction ==========")
    print(f"Input predicted RPS : {input_rps}")
    print(f"Features count      : {len(feature_columns)}")
    print("Metric estimation   : RPS bucket/interpolation")
    print("")

    commands = []
    results = []

    for service in TARGET_SERVICES:
        metrics = estimate_metric_by_rps_bucket(debug_df, service, input_rps)

        feature_row = build_feature_row(
            feature_columns=feature_columns,
            service=service,
            input_rps=input_rps,
            metrics=metrics,
        )

        X = np.array([feature_row])
        X_scaled = input_scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()

        pred_real = output_scaler.inverse_transform(pred_scaled)[0][0]
        recommended_replicas = clamp_replica(pred_real)

        results.append({
            "service": service,
            "raw_output": pred_real,
            "recommended_replicas": recommended_replicas,
            **metrics
        })

        print("--------------------------------------------------")
        print(f"Service              : {service}")
        print(f"Estimated CPU        : {metrics['cpu_usage_cores']:.4f} cores")
        print(f"Estimated Memory     : {metrics['memory_usage_bytes'] / (1024 * 1024):.2f} MB")
        print(f"Estimated P95 latency: {metrics['latency_p95_ms']:.2f} ms")
        print(f"Estimated AVG latency: {metrics['latency_avg_ms']:.2f} ms")
        print(f"Raw model output     : {pred_real:.3f}")
        print(f"Recommended replicas : {recommended_replicas}")

        command = f"kubectl scale deployment {service} --replicas={recommended_replicas}"
        commands.append(command)
        print(f"Kubernetes command   : {command}")

    print("\n========== Summary Commands ==========")

    for command in commands:
        print(command)

    print("\n========== Compact Result ==========")
    for item in results:
        print(f"{item['service']:30s} -> {item['recommended_replicas']} replicas")

    if args.apply:
        print("\n========== Applying Scaling Commands ==========")
        for command in commands:
            print(f"Executing: {command}")
            subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
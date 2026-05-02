import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ========================
# Paths
# ========================
MODEL2_DIR = PROJECT_ROOT / "data" / "processed" / "two_model_dataset" / "model2_capacity_per_service"
MODEL2_PATH = PROJECT_ROOT / "models" / "model2_capacity_per_service_nn.pth"

# ========================
# Scaling settings
# ========================
CPU_PER_POD = 0.5
MEMORY_PER_POD_MB = 512
SAFETY_FACTOR = 1.2

MIN_REPLICAS = 1
MAX_REPLICAS = 10

# Services you may not want to scale using deployment
EXCLUDE_FROM_SCALING = [
    "redis-cart"
]


# ========================
# Model 2: Per-service Capacity NN
# ========================
class CapacityPerServiceNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)


def clamp_replicas(value):
    return max(MIN_REPLICAS, min(MAX_REPLICAS, value))


def get_service_name(target_column):
    if target_column.endswith("_cpu"):
        return target_column.replace("_cpu", "")
    if target_column.endswith("_memory"):
        return target_column.replace("_memory", "")
    return target_column


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rps",
        type=float,
        required=True,
        help="Expected or predicted requests per second, example: --rps 80"
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually execute kubectl scale commands. Without this, only prints recommendations."
    )

    args = parser.parse_args()
    input_rps = args.rps

    # ========================
    # Load target columns
    # ========================
    with open(os.path.join(MODEL2_DIR, "target_columns.txt"), "r") as f:
        target_columns = [line.strip() for line in f.readlines()]

    output_size = len(target_columns)

    # ========================
    # Load scalers
    # ========================
    input_scaler = joblib.load(
        os.path.join(MODEL2_DIR, "capacity_input_scaler.pkl")
    )

    output_scaler = joblib.load(
        os.path.join(MODEL2_DIR, "capacity_output_scaler.pkl")
    )

    # ========================
    # Load model
    # ========================
    model = CapacityPerServiceNN(output_size=output_size)
    model.load_state_dict(torch.load(MODEL2_PATH, map_location="cpu"))
    model.eval()

    # ========================
    # Predict per-service capacity
    # ========================
    X = np.array([[input_rps, input_rps ** 2]])
    X_scaled = input_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction_scaled = model(X_tensor).cpu().numpy()

    prediction_real = output_scaler.inverse_transform(prediction_scaled)[0]

    # ========================
    # Build service resource map
    # ========================
    services = {}

    for col, value in zip(target_columns, prediction_real):
        service = get_service_name(col)

        if service not in services:
            services[service] = {
                "cpu": 0.0,
                "memory_bytes": 0.0
            }

        if col.endswith("_cpu"):
            services[service]["cpu"] = max(0.0, float(value))
        elif col.endswith("_memory"):
            services[service]["memory_bytes"] = max(0.0, float(value))

    # ========================
    # Calculate replicas per service
    # ========================
    print("\n========== Per-Service Capacity Prediction ==========")
    print(f"Input RPS: {input_rps:.2f}")
    print(f"CPU per pod: {CPU_PER_POD} cores")
    print(f"Memory per pod: {MEMORY_PER_POD_MB} MB")
    print(f"Safety factor: {SAFETY_FACTOR}")
    print("")

    commands = []

    for service, values in services.items():
        predicted_cpu = values["cpu"]
        predicted_memory_mb = values["memory_bytes"] / (1024 * 1024)

        # ========================
        # Correction layer
        # ========================
        # Because our dataset is still small, the NN may smooth or reduce CPU at higher RPS.
        # This correction helps enforce realistic behavior: higher RPS should need higher CPU.
        BASE_RPS = 50

        if input_rps > BASE_RPS:
            scale_factor = input_rps / BASE_RPS
            predicted_cpu = predicted_cpu * scale_factor

        required_cpu = predicted_cpu * SAFETY_FACTOR
        required_memory_mb = predicted_memory_mb * SAFETY_FACTOR

        replicas_by_cpu = math.ceil(required_cpu / CPU_PER_POD)
        replicas_by_memory = math.ceil(required_memory_mb / MEMORY_PER_POD_MB)

        recommended_replicas = max(replicas_by_cpu, replicas_by_memory)
        recommended_replicas = clamp_replicas(recommended_replicas)

        print("--------------------------------------------------")
        print(f"Service              : {service}")
        print(f"Predicted CPU        : {predicted_cpu:.4f} cores")
        print(f"Predicted Memory     : {predicted_memory_mb:.2f} MB")
        print(f"Required CPU+buffer  : {required_cpu:.4f} cores")
        print(f"Required Mem+buffer  : {required_memory_mb:.2f} MB")
        print(f"Replicas by CPU      : {replicas_by_cpu}")
        print(f"Replicas by Memory   : {replicas_by_memory}")
        print(f"Recommended replicas : {recommended_replicas}")

        if service in EXCLUDE_FROM_SCALING:
            print(f"Kubernetes command   : SKIPPED ({service} excluded)")
            continue

        command = f"kubectl scale deployment {service} --replicas={recommended_replicas}"
        commands.append(command)

        print(f"Kubernetes command   : {command}")

    print("\n========== Summary Commands ==========")

    for command in commands:
        print(command)

    if input_rps > 100:
        print("\n⚠️ Warning:")
        print("Current model was trained roughly on 0–100 RPS.")
        print("Predictions above this range may not be reliable until more high-load data is collected.")

    # ========================
    # Optional execution
    # ========================
    if args.apply:
        print("\n========== Applying Scaling Commands ==========")
        import subprocess

        for command in commands:
            print(f"Executing: {command}")
            result = subprocess.run(command, shell=True)

            if result.returncode != 0:
                print(f"Failed command: {command}")


if __name__ == "__main__":
    main()
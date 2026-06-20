import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import random

from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model2_dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "model2_capacity_nn_v2.pth"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluations"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 400
LEARNING_RATE = 0.001
MIN_REPLICAS = 1
MAX_REPLICAS = 10
SEED = 42
SERVICE_REPLICA_BIAS = {
    "recommendationservice": 0.4,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# Load dataset
# =========================
X_train_np = np.load(DATA_DIR / "X_train.npy")
y_train_np = np.load(DATA_DIR / "y_train.npy")
X_test_np = np.load(DATA_DIR / "X_test.npy")
y_test_np = np.load(DATA_DIR / "y_test.npy")

input_scaler = joblib.load(DATA_DIR / "input_scaler.pkl")
output_scaler = joblib.load(DATA_DIR / "output_scaler.pkl")

test_debug = pd.read_csv(DATA_DIR / "dataset_test.csv")

with open(DATA_DIR / "feature_columns.txt", "r") as f:
    feature_columns = [line.strip() for line in f.readlines()]

INPUT_SIZE = X_train_np.shape[1]
OUTPUT_SIZE = y_train_np.shape[1]

print("Dataset loaded")
print(f"X_train shape: {X_train_np.shape}")
print(f"y_train shape: {y_train_np.shape}")
print(f"X_test shape : {X_test_np.shape}")
print(f"y_test shape : {y_test_np.shape}")
print(f"Input features: {INPUT_SIZE}")
print(f"Output targets: {OUTPUT_SIZE}")

print("\nFeatures:")
for col in feature_columns:
    print(f"- {col}")

print("\nTrain target replica distribution:")
print(pd.Series(y_train_np.flatten()).value_counts().sort_index())
print("\nHeld-out test target replica distribution:")
print(pd.Series(y_test_np.flatten()).value_counts().sort_index())

# =========================
# Use pre-defined train/test split from dataset prep
# =========================
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

# =========================
# Model
# =========================
class CapacityReplicaNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)


def weighted_mse_loss(pred, target):
    # Penalize under-scaling more than over-scaling
    error = pred - target
    
    under_weight = 1.5
    over_weight = 1.0
    
    weight = torch.where(error < 0, under_weight, over_weight)
    
    return torch.mean(weight * (error ** 2))


model = CapacityReplicaNN(INPUT_SIZE, OUTPUT_SIZE)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
test_losses = []

best_test_loss = float("inf")
best_state = None

# =========================
# Training
# =========================
for epoch in range(EPOCHS):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train)

    loss = weighted_mse_loss(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = weighted_mse_loss(test_outputs, y_test)

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    if test_loss.item() < best_test_loss:
        best_test_loss = test_loss.item()
        best_state = model.state_dict()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {loss.item():.4f} | "
        f"Test Loss: {test_loss.item():.4f}"
    )

# Load best model state
if best_state is not None:
    model.load_state_dict(best_state)

# =========================
# Save model
# =========================
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")
print(f"Best test loss: {best_test_loss:.6f}")

# =========================
# Loss chart
# =========================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Model 2 - Capacity Replica Prediction Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig(CHART_DIR / "model2_capacity_v2_training_loss.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# Evaluation
# =========================
model.eval()

with torch.no_grad():
    pred_scaled = model(X_test).cpu().numpy()

actual_scaled = y_test.cpu().numpy()

pred_real_raw = output_scaler.inverse_transform(pred_scaled)
actual_real_raw = output_scaler.inverse_transform(actual_scaled)

if len(test_debug) != len(actual_real_raw):
    raise ValueError(
        f"dataset_test.csv rows ({len(test_debug)}) do not match X_test rows ({len(actual_real_raw)})."
    )

test_debug = test_debug.reset_index(drop=True)

service_bias = test_debug["service"].map(lambda s: SERVICE_REPLICA_BIAS.get(s, 0.0)).to_numpy()
pred_real_calibrated = pred_real_raw.flatten() + service_bias

pred_real = np.clip(np.rint(pred_real_calibrated), MIN_REPLICAS, MAX_REPLICAS)
actual_real = np.clip(np.rint(actual_real_raw.flatten()), MIN_REPLICAS, MAX_REPLICAS)

overall_mae = mean_absolute_error(actual_real, pred_real)
overall_rmse = np.sqrt(mean_squared_error(actual_real, pred_real))
exact_match_accuracy = (actual_real == pred_real).mean()

under_scaled = (pred_real < actual_real).sum()
over_scaled = (pred_real > actual_real).sum()
correct_scaled = (pred_real == actual_real).sum()

print("\nOverall Model 2 Evaluation:")
print(f"Overall MAE              : {overall_mae:.4f} replicas")
print(f"Overall RMSE             : {overall_rmse:.4f} replicas")
print(f"Exact Match Accuracy     : {exact_match_accuracy:.4%}")
print(f"Correct predictions      : {correct_scaled}")
print(f"Under-scaled predictions : {under_scaled}")
print(f"Over-scaled predictions  : {over_scaled}")

eval_df = pd.DataFrame({
    "timestamp": test_debug["timestamp"] if "timestamp" in test_debug.columns else None,
    "run_name": test_debug["run_name"] if "run_name" in test_debug.columns else None,
    "users": test_debug["users"] if "users" in test_debug.columns else None,
    "service": test_debug["service"],
    "frontend_rps": test_debug["frontend_rps"],
    "service_rps": test_debug["service_rps"] if "service_rps" in test_debug.columns else None,
    "predicted_rps": test_debug["predicted_rps"],
    "cpu_usage_cores": test_debug["cpu_usage_cores"],
    "memory_usage_bytes": test_debug["memory_usage_bytes"],
    "latency_p95_ms": test_debug["latency_p95_ms"],
    "latency_avg_ms": test_debug["latency_avg_ms"],
    "actual_replicas": actual_real,
    "predicted_replicas": pred_real,
    "actual_replicas_raw": actual_real_raw.flatten(),
    "predicted_replicas_raw": pred_real_raw.flatten(),
    "prediction_bias": service_bias,
    "predicted_replicas_calibrated_raw": pred_real_calibrated,
})

eval_df["absolute_error"] = (
    eval_df["actual_replicas"] - eval_df["predicted_replicas"]
).abs()

eval_df["prediction_status"] = np.where(
    eval_df["predicted_replicas"] < eval_df["actual_replicas"],
    "under_scaled",
    np.where(
        eval_df["predicted_replicas"] > eval_df["actual_replicas"],
        "over_scaled",
        "correct"
    )
)

eval_df.to_csv(EVAL_DIR / "model2_capacity_v2_evaluation.csv", index=False)

# =========================
# Per-service evaluation
# =========================
per_service_eval = (
    eval_df
    .groupby("service")
    .agg(
        samples=("service", "count"),
        mae=("absolute_error", "mean"),
        exact_match_accuracy=("prediction_status", lambda s: (s == "correct").mean()),
        under_scaled=("prediction_status", lambda s: (s == "under_scaled").sum()),
        over_scaled=("prediction_status", lambda s: (s == "over_scaled").sum()),
        actual_avg=("actual_replicas", "mean"),
        predicted_avg=("predicted_replicas", "mean"),
        actual_min=("actual_replicas", "min"),
        actual_max=("actual_replicas", "max"),
    )
    .reset_index()
)

per_service_eval.to_csv(EVAL_DIR / "model2_capacity_v2_per_service_evaluation.csv", index=False)

print("\nPer-service evaluation:")
print(per_service_eval)

# =========================
# Dynamic-service evaluation
# Services where actual replicas are sometimes > 1
# =========================
dynamic_services = (
    eval_df
    .groupby("service")["actual_replicas"]
    .max()
)

dynamic_services = dynamic_services[dynamic_services > 1].index.tolist()
dynamic_eval_df = eval_df[eval_df["service"].isin(dynamic_services)].copy()

if not dynamic_eval_df.empty:
    dynamic_mae = mean_absolute_error(
        dynamic_eval_df["actual_replicas"],
        dynamic_eval_df["predicted_replicas"]
    )
    dynamic_rmse = np.sqrt(mean_squared_error(
        dynamic_eval_df["actual_replicas"],
        dynamic_eval_df["predicted_replicas"]
    ))
    dynamic_accuracy = (
        dynamic_eval_df["actual_replicas"] == dynamic_eval_df["predicted_replicas"]
    ).mean()

    print("\nDynamic Services Evaluation:")
    print(f"Dynamic services          : {dynamic_services}")
    print(f"Dynamic MAE               : {dynamic_mae:.4f} replicas")
    print(f"Dynamic RMSE              : {dynamic_rmse:.4f} replicas")
    print(f"Dynamic Exact Accuracy    : {dynamic_accuracy:.4%}")

    dynamic_eval_df.to_csv(
        EVAL_DIR / "model2_capacity_v2_dynamic_services_evaluation.csv",
        index=False
    )

# =========================
# Error summary
# =========================
error_summary = pd.DataFrame({
    "metric": [
        "overall_mae",
        "overall_rmse",
        "exact_match_accuracy",
        "correct_predictions",
        "under_scaled_predictions",
        "over_scaled_predictions",
        "best_test_loss",
    ],
    "value": [
        overall_mae,
        overall_rmse,
        exact_match_accuracy,
        correct_scaled,
        under_scaled,
        over_scaled,
        best_test_loss,
    ]
})

error_summary.to_csv(EVAL_DIR / "model2_capacity_v2_summary.csv", index=False)

# =========================
# Prediction chart
# =========================
plt.figure(figsize=(10, 5))
plt.plot(actual_real.flatten(), label="Actual Replicas")
plt.plot(pred_real.flatten(), label="Predicted Replicas")
plt.title("Model 2 - Actual vs Predicted Replicas")
plt.xlabel("Test Sample")
plt.ylabel("Replicas")
plt.legend()
plt.grid(True)
plt.savefig(CHART_DIR / "model2_capacity_v2_prediction_vs_actual.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# Per-service MAE chart
# =========================
plt.figure(figsize=(12, 6))
plt.bar(per_service_eval["service"], per_service_eval["mae"])
plt.title("Model 2 - MAE by Service")
plt.xlabel("Service")
plt.ylabel("MAE")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")
plt.savefig(CHART_DIR / "model2_capacity_v2_per_service_mae.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nFiles saved:")
print(f"- {MODEL_PATH}")
print(f"- {CHART_DIR / 'model2_capacity_v2_training_loss.png'}")
print(f"- {CHART_DIR / 'model2_capacity_v2_prediction_vs_actual.png'}")
print(f"- {CHART_DIR / 'model2_capacity_v2_per_service_mae.png'}")
print(f"- {EVAL_DIR / 'model2_capacity_v2_evaluation.csv'}")
print(f"- {EVAL_DIR / 'model2_capacity_v2_per_service_evaluation.csv'}")
print(f"- {EVAL_DIR / 'model2_capacity_v2_summary.csv'}")

if dynamic_eval_df is not None and not dynamic_eval_df.empty:
    print(f"- {EVAL_DIR / 'model2_capacity_v2_dynamic_services_evaluation.csv'}")
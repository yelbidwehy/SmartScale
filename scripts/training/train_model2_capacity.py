import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "two_model_dataset" / "model2_capacity_per_service"
MODEL_PATH = PROJECT_ROOT / "models" / "model2_capacity_per_service_nn.pth"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluations"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 300
LEARNING_RATE = 0.001
MIN_REPLICAS = 1
MAX_REPLICAS = 10

X = np.load(DATA_DIR / "X_capacity.npy")
y = np.load(DATA_DIR / "y_capacity.npy")

input_scaler = joblib.load(DATA_DIR / "capacity_input_scaler.pkl")
output_scaler = joblib.load(DATA_DIR / "capacity_output_scaler.pkl")

with open(DATA_DIR / "target_columns.txt", "r") as f:
    target_columns = [line.strip() for line in f.readlines()]

with open(DATA_DIR / "feature_columns.txt", "r") as f:
    feature_columns = [line.strip() for line in f.readlines()]

INPUT_SIZE = X.shape[1]
OUTPUT_SIZE = len(target_columns)

print("Dataset loaded")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Input features: {INPUT_SIZE}")
print(f"Features: {feature_columns}")
print(f"Output targets: {OUTPUT_SIZE}")
print(f"Targets: {target_columns}")

if not all(col.endswith("_replicas") for col in target_columns):
    print("\nWARNING:")
    print("Some target columns do not end with '_replicas'.")
    print("Please confirm your dataset preparation script is producing replica targets.\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


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


model = CapacityReplicaNN(
    input_size=INPUT_SIZE,
    output_size=OUTPUT_SIZE
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train)

    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {loss.item():.4f} | "
        f"Test Loss: {test_loss.item():.4f}"
    )

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Model 2 - Replica Prediction Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig(CHART_DIR / "model2_replica_training_loss.png", dpi=300, bbox_inches="tight")
plt.close()

model.eval()

with torch.no_grad():
    pred_scaled = model(X_test).cpu().numpy()

actual_scaled = y_test.cpu().numpy()

pred_real = output_scaler.inverse_transform(pred_scaled)
actual_real = output_scaler.inverse_transform(actual_scaled)

pred_real = np.clip(np.rint(pred_real), MIN_REPLICAS, MAX_REPLICAS)
actual_real = np.clip(np.rint(actual_real), MIN_REPLICAS, MAX_REPLICAS)

overall_mae = mean_absolute_error(actual_real, pred_real)
overall_rmse = np.sqrt(mean_squared_error(actual_real, pred_real))

print("\nOverall Model 2 Evaluation:")
print(f"Overall MAE  : {overall_mae:.4f} replicas")
print(f"Overall RMSE : {overall_rmse:.4f} replicas")

results = []

print("\nPer-Service Replica Evaluation:")

for i, col in enumerate(target_columns):
    actual_col = actual_real[:, i]
    pred_col = pred_real[:, i]

    mae = mean_absolute_error(actual_col, pred_col)
    rmse = np.sqrt(mean_squared_error(actual_col, pred_col))

    results.append({
        "target": col,
        "mae_replicas": mae,
        "rmse_replicas": rmse
    })

    print(f"{col:40s} | MAE: {mae:.4f} replicas | RMSE: {rmse:.4f} replicas")

eval_df = pd.DataFrame(results)
eval_df.to_csv(EVAL_DIR / "model2_replica_evaluation.csv", index=False)

selected_targets = target_columns[:6]

for target in selected_targets:
    idx = target_columns.index(target)

    plt.figure(figsize=(10, 5))
    plt.plot(actual_real[:, idx], label=f"Actual {target}")
    plt.plot(pred_real[:, idx], label=f"Predicted {target}")
    plt.title(f"Replica Prediction - {target}")
    plt.xlabel("Test Sample")
    plt.ylabel("Replicas")
    plt.legend()
    plt.grid(True)
    plt.savefig(CHART_DIR / f"model2_{target}_prediction.png", dpi=300, bbox_inches="tight")
    plt.close()

print("\nFiles saved:")
print(f"- {MODEL_PATH}")
print(f"- {CHART_DIR / 'model2_replica_training_loss.png'}")
print(f"- {EVAL_DIR / 'model2_replica_evaluation.csv'}")
print("- selected replica prediction charts")
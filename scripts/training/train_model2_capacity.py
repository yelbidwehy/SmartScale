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

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model2_dataset_v2"
MODEL_PATH = PROJECT_ROOT / "models" / "model2_capacity_nn_v2.pth"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluations"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 200
LEARNING_RATE = 0.001
MIN_REPLICAS = 1
MAX_REPLICAS = 10

X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")

input_scaler = joblib.load(DATA_DIR / "input_scaler.pkl")
output_scaler = joblib.load(DATA_DIR / "output_scaler.pkl")

debug_df = pd.read_csv(DATA_DIR / "dataset_debug.csv")

feature_columns = [
    "frontend_rps",
    "cpu_usage_cores",
    "memory_usage_bytes",
    "latency_p95_ms",
    "latency_avg_ms"
]

INPUT_SIZE = X.shape[1]
OUTPUT_SIZE = y.shape[1]

print("Dataset loaded")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Input features: {INPUT_SIZE}")
print(f"Output targets: {OUTPUT_SIZE}")
print(f"Features: {feature_columns}")

print("\nTarget replica distribution:")
print(debug_df["required_replicas"].value_counts().sort_index())

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


model = CapacityReplicaNN(INPUT_SIZE, OUTPUT_SIZE)

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
plt.title("Model 2 - Capacity Replica Prediction Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig(CHART_DIR / "model2_capacity_v2_training_loss.png", dpi=300, bbox_inches="tight")
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

eval_df = pd.DataFrame({
    "actual_replicas": actual_real.flatten(),
    "predicted_replicas": pred_real.flatten()
})

eval_df.to_csv(EVAL_DIR / "model2_capacity_v2_evaluation.csv", index=False)

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

print("\nFiles saved:")
print(f"- {MODEL_PATH}")
print(f"- {CHART_DIR / 'model2_capacity_v2_training_loss.png'}")
print(f"- {CHART_DIR / 'model2_capacity_v2_prediction_vs_actual.png'}")
print(f"- {EVAL_DIR / 'model2_capacity_v2_evaluation.csv'}")
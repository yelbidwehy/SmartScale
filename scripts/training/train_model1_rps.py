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

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "two_model_dataset" / "model1_rps_forecast"
MODEL_PATH = PROJECT_ROOT / "models" / "model1_rps_lstm.pth"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluations"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 150
LEARNING_RATE = 0.001
PATIENCE = 15

X = np.load(DATA_DIR / "X_rps.npy")
y = np.load(DATA_DIR / "y_rps.npy")

rps_scaler = joblib.load(DATA_DIR / "rps_scaler.pkl")

print("Dataset loaded")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

indices = np.arange(len(X))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X,
    y,
    indices,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


class RPSLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


model = RPSLSTMModel(input_size=1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
test_losses = []

best_test_loss = float("inf")
best_state = None
patience_counter = 0

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

    if test_loss.item() < best_test_loss:
        best_test_loss = test_loss.item()
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {loss.item():.4f} | "
        f"Test Loss: {test_loss.item():.4f}"
    )

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}")
        break

if best_state is not None:
    model.load_state_dict(best_state)

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")
print(f"Best test loss: {best_test_loss:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Model 1 - RPS Forecasting Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig(CHART_DIR / "model1_rps_training_loss.png", dpi=300, bbox_inches="tight")
plt.close()

model.eval()

with torch.no_grad():
    predictions_scaled = model(X_test).cpu().numpy()

actual_scaled = y_test.cpu().numpy()

pred_real = rps_scaler.inverse_transform(predictions_scaled)
actual_real = rps_scaler.inverse_transform(actual_scaled)

pred_real = np.clip(pred_real, 0, None)

mae = mean_absolute_error(actual_real, pred_real)
rmse = np.sqrt(mean_squared_error(actual_real, pred_real))

print("\nModel 1 Evaluation:")
print(f"MAE  : {mae:.2f} RPS")
print(f"RMSE : {rmse:.2f} RPS")

eval_df = pd.DataFrame({
    "sample_index": idx_test,
    "actual_rps": actual_real.flatten(),
    "predicted_rps": pred_real.flatten(),
})

eval_df["absolute_error"] = (eval_df["actual_rps"] - eval_df["predicted_rps"]).abs()
eval_df["squared_error"] = eval_df["absolute_error"] ** 2

eval_df.to_csv(EVAL_DIR / "model1_rps_evaluation.csv", index=False)

summary_df = pd.DataFrame({
    "metric": ["mae_rps", "rmse_rps", "best_test_loss", "samples"],
    "value": [mae, rmse, best_test_loss, len(eval_df)]
})

summary_df.to_csv(EVAL_DIR / "model1_rps_summary.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(actual_real.flatten(), label="Actual RPS")
plt.plot(pred_real.flatten(), label="Predicted RPS")
plt.title("Model 1 - Predicted vs Actual RPS")
plt.xlabel("Test Sample")
plt.ylabel("Requests per Second")
plt.legend()
plt.grid(True)
plt.savefig(CHART_DIR / "model1_rps_prediction_vs_actual.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.hist(eval_df["absolute_error"], bins=20)
plt.title("Model 1 - Absolute Error Distribution")
plt.xlabel("Absolute Error (RPS)")
plt.ylabel("Count")
plt.grid(True)
plt.savefig(CHART_DIR / "model1_rps_error_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nFiles saved:")
print(f"- {MODEL_PATH}")
print(f"- {CHART_DIR / 'model1_rps_training_loss.png'}")
print(f"- {CHART_DIR / 'model1_rps_prediction_vs_actual.png'}")
print(f"- {CHART_DIR / 'model1_rps_error_distribution.png'}")
print(f"- {EVAL_DIR / 'model1_rps_evaluation.csv'}")
print(f"- {EVAL_DIR / 'model1_rps_summary.csv'}")
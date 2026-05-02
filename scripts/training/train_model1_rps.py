import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "two_model_dataset" / "model1_rps_forecast"
MODEL_PATH = PROJECT_ROOT / "models" / "model1_rps_lstm.pth"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"


EPOCHS = 100
LEARNING_RATE = 0.001

X = np.load(os.path.join(DATA_DIR, "X_rps.npy"))
y = np.load(os.path.join(DATA_DIR, "y_rps.npy"))

rps_scaler = joblib.load(os.path.join(DATA_DIR, "rps_scaler.pkl"))

print("Dataset loaded")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

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


class RPSLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


model = RPSLSTMModel(input_size=1)

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
print("Model saved: model1_rps_lstm.pth")

# ========================
# Plot training loss
# ========================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Model 1 - RPS Forecasting Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig(CHART_DIR / "model1_rps_training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# ========================
# Prediction vs Actual
# ========================
model.eval()

with torch.no_grad():
    predictions_scaled = model(X_test).cpu().numpy()

actual_scaled = y_test.cpu().numpy()

pred_real = rps_scaler.inverse_transform(predictions_scaled)
actual_real = rps_scaler.inverse_transform(actual_scaled)

mae = mean_absolute_error(actual_real, pred_real)
rmse = np.sqrt(mean_squared_error(actual_real, pred_real))

print("\nModel 1 Evaluation:")
print(f"MAE  : {mae:.2f} RPS")
print(f"RMSE : {rmse:.2f} RPS")

plt.figure(figsize=(10, 5))
plt.plot(actual_real.flatten(), label="Actual RPS")
plt.plot(pred_real.flatten(), label="Predicted RPS")
plt.title("Model 1 - Predicted vs Actual RPS")
plt.xlabel("Test Sample")
plt.ylabel("Requests per Second")
plt.legend()
plt.grid(True)
plt.savefig(CHART_DIR / "model1_rps_prediction_vs_actual.png", dpi=300, bbox_inches="tight")
#plt.show()

print("Charts saved:")
print("- model1_rps_training_loss.png")
print("- model1_rps_prediction_vs_actual.png")
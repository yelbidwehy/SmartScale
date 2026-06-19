import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import random


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model1_dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "model1_rps_lstm.pth"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluations"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 150
LEARNING_RATE = 0.001
PATIENCE = 15
VAL_SPLIT = 0.2
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0

# --- CHANGE 1: LR scheduler settings ---
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 7

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

X_train_np = np.load(DATA_DIR / "X_train.npy")
y_train_np = np.load(DATA_DIR / "y_train.npy")
X_test_np = np.load(DATA_DIR / "X_test.npy")
y_test_np = np.load(DATA_DIR / "y_test.npy")

rps_scaler = joblib.load(DATA_DIR / "rps_scaler.pkl")

print("Dataset loaded")
print(f"X_train shape: {X_train_np.shape}")
print(f"y_train shape: {y_train_np.shape}")
print(f"X_test shape : {X_test_np.shape}")
print(f"y_test shape : {y_test_np.shape}")

if X_train_np.size == 0 or X_test_np.size == 0:
    raise ValueError(
        "Train/test arrays are empty. Run prepare_model1_rps_dataset.py first "
        "and check TRAIN_RUNS / TEST_RUNS."
    )

idx_test = np.arange(len(X_test_np))

if len(X_train_np) < 10:
    raise ValueError("Training set is too small to create a validation split safely.")

split_index = int(len(X_train_np) * (1 - VAL_SPLIT))

if split_index <= 0 or split_index >= len(X_train_np):
    raise ValueError("Invalid validation split for current training set size.")

X_fit_np = X_train_np[:split_index]
X_val_np = X_train_np[split_index:]
y_fit_np = y_train_np[:split_index]
y_val_np = y_train_np[split_index:]

print("Validation strategy: chronological holdout from training sequences")

print(f"X_fit shape  : {X_fit_np.shape}")
print(f"y_fit shape  : {y_fit_np.shape}")
print(f"X_val shape  : {X_val_np.shape}")
print(f"y_val shape  : {y_val_np.shape}")

X_fit = torch.tensor(X_fit_np, dtype=torch.float32)
y_fit = torch.tensor(y_fit_np, dtype=torch.float32)
X_val = torch.tensor(X_val_np, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)


class RPSLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


model = RPSLSTMModel(
    input_size=1,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

# --- CHANGE 2: MSE loss instead of SmoothL1Loss ---
# SmoothL1 (Huber) under-weights large errors, which are exactly the high-RPS
# spikes that matter most for a scaling decision. MSE penalizes them quadratically.
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- CHANGE 1 (cont.): LR scheduler, reduces LR when val loss plateaus ---
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
)

train_dataset = torch.utils.data.TensorDataset(X_fit, y_fit)

# --- CHANGE 3: shuffle=True ---
# Chronological train/val split already prevents leakage (val is a later time
# window than fit). Shuffling batches WITHIN the fit set is safe and standard;
# shuffle=False meant every epoch saw batches in identical order, which can
# bias gradient updates toward position-in-sequence rather than RPS dynamics.
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

train_losses = []
val_losses = []
lr_history = []

best_val_loss = float("inf")
best_state = None
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        running_loss += loss.item() * len(X_batch)

    epoch_train_loss = running_loss / len(train_dataset)

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    train_losses.append(epoch_train_loss)
    val_losses.append(val_loss.item())
    lr_history.append(optimizer.param_groups[0]["lr"])

    # step the scheduler on validation loss
    scheduler.step(val_loss.item())

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Val Loss: {val_loss.item():.4f} | "
        f"LR: {current_lr:.6f}"
    )

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}")
        break

if best_state is not None:
    model.load_state_dict(best_state)

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")
print(f"Best val loss: {best_val_loss:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Model 1 - RPS Forecasting Loss (improved)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig(CHART_DIR / "model1_rps_training_loss.png", dpi=300, bbox_inches="tight")
plt.close()

# Optional: plot the LR schedule so you can see when it dropped
plt.figure(figsize=(10, 3))
plt.plot(lr_history, label="Learning rate")
plt.title("Model 1 - Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.savefig(CHART_DIR / "model1_rps_lr_schedule.png", dpi=300, bbox_inches="tight")
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

r2 = r2_score(actual_real, pred_real)


print("\nModel 1 Evaluation:")
print(f"MAE  : {mae:.2f} RPS")
print(f"RMSE : {rmse:.2f} RPS")

print(f"R2   : {r2:.4f}")

eval_df = pd.DataFrame({
    "sample_index": idx_test,
    "actual_rps": actual_real.flatten(),
    "predicted_rps": pred_real.flatten(),
})

eval_df["absolute_error"] = (eval_df["actual_rps"] - eval_df["predicted_rps"]).abs()
eval_df["squared_error"] = eval_df["absolute_error"] ** 2

eval_df.to_csv(EVAL_DIR / "model1_rps_evaluation.csv", index=False)

summary_df = pd.DataFrame({
    "metric": ["mae_rps", "rmse_rps", "r2", "best_val_loss", "samples"],
    "value": [mae, rmse, r2, best_val_loss, len(eval_df)]
})

summary_df.to_csv(EVAL_DIR / "model1_rps_summary.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(actual_real.flatten(), label="Actual RPS")
plt.plot(pred_real.flatten(), label="Predicted RPS")
plt.title("Model 1 - Predicted vs Actual RPS (improved)")
plt.xlabel("Test Sample")
plt.ylabel("Requests per Second")
plt.legend()
plt.grid(True)
plt.savefig(CHART_DIR / "model1_rps_prediction_vs_actual.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.hist(eval_df["absolute_error"], bins=20)
plt.title("Model 1 - Absolute Error Distribution (improved)")
plt.xlabel("Absolute Error (RPS)")
plt.ylabel("Count")
plt.grid(True)
plt.savefig(CHART_DIR / "model1_rps_error_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nFiles saved:")
print(f"- {MODEL_PATH}")
print(f"- {CHART_DIR / 'model1_rps_training_loss.png'}")
print(f"- {CHART_DIR / 'model1_rps_lr_schedule.png'}")
print(f"- {CHART_DIR / 'model1_rps_prediction_vs_actual.png'}")
print(f"- {CHART_DIR / 'model1_rps_error_distribution.png'}")
print(f"- {EVAL_DIR / 'model1_rps_evaluation.csv'}")
print(f"- {EVAL_DIR / 'model1_rps_summary.csv'}")
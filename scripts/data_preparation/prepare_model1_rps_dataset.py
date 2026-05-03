import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "smartscale_training_dataset_cleaned.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "two_model_dataset" / "model1_rps_forecast"

WINDOW_SIZE = 12      # 12 rows × 5 sec = 60 seconds history
PREDICT_STEP = 1      # predict next step = next 5 seconds
TARGET_COLUMN = "frontend_rps"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["frontend_rps"] = pd.to_numeric(df["frontend_rps"], errors="coerce")

    df = df.dropna(subset=["timestamp", "run_name", "frontend_rps"])

    # The cleaned dataset has one row per service per timestamp.
    # For Model 1 we only need one frontend RPS value per timestamp/run.
    rps_df = (
        df.groupby(["run_name", "users", "timestamp"], as_index=False)["frontend_rps"]
        .mean()
    )

    rps_df = rps_df.sort_values(["run_name", "timestamp"])
    rps_df["frontend_rps"] = rps_df["frontend_rps"].ffill().bfill().fillna(0)

    return rps_df


def create_sequences_by_run(df, scaler):
    X_all, y_all = [], []

    for run_name, group in df.groupby("run_name"):
        group = group.sort_values("timestamp")

        values = group[[TARGET_COLUMN]].values
        scaled_values = scaler.transform(values)

        for i in range(len(scaled_values) - WINDOW_SIZE - PREDICT_STEP + 1):
            X_all.append(scaled_values[i:i + WINDOW_SIZE])
            y_all.append(scaled_values[i + WINDOW_SIZE + PREDICT_STEP - 1, 0])

    return np.array(X_all), np.array(y_all).reshape(-1, 1)


def main():
    rps_df = load_and_prepare_data()

    output_csv = OUTPUT_DIR / "model1_rps_dataset.csv"
    rps_df.to_csv(output_csv, index=False)

    scaler = MinMaxScaler()
    scaler.fit(rps_df[[TARGET_COLUMN]].values)

    X, y = create_sequences_by_run(rps_df, scaler)

    np.save(OUTPUT_DIR / "X_rps.npy", X)
    np.save(OUTPUT_DIR / "y_rps.npy", y)

    joblib.dump(scaler, OUTPUT_DIR / "rps_scaler.pkl")

    with open(OUTPUT_DIR / "feature_columns.txt", "w") as f:
        f.write(TARGET_COLUMN + "\n")

    print("Model 1 RPS dataset created successfully.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Rows in RPS time-series: {len(rps_df)}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Predict step: {PREDICT_STEP}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

INPUT_FILE = "./prometheus_export/run1_400users/combined_metrics.csv"
OUTPUT_DIR = "lstm_dataset"

WINDOW_SIZE = 12      # 12 rows × 5 sec = 60 seconds history
PREDICT_STEP = 1      # predict next step
TARGET_COLUMN = "latency_p95_ms"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prepare_data():
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["timestamp", "metric", "value"])

    # CPU and memory have multiple pod rows per timestamp, so we sum them.
    # RPS, latency, and total_requests are already total/single series, so we average them.
    def aggregate_metric(group):
        metric_name = group.name[1]  # group.name = (timestamp, metric)

        if metric_name in ["cpu_usage_cores", "memory_usage_bytes"]:
            return group["value"].sum()
        else:
            return group["value"].mean()

    aggregated_df = (
        df.groupby(["timestamp", "metric"])
          .apply(aggregate_metric, include_groups=False)
          .reset_index(name="value")
    )

    # Convert metric names into columns
    wide_df = aggregated_df.pivot(
        index="timestamp",
        columns="metric",
        values="value"
    ).reset_index()

    wide_df = wide_df.sort_values("timestamp")

    # Fill missing values
    wide_df = wide_df.ffill().bfill()

    # Optional: remove column index name caused by pivot
    wide_df.columns.name = None

    return wide_df


def create_sequences(data, target_index):
    X, y = [], []

    for i in range(len(data) - WINDOW_SIZE - PREDICT_STEP + 1):
        X.append(data[i:i + WINDOW_SIZE])
        y.append(data[i + WINDOW_SIZE + PREDICT_STEP - 1, target_index])

    return np.array(X), np.array(y)


def main():
    wide_df = load_and_prepare_data()

    output_csv = os.path.join(OUTPUT_DIR, "lstm_timeseries.csv")
    wide_df.to_csv(output_csv, index=False)

    feature_columns = [
        col for col in wide_df.columns
        if col != "timestamp"
    ]

    if TARGET_COLUMN not in feature_columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. Available columns: {feature_columns}"
        )

    values = wide_df[feature_columns].values

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    target_index = feature_columns.index(TARGET_COLUMN)

    X, y = create_sequences(scaled_values, target_index)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    with open(os.path.join(OUTPUT_DIR, "feature_columns.txt"), "w") as f:
        for col in feature_columns:
            f.write(col + "\n")

    print("Dataset created successfully.")
    print(f"Rows in time-series: {len(wide_df)}")
    print(f"Features: {feature_columns}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
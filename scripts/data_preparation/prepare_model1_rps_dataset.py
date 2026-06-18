import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "smartscale_training_dataset_cleaned.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "model1_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Config
# =========================================================

WINDOW_SIZE = 12      # 12 rows × 5s = 60 seconds history
PREDICT_STEP = 6      # 6 rows × 5s = predict 30 seconds ahead
TARGET_COLUMN = "frontend_rps"

EXPECTED_INTERVAL_S = 5


# =========================================================
# Data loading & preparation
# =========================================================

def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)

    required_cols = ["timestamp", "run_name", "run_type", "frontend_rps"]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["frontend_rps"] = pd.to_numeric(df["frontend_rps"], errors="coerce")

    df = df.dropna(subset=["timestamp", "run_name", "run_type", "frontend_rps"])

    # Remove zero-load rows: warmup / stop / idle
    df = df[df["frontend_rps"] > 0].copy()

    # Cleaned dataset has one row per service per timestamp.
    # Model 1 needs one global frontend RPS per timestamp/run.
    std_per_ts = (
        df.groupby(["run_name", "run_type", "timestamp"])["frontend_rps"]
        .std()
        .reset_index(name="rps_std")
    )

    warmup_mask = std_per_ts["rps_std"] > 0.01
    n_warmup = int(warmup_mask.sum())

    if n_warmup > 0:
        warmup_detail = std_per_ts.loc[warmup_mask, ["run_name", "run_type", "rps_std"]]
        print(
            f"Dropping {n_warmup} timestamp(s) where frontend_rps differs across services:\n"
            f"{warmup_detail.to_string(index=False)}"
        )

    valid_ts = std_per_ts.loc[
        ~warmup_mask,
        ["run_name", "run_type", "timestamp"]
    ]

    df = df.merge(
        valid_ts,
        on=["run_name", "run_type", "timestamp"],
        how="inner"
    )

    rps_df = (
        df.groupby(["run_name", "run_type", "timestamp"], as_index=False)[TARGET_COLUMN]
        .mean()
    )

    rps_df = rps_df.sort_values(["run_name", "timestamp"]).reset_index(drop=True)

    nan_before = int(rps_df[TARGET_COLUMN].isna().sum())
    rps_df[TARGET_COLUMN] = (
        rps_df.groupby("run_name")[TARGET_COLUMN]
        .transform(lambda s: s.ffill().bfill().fillna(0))
    )

    if nan_before > 0:
        print(f"WARNING: filled {nan_before} NaN value(s) in {TARGET_COLUMN}.")

    return rps_df


# =========================================================
# Timestamp interval validation
# =========================================================

def validate_intervals(df: pd.DataFrame) -> None:
    runs_with_gaps = []

    for run_name, group in df.groupby("run_name"):
        diffs = (
            group["timestamp"]
            .sort_values()
            .diff()
            .dt.total_seconds()
            .dropna()
        )

        bad = diffs[diffs != EXPECTED_INTERVAL_S]

        if not bad.empty:
            runs_with_gaps.append((run_name, bad.value_counts().to_dict()))

    if runs_with_gaps:
        print("WARNING: irregular timestamp gaps detected after filtering zero-load rows.")
        for run_name, gap_counts in runs_with_gaps:
            print(f"  {run_name}: {gap_counts}")
        print("Sequences will be created only within contiguous 5-second segments.")
    else:
        print("Timestamp interval validation passed — all runs have 5-second spacing.")


# =========================================================
# Sequence creation
# =========================================================

def create_sequences_by_run(
    df: pd.DataFrame,
    scaler: MinMaxScaler
) -> tuple[np.ndarray, np.ndarray]:

    X_all = []
    y_all = []

    for run_name, group in df.groupby("run_name"):
        group = group.sort_values("timestamp").copy()
        group["segment_id"] = (
            group["timestamp"]
            .diff()
            .dt.total_seconds()
            .fillna(EXPECTED_INTERVAL_S)
            .ne(EXPECTED_INTERVAL_S)
            .cumsum()
        )

        for segment_id, segment in group.groupby("segment_id"):
            values = segment[[TARGET_COLUMN]].values
            scaled_values = scaler.transform(values)

            max_i = len(scaled_values) - WINDOW_SIZE - PREDICT_STEP + 1

            if max_i <= 0:
                print(
                    f"Skipping {run_name} segment {segment_id}: only {len(scaled_values)} rows, "
                    f"need at least {WINDOW_SIZE + PREDICT_STEP}."
                )
                continue

            for i in range(max_i):
                X_all.append(scaled_values[i: i + WINDOW_SIZE])

                target_idx = i + WINDOW_SIZE + PREDICT_STEP - 1
                y_all.append(scaled_values[target_idx, 0])

    return np.array(X_all), np.array(y_all).reshape(-1, 1)


# =========================================================
# Main
# =========================================================

def main() -> None:
    rps_df = load_and_prepare_data()

    validate_intervals(rps_df)

    if "run_type" not in rps_df.columns:
        raise ValueError("Missing run_type after preparation. Cannot split train/test.")

    run_type_values = sorted(rps_df["run_type"].astype(str).str.lower().unique().tolist())
    expected_types = {"train", "test"}

    if not expected_types.issubset(set(run_type_values)):
        raise ValueError(
            f"Expected run_type to include {expected_types}, found: {run_type_values}"
        )

    train_df = rps_df[rps_df["run_type"].astype(str).str.lower() == "train"].copy()
    test_df = rps_df[rps_df["run_type"].astype(str).str.lower() == "test"].copy()

    train_runs = sorted(train_df["run_name"].unique().tolist())
    test_runs = sorted(test_df["run_name"].unique().tolist())

    if train_df.empty:
        raise ValueError("Training dataframe is empty. Check TRAIN_RUNS.")

    if test_df.empty:
        raise ValueError("Test dataframe is empty. Check TEST_RUNS.")

    print(f"\nTrain runs count : {len(train_runs)}  ({len(train_df)} rows)")
    print(f"Test  runs count : {len(test_runs)}   ({len(test_df)} rows)")

    scaler = MinMaxScaler()
    scaler.fit(train_df[[TARGET_COLUMN]].values)

    print(
        f"Scaler fit on train — "
        f"min={scaler.data_min_[0]:.4f}, max={scaler.data_max_[0]:.4f}"
    )

    test_max = test_df[TARGET_COLUMN].max()
    test_min = test_df[TARGET_COLUMN].min()

    if test_max > scaler.data_max_[0]:
        print(
            f"WARNING: test max RPS ({test_max:.4f}) exceeds training max "
            f"({scaler.data_max_[0]:.4f})."
        )

    if test_min < scaler.data_min_[0]:
        print(
            f"WARNING: test min RPS ({test_min:.4f}) is below training min "
            f"({scaler.data_min_[0]:.4f})."
        )

    X_train, y_train = create_sequences_by_run(train_df, scaler)
    X_test, y_test = create_sequences_by_run(test_df, scaler)

    rps_df.to_csv(OUTPUT_DIR / "model1_rps_dataset.csv", index=False)
    train_df.to_csv(OUTPUT_DIR / "model1_rps_train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "model1_rps_test.csv", index=False)

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    joblib.dump(scaler, OUTPUT_DIR / "rps_scaler.pkl")

    with open(OUTPUT_DIR / "feature_columns.txt", "w") as f:
        f.write(TARGET_COLUMN + "\n")

    print("\n" + "=" * 60)
    print("Model 1 RPS dataset created successfully.")
    print(f"Input file            : {INPUT_FILE}")
    print(f"Total RPS rows        : {len(rps_df)}")
    print(f"  Train rows          : {len(train_df)}")
    print(f"  Test  rows          : {len(test_df)}")
    print(f"Window size           : {WINDOW_SIZE} rows ({WINDOW_SIZE * 5}s history)")
    print(f"Predict step          : {PREDICT_STEP} rows ({PREDICT_STEP * 5}s horizon)")
    print(f"X_train shape         : {X_train.shape}")
    print(f"y_train shape         : {y_train.shape}")
    print(f"X_test  shape         : {X_test.shape}")
    print(f"y_test  shape         : {y_test.shape}")
    print(f"Target column         : {TARGET_COLUMN}")
    print(f"Output directory      : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
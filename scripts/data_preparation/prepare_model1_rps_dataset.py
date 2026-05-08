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
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "two_model_dataset" / "model1_rps_forecast"
)

# =========================================================
# Config
# =========================================================

# FIX: comment corrected from "24 × 5s = 120s" — 12 × 5s = 60 seconds of history.
WINDOW_SIZE = 12      # 12 rows × 5s = 60 seconds of history
PREDICT_STEP = 6      # 6 rows × 5s  = predict 30 seconds ahead
TARGET_COLUMN = "frontend_rps"

# Runs used for training. The remaining run(s) become the held-out test set.
# FIX: scaler was previously fit on the entire dataset before any split,
# leaking test-set statistics into the scaler.
TRAIN_RUNS = ["run_100_users", "run_200_users", "run_300_users"]
TEST_RUNS  = ["run_400_users"]

# Minimum interval between consecutive timestamps (seconds).
# Used to validate that no gap exists in each run's time series.
EXPECTED_INTERVAL_S = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Data loading & preparation
# =========================================================

def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["frontend_rps"] = pd.to_numeric(df["frontend_rps"], errors="coerce")

    df = df.dropna(subset=["timestamp", "run_name", "frontend_rps"])

    # Remove zero-load rows (warmup / stop / idle).
    df = df[df["frontend_rps"] > 0].copy()

    # The cleaned dataset has one row per *service* per timestamp.
    # Model 1 only needs one global frontend RPS per timestamp/run.
    # In most timestamps frontend_rps is identical across all services,
    # but the first 15 timestamps of run_100_users show cross-service
    # variation (std up to 1.12) because adservice reports a different
    # value during warm-up. Averaging those rows silently mixes a
    # warm-up artefact into the series.
    #
    # FIX: detect and drop any timestamp where frontend_rps is not
    # consistent across services (std > threshold), then take the mean
    # of the remaining uniform timestamps.
    std_per_ts = (
        df.groupby(["run_name", "timestamp"])["frontend_rps"]
        .std()
        .reset_index(name="rps_std")
    )

    warmup_mask = std_per_ts["rps_std"] > 0.01
    n_warmup = warmup_mask.sum()
    if n_warmup > 0:
        warmup_detail = std_per_ts.loc[warmup_mask, ["run_name", "rps_std"]]
        print(
            f"Dropping {n_warmup} warm-up timestamp(s) where frontend_rps "
            f"differs across services:\n{warmup_detail.to_string(index=False)}"
        )

    valid_ts = std_per_ts.loc[~warmup_mask, ["run_name", "timestamp"]]
    df = df.merge(valid_ts, on=["run_name", "timestamp"], how="inner")

    rps_df = (
        df.groupby(["run_name", "users", "timestamp"], as_index=False)["frontend_rps"]
        .mean()
    )

    rps_df = rps_df.sort_values(["run_name", "timestamp"]).reset_index(drop=True)

    # Forward/back fill only to handle any isolated NaNs that survive groupby;
    # after the warm-up filter above these should be extremely rare.
    nan_before = rps_df["frontend_rps"].isna().sum()
    rps_df["frontend_rps"] = rps_df["frontend_rps"].ffill().bfill().fillna(0)
    if nan_before > 0:
        print(f"WARNING: filled {nan_before} NaN value(s) in frontend_rps.")

    return rps_df


# =========================================================
# Timestamp interval validation
# =========================================================

def validate_intervals(df: pd.DataFrame) -> None:
    """Assert that every run has perfectly uniform 5-second timestamps."""
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
            raise ValueError(
                f"Irregular timestamp intervals in {run_name}: "
                f"{bad.unique()} seconds (expected {EXPECTED_INTERVAL_S}s). "
                "Check the source data for gaps or duplicates."
            )
    print("Timestamp interval validation passed — all runs have 5-second spacing.")


# =========================================================
# Sequence creation
# =========================================================

def create_sequences_by_run(
    df: pd.DataFrame, scaler: MinMaxScaler
) -> tuple[np.ndarray, np.ndarray]:
    X_all: list = []
    y_all: list = []

    for run_name, group in df.groupby("run_name"):
        group = group.sort_values("timestamp")

        values = group[[TARGET_COLUMN]].values
        scaled_values = scaler.transform(values)

        max_i = len(scaled_values) - WINDOW_SIZE - PREDICT_STEP + 1

        if max_i <= 0:
            print(f"Skipping {run_name}: only {len(scaled_values)} rows, "
                  f"need at least {WINDOW_SIZE + PREDICT_STEP}.")
            continue

        for i in range(max_i):
            X_all.append(scaled_values[i : i + WINDOW_SIZE])
            # FIX: named variable clarifies the off-by-one intent.
            # target_idx is the row that is PREDICT_STEP steps beyond the
            # end of the input window — i.e. 30 seconds into the future.
            target_idx = i + WINDOW_SIZE + PREDICT_STEP - 1
            y_all.append(scaled_values[target_idx, 0])

    return np.array(X_all), np.array(y_all).reshape(-1, 1)


# =========================================================
# Main
# =========================================================

def main() -> None:
    rps_df = load_and_prepare_data()

    validate_intervals(rps_df)

    # -------------------------------------------------------
    # Train / test split — by run, not by row.
    # FIX: splitting after fitting the scaler on the full
    # dataset caused test-set leakage. Scaler is now fit only
    # on training runs.
    # -------------------------------------------------------
    all_runs = sorted(rps_df["run_name"].unique().tolist())
    unrecognised = set(all_runs) - set(TRAIN_RUNS) - set(TEST_RUNS)
    if unrecognised:
        raise ValueError(
            f"Found run(s) not assigned to TRAIN_RUNS or TEST_RUNS: {unrecognised}. "
            "Update the TRAIN_RUNS / TEST_RUNS constants."
        )

    train_df = rps_df[rps_df["run_name"].isin(TRAIN_RUNS)].copy()
    test_df  = rps_df[rps_df["run_name"].isin(TEST_RUNS)].copy()

    print(f"\nTrain runs : {TRAIN_RUNS}  ({len(train_df)} rows)")
    print(f"Test  runs : {TEST_RUNS}   ({len(test_df)} rows)")

    # Fit scaler exclusively on training data.
    scaler = MinMaxScaler()
    scaler.fit(train_df[[TARGET_COLUMN]].values)
    print(
        f"Scaler fit on train — "
        f"min={scaler.data_min_[0]:.4f}, max={scaler.data_max_[0]:.4f}"
    )

    # Warn if the test set exceeds the training range (extrapolation risk).
    test_max = test_df[TARGET_COLUMN].max()
    test_min = test_df[TARGET_COLUMN].min()
    if test_max > scaler.data_max_[0]:
        print(
            f"WARNING: test set max RPS ({test_max:.4f}) exceeds training max "
            f"({scaler.data_max_[0]:.4f}) — MinMaxScaler will clamp to 1.0."
        )
    if test_min < scaler.data_min_[0]:
        print(
            f"WARNING: test set min RPS ({test_min:.4f}) is below training min "
            f"({scaler.data_min_[0]:.4f}) — MinMaxScaler will clamp to 0.0."
        )

    # -------------------------------------------------------
    # Build sequences for train and test separately.
    # -------------------------------------------------------
    X_train, y_train = create_sequences_by_run(train_df, scaler)
    X_test,  y_test  = create_sequences_by_run(test_df,  scaler)

    # -------------------------------------------------------
    # Save outputs.
    # -------------------------------------------------------
    rps_df.to_csv(OUTPUT_DIR / "model1_rps_dataset.csv", index=False)
    train_df.to_csv(OUTPUT_DIR / "model1_rps_train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "model1_rps_test.csv",  index=False)

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_test.npy",  X_test)
    np.save(OUTPUT_DIR / "y_test.npy",  y_test)

    # Keep legacy filenames so downstream scripts don't break immediately;
    # they point to training data only (the most common use case).
    np.save(OUTPUT_DIR / "X_rps.npy", X_train)
    np.save(OUTPUT_DIR / "y_rps.npy", y_train)

    joblib.dump(scaler, OUTPUT_DIR / "rps_scaler.pkl")

    with open(OUTPUT_DIR / "feature_columns.txt", "w") as f:
        f.write(TARGET_COLUMN + "\n")

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
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
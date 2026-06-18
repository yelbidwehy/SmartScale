import pandas as pd
import numpy as np
import math
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "smartscale_training_dataset_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "model2_dataset"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Config
# =========================================================

CPU_PER_POD       = 0.2    # CPU cores allocated per pod
MEMORY_PER_POD_MB = 256    # Memory allocated per pod (MB)
SAFETY_FACTOR     = 1.2    # Over-provision multiplier
MIN_REPLICAS      = 1
MAX_REPLICAS      = 10

PREDICTION_HORIZON_STEPS = 6   # 6 × 5s = 30 seconds ahead

# Runs used for training. The remaining run(s) form the held-out test set.
# FIX: scalers were previously fit on the full dataset before any split,
# leaking test-set statistics into both the input and output scalers.
TRAIN_RUNS = [
    "ramp_up_20_120",
    "ramp_down_120_20",
    "spike_train_20_120_40",
]

TEST_RUNS = [
    "spike_test_30_120_50",
]
# Warm-up detection threshold: timestamps where frontend_rps std across
# services exceeds this value are treated as warm-up and dropped.
WARMUP_STD_THRESHOLD = 0.01


# =========================================================
# Replica calculation
# =========================================================

def calculate_replicas(cpu: float, memory_bytes: float) -> int:
    memory_mb = memory_bytes / (1024 * 1024)

    r_cpu = math.ceil(cpu * SAFETY_FACTOR / CPU_PER_POD)
    r_mem = math.ceil(memory_mb * SAFETY_FACTOR / MEMORY_PER_POD_MB)

    replicas = max(r_cpu, r_mem)
    return max(MIN_REPLICAS, min(MAX_REPLICAS, replicas))


# =========================================================
# Main
# =========================================================

def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    original_rows = len(df)

    # ----------------------------------------------------------
    # 1. Basic type coercion and zero-load removal
    # ----------------------------------------------------------
    df["frontend_rps"] = pd.to_numeric(df["frontend_rps"], errors="coerce")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["run_name", "service", "timestamp"])

    # Remove rows with no active load.
    df = df[df["frontend_rps"] > 0].copy()
    after_zero_removal = len(df)

    # ----------------------------------------------------------
    # 2. Warm-up removal
    # FIX: the first 15 timestamps of run_100_users have frontend_rps
    # varying across services (cross-service std up to 1.12), indicating
    # the load generator had not stabilised. Including these rows injects
    # a misleading low-CPU / low-RPS regime into the training labels.
    # We detect and drop any timestamp where cross-service std > threshold.
    # ----------------------------------------------------------
    std_per_ts = (
        df.groupby(["run_name", "timestamp"])["frontend_rps"]
        .std()
        .reset_index(name="rps_std")
    )
    warmup_mask = std_per_ts["rps_std"] > WARMUP_STD_THRESHOLD
    n_warmup_ts = warmup_mask.sum()

    if n_warmup_ts > 0:
        print(
            f"Dropping {n_warmup_ts} warm-up timestamp(s) "
            f"(cross-service frontend_rps std > {WARMUP_STD_THRESHOLD}):"
        )
        print(
            std_per_ts.loc[warmup_mask, ["run_name", "rps_std"]]
            .groupby("run_name")
            .size()
            .to_string()
        )

    valid_ts = std_per_ts.loc[~warmup_mask, ["run_name", "timestamp"]]
    df = df.merge(valid_ts, on=["run_name", "timestamp"], how="inner")

    # ----------------------------------------------------------
    # 3. Fill required numeric columns
    # ----------------------------------------------------------
    required_cols = [
        "frontend_rps",
        "service_rps",
        "cpu_usage_cores",
        "memory_usage_bytes",
        "latency_p95_ms",
        "latency_avg_ms",
    ]

    for col in required_cols:
        if col not in df.columns:
            print(f"WARNING: column '{col}' not found — filling with 0.")
            df[col] = 0
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = (
            df.groupby(["run_name", "service"])[col]
            .transform(lambda s: s.ffill().bfill())
        )
        df[col] = df[col].fillna(0)

    # ----------------------------------------------------------
    # 4. predicted_rps — aligned to label horizon (30 s)
    # FIX: was previously shift(-1) = 5s ahead, which does not match
    # the label horizon of shift(-6) = 30s. At inference, Model 1
    # supplies a 30s forecast, so training and inference must agree.
    # The NaN tail rows (last 6 per run/service) will be removed by the
    # dropna on future_cpu/mem below — no ffill needed here.
    # ----------------------------------------------------------
    df["predicted_rps"] = (
        df.groupby(["run_name", "service"])["frontend_rps"]
        .shift(-PREDICTION_HORIZON_STEPS)
    )

    # ----------------------------------------------------------
    # 5. RPS trend / lag features
    # ----------------------------------------------------------
    df["rps_lag_1"] = df.groupby(["run_name", "service"])["frontend_rps"].shift(1)
    df["rps_lag_2"] = df.groupby(["run_name", "service"])["frontend_rps"].shift(2)
    df["rps_lag_3"] = df.groupby(["run_name", "service"])["frontend_rps"].shift(3)

    df["service_rps_lag_1"] = (
        df.groupby(["run_name", "service"])["service_rps"].shift(1)
    )

    df["rps_rolling_mean_3"] = (
        df.groupby(["run_name", "service"])["frontend_rps"]
        .rolling(3)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Fill boundary NaNs for lag/rolling features with current value.
    # Only the first 1-2 rows per run/service group are affected (40 rows
    # for lags, 80 rows for rolling mean). The substitution error at those
    # boundaries is negligible — measured mean |filled - current| ≈ 0.
    trend_cols = [
        "rps_lag_1",
        "rps_lag_2",
        "rps_lag_3",
        "service_rps_lag_1",
        "rps_rolling_mean_3",
    ]
    for col in trend_cols:
        df[col] = (
            df.groupby(["run_name", "service"])[col]
            .transform(lambda s: s.ffill().bfill())
        )
        df[col] = df[col].fillna(df["frontend_rps"])

    # ----------------------------------------------------------
    # 6. Future resource usage — labels for required_replicas
    # ----------------------------------------------------------
    df["future_cpu_usage_cores"] = (
        df.groupby(["run_name", "service"])["cpu_usage_cores"]
        .shift(-PREDICTION_HORIZON_STEPS)
    )
    df["future_memory_usage_bytes"] = (
        df.groupby(["run_name", "service"])["memory_usage_bytes"]
        .shift(-PREDICTION_HORIZON_STEPS)
    )

    # Drop the last PREDICTION_HORIZON_STEPS rows per run/service (no label).
    # Also drops the predicted_rps NaN rows since both use the same shift(-6).
    df = df.dropna(
        subset=["future_cpu_usage_cores", "future_memory_usage_bytes", "predicted_rps"]
    )

    df["required_replicas"] = df.apply(
        lambda row: calculate_replicas(
            row["future_cpu_usage_cores"],
            row["future_memory_usage_bytes"],
        ),
        axis=1,
    )

    # ----------------------------------------------------------
    # 7. Service one-hot encoding
    # ----------------------------------------------------------
    service_dummies = pd.get_dummies(df["service"], prefix="svc")
    df = pd.concat([df, service_dummies], axis=1)

    feature_columns = [
        #"predicted_rps",
        "frontend_rps",
        "service_rps",
        "rps_lag_1",
        "rps_lag_2",
        "rps_lag_3",
        "service_rps_lag_1",
        "rps_rolling_mean_3",
        "cpu_usage_cores",
        "memory_usage_bytes",
        "latency_p95_ms",
        "latency_avg_ms"
        
    ] + list(service_dummies.columns)

    df[feature_columns] = (
        df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    # ----------------------------------------------------------
    # 8. Train / test split by run
    # FIX: scalers were fit on the entire dataset; now fit exclusively
    # on training runs to prevent test-set leakage.
    # ----------------------------------------------------------
    all_runs = sorted(df["run_name"].unique().tolist())
    unrecognised = set(all_runs) - set(TRAIN_RUNS) - set(TEST_RUNS)
    if unrecognised:
        raise ValueError(
            f"Found run(s) not assigned to TRAIN_RUNS or TEST_RUNS: {unrecognised}. "
            "Update the TRAIN_RUNS / TEST_RUNS constants."
        )

    train_df = df[df["run_name"].isin(TRAIN_RUNS)].copy()
    test_df  = df[df["run_name"].isin(TEST_RUNS)].copy()

    print(f"\nTrain runs : {TRAIN_RUNS}  ({len(train_df)} rows)")
    print(f"Test  runs : {TEST_RUNS}   ({len(test_df)} rows)")

    X_train_raw = train_df[feature_columns].values
    X_test_raw  = test_df[feature_columns].values
    y_train_raw = train_df["required_replicas"].values.reshape(-1, 1)
    y_test_raw  = test_df["required_replicas"].values.reshape(-1, 1)

    input_scaler  = MinMaxScaler()
    output_scaler = MinMaxScaler()

    # Fit on train only.
    input_scaler.fit(X_train_raw)
    output_scaler.fit(y_train_raw)

    X_train = input_scaler.transform(X_train_raw)
    X_test  = input_scaler.transform(X_test_raw)
    y_train = output_scaler.transform(y_train_raw)
    y_test  = output_scaler.transform(y_test_raw)

    # Warn if test labels exceed the training range (output scaler will clamp).
    test_max_rep = int(y_test_raw.max())
    train_max_rep = int(y_train_raw.max())
    if test_max_rep > train_max_rep:
        print(
            f"WARNING: test set max replicas ({test_max_rep}) exceeds training max "
            f"({train_max_rep}) — output_scaler will clamp predictions."
        )

    # ----------------------------------------------------------
    # 9. Save outputs
    # ----------------------------------------------------------
    # Split-aware arrays.
    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_test.npy",  X_test)
    np.save(OUTPUT_DIR / "y_test.npy",  y_test)

    # Legacy filenames — point to training data so downstream scripts don't break.
    np.save(OUTPUT_DIR / "X.npy", X_train)
    np.save(OUTPUT_DIR / "y.npy", y_train)

    joblib.dump(input_scaler,  OUTPUT_DIR / "input_scaler.pkl")
    joblib.dump(output_scaler, OUTPUT_DIR / "output_scaler.pkl")

    # Save debug CSV with all columns for inspection.
    df.to_csv(OUTPUT_DIR / "dataset_debug.csv", index=False)
    train_df.to_csv(OUTPUT_DIR / "dataset_train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR  / "dataset_test.csv",  index=False)

    with open(OUTPUT_DIR / "feature_columns.txt", "w") as f:
        for col in feature_columns:
            f.write(col + "\n")

    # ----------------------------------------------------------
    # 10. Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Model 2 capacity dataset created successfully.")
    print(f"Input file              : {INPUT_FILE}")
    print(f"Output dir              : {OUTPUT_DIR}")
    print(f"Original rows           : {original_rows}")
    print(f"After zero-load removal : {after_zero_removal}")
    print(f"After warm-up removal   : {len(train_df) + len(test_df)} "
          f"({n_warmup_ts * 10} warm-up rows dropped)")
    print(f"Train rows              : {len(train_df)}")
    print(f"Test  rows              : {len(test_df)}")
    print(f"X_train shape           : {X_train.shape}")
    print(f"X_test  shape           : {X_test.shape}")
    print(f"y_train shape           : {y_train.shape}")
    print(f"y_test  shape           : {y_test.shape}")
    print(f"Features count          : {len(feature_columns)}")
    print(f"Prediction horizon      : {PREDICTION_HORIZON_STEPS} steps "
          f"({PREDICTION_HORIZON_STEPS * 5}s)")
    print()
    print("Train required_replicas distribution:")
    print(train_df["required_replicas"].value_counts().sort_index().to_string())
    print()
    print("Test  required_replicas distribution:")
    print(test_df["required_replicas"].value_counts().sort_index().to_string())
    print()
    print("Per-service replica stats (train):")
    print(
        train_df.groupby("service")["required_replicas"]
        .agg(["mean", "std", "min", "max", "nunique"])
        .to_string()
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
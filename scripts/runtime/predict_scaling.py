import time
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import logging

from pathlib import Path
from collections import deque

# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("smartscale_inference.log"),
    ],
)
log = logging.getLogger(__name__)

# =========================================================
# Config
# =========================================================

PROMETHEUS_URL = "http://localhost:9090"

WINDOW_SIZE = 12
PREDICTION_INTERVAL_SECONDS = 5

MODEL1_PREDICT_STEP_SECONDS = 30

SERVICES = [
    "frontend",
    "cartservice",
    "productcatalogservice",
    "recommendationservice",
    "checkoutservice",
    "currencyservice",
    "paymentservice",
    "shippingservice",
    "emailservice",
    "adservice",
]

# Number of historical per-service RPS values to retain for lag features.
# Needs at least 4 slots: current + lag1 + lag2 + lag3.
SERVICE_RPS_HISTORY_LEN = 4

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL1_PATH = PROJECT_ROOT / "models" / "model1_rps_lstm.pth"
MODEL2_PATH = PROJECT_ROOT / "models" / "model2_capacity_nn_v2.pth"

MODEL1_SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "two_model_dataset"
    / "model1_rps_forecast"
    / "rps_scaler.pkl"
)

MODEL2_INPUT_SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model2_dataset_v2"
    / "input_scaler.pkl"
)

MODEL2_OUTPUT_SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model2_dataset_v2"
    / "output_scaler.pkl"
)

FEATURE_COLUMNS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model2_dataset_v2"
    / "feature_columns.txt"
)

# =========================================================
# Load scalers
# =========================================================

model1_scaler = joblib.load(MODEL1_SCALER_PATH)
model2_input_scaler = joblib.load(MODEL2_INPUT_SCALER_PATH)
model2_output_scaler = joblib.load(MODEL2_OUTPUT_SCALER_PATH)

with open(FEATURE_COLUMNS_PATH, "r") as f:
    FEATURE_COLUMNS = [line.strip() for line in f.readlines()]

# Cache training-time scaler bounds for OOD warnings.
MODEL1_RPS_MAX = float(model1_scaler.data_max_[0])
MODEL1_RPS_MIN = float(model1_scaler.data_min_[0])

# =========================================================
# Model 1
# =========================================================

class RPSLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# =========================================================
# Model 2
# =========================================================

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
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.model(x)


# =========================================================
# Load models
# =========================================================

model1 = RPSLSTMModel()
model1.load_state_dict(torch.load(MODEL1_PATH, map_location="cpu"))
model1.eval()

model2 = CapacityReplicaNN(input_size=len(FEATURE_COLUMNS), output_size=1)
model2.load_state_dict(torch.load(MODEL2_PATH, map_location="cpu"))
model2.eval()

log.info("Models loaded successfully.")

# =========================================================
# Prometheus helper
# =========================================================

def query_prometheus(query: str) -> list:
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if data["status"] != "success":
        raise RuntimeError(f"Prometheus query failed: {query}")
    return data["data"]["result"]


# =========================================================
# Fetch frontend RPS
# =========================================================

def get_frontend_rps() -> float:
    query = """
sum(
  rate(
    istio_requests_total{
      reporter="source",
      source_workload="istio-gateway-istio",
      destination_workload="frontend"
    }[1m]
  )
)
"""
    result = query_prometheus(query)
    if not result:
        return 0.0
    return float(result[0]["value"][1])


# =========================================================
# Fetch per-service metrics
# =========================================================

def get_service_metrics(service_name: str) -> dict:
    queries = {
        "service_rps": f"""
sum(
  rate(
    istio_requests_total{{
      reporter="destination",
      destination_workload="{service_name}"
    }}[1m]
  )
)
""",
        "latency_p95_ms": f"""
histogram_quantile(
  0.95,
  sum by (le) (
    rate(
      istio_request_duration_milliseconds_bucket{{
        reporter="destination",
        destination_workload="{service_name}"
      }}[1m]
    )
  )
)
""",
        "latency_avg_ms": f"""
sum(
  rate(
    istio_request_duration_milliseconds_sum{{
      reporter="destination",
      destination_workload="{service_name}"
    }}[1m]
  )
)
/
sum(
  rate(
    istio_request_duration_milliseconds_count{{
      reporter="destination",
      destination_workload="{service_name}"
    }}[1m]
  )
)
""",
        "cpu_usage_cores": f"""
sum(
  rate(
    container_cpu_usage_seconds_total{{
      namespace="default",
      container!="POD",
      container!="istio-proxy",
      pod=~"{service_name}.*"
    }}[1m]
  )
)
""",
        "memory_usage_bytes": f"""
sum(
  container_memory_working_set_bytes{{
    namespace="default",
    container!="POD",
    container!="istio-proxy",
    pod=~"{service_name}.*"
  }}
)
""",
    }

    metrics: dict = {}
    for key, query in queries.items():
        try:
            result = query_prometheus(query)
            if result:
                value = float(result[0]["value"][1])
                metrics[key] = 0.0 if (np.isnan(value) or np.isinf(value)) else value
            else:
                metrics[key] = 0.0
        except requests.RequestException as exc:
            # Network / Prometheus unavailability — recoverable, use 0.
            log.warning("Prometheus fetch failed for %s[%s]: %s", service_name, key, exc)
            metrics[key] = 0.0

    # Reset stale latency when there is almost no traffic.
    if metrics.get("service_rps", 0) < 1:
        metrics["latency_p95_ms"] = 0.0
        metrics["latency_avg_ms"] = 0.0

    return metrics


# =========================================================
# History buffers
# =========================================================

# Frontend RPS ring buffer — used by Model 1 and for frontend lag features.
rps_history: deque = deque(maxlen=WINDOW_SIZE)

# Per-service RPS ring buffer — used to compute lag features for Model 2.
# FIX #1: was previously missing; lag features were all set to current_rps,
# destroying the trend signal the model was trained to use.
service_rps_history: dict[str, deque] = {
    svc: deque(maxlen=SERVICE_RPS_HISTORY_LEN) for svc in SERVICES
}

# =========================================================
# Warmup: fill rps_history with active-traffic samples only.
# FIX #4: original loop appended zero-RPS values, producing an
# out-of-distribution input window for Model 1.
# =========================================================

log.info("Collecting initial RPS history (%d samples, traffic-only)...", WINDOW_SIZE)

while len(rps_history) < WINDOW_SIZE:
    rps = get_frontend_rps()
    if rps > 0:
        rps_history.append(rps)
        log.info("Buffered RPS sample %d/%d: %.2f", len(rps_history), WINDOW_SIZE, rps)
    else:
        log.info("Skipping zero-RPS sample (no active traffic yet).")
    time.sleep(PREDICTION_INTERVAL_SECONDS)

log.info("Warmup complete. Starting prediction loop.\n")

# =========================================================
# Runtime prediction loop
# =========================================================

while True:
    try:

        # --------------------------------------------------
        # 1. Fetch current frontend RPS
        # --------------------------------------------------
        current_rps = get_frontend_rps()

        # Only advance the history buffer during active traffic.
        # This prevents OOD windows during traffic lulls.
        if current_rps > 0:
            rps_history.append(current_rps)

        log.info("=" * 80)
        log.info("Current frontend RPS: %.2f", current_rps)

        # --------------------------------------------------
        # 2. Model 1 — predict RPS 30 s ahead
        # --------------------------------------------------

        # FIX #5: Warn when current RPS is outside the training range.
        # MinMaxScaler clips silently; values above the training max will be
        # clamped to 1.0, making predictions unreliable.
        if current_rps > MODEL1_RPS_MAX:
            log.warning(
                "RPS %.2f exceeds training max %.2f — Model 1 prediction may be unreliable.",
                current_rps,
                MODEL1_RPS_MAX,
            )
        elif current_rps < MODEL1_RPS_MIN:
            log.warning(
                "RPS %.2f is below training min %.2f — Model 1 prediction may be unreliable.",
                current_rps,
                MODEL1_RPS_MIN,
            )

        history_array = np.array(rps_history).reshape(-1, 1)
        scaled_history = model1_scaler.transform(history_array)

        X_model1 = torch.tensor(
            scaled_history.reshape(1, WINDOW_SIZE, 1),
            dtype=torch.float32,
        )

        with torch.no_grad():
            pred_scaled = model1(X_model1).numpy()

        predicted_rps = float(model1_scaler.inverse_transform(pred_scaled)[0][0])
        predicted_rps = max(predicted_rps, 0.0)

        log.info("Predicted frontend RPS (+30s): %.2f", predicted_rps)

        # --------------------------------------------------
        # 3. Model 2 — predict required replicas per service
        # --------------------------------------------------

        # FIX #2: Use the last 3 frontend RPS values for rolling mean,
        # matching the training-time rolling(3) window.
        # Original code used np.mean(rps_history) which is a 12-step mean.
        rps_history_list = list(rps_history)
        rps_rolling_mean_3 = float(np.mean(rps_history_list[-3:]))

        # FIX #1 (frontend lags): use rps_history ring buffer instead of
        # setting all lags to current_rps, which destroyed the trend signal.
        frontend_lag_1 = rps_history_list[-2] if len(rps_history_list) >= 2 else current_rps
        frontend_lag_2 = rps_history_list[-3] if len(rps_history_list) >= 3 else current_rps
        frontend_lag_3 = rps_history_list[-4] if len(rps_history_list) >= 4 else current_rps

        scaling_results = []

        for service in SERVICES:

            metrics = get_service_metrics(service)

            # FIX #1 (per-service lag): update per-service history and derive
            # the lag feature from it. Original code set service_rps_lag_1 to
            # the current service_rps, erasing 5s of trend information.
            service_rps_history[service].append(metrics["service_rps"])
            svc_hist = list(service_rps_history[service])
            service_rps_lag_1 = svc_hist[-2] if len(svc_hist) >= 2 else metrics["service_rps"]

            # FIX #6: Build the one-hot encoding explicitly so the active
            # service is always set correctly regardless of dict init order.
            # Original code relied on a fragile key-existence check.
            svc_onehot = {
                col: (1 if col == f"svc_{service}" else 0)
                for col in FEATURE_COLUMNS
                if col.startswith("svc_")
            }

            feature_row = {
                # FIX #3: predicted_rps should carry the Model 1 output (30s
                # horizon). This matches the intended pipeline design, though
                # Model 2 was trained with shift(-1) (5s). Retrain Model 2
                # with shift(-6) in prepare_model2_capacity_dataset.py to
                # fully resolve the horizon mismatch.
                "predicted_rps": predicted_rps,
                "frontend_rps": current_rps,
                "service_rps": metrics["service_rps"],
                "rps_lag_1": frontend_lag_1,
                "rps_lag_2": frontend_lag_2,
                "rps_lag_3": frontend_lag_3,
                "service_rps_lag_1": service_rps_lag_1,
                "rps_rolling_mean_3": rps_rolling_mean_3,
                "cpu_usage_cores": metrics["cpu_usage_cores"],
                "memory_usage_bytes": metrics["memory_usage_bytes"],
                "latency_p95_ms": metrics["latency_p95_ms"],
                "latency_avg_ms": metrics["latency_avg_ms"],
                **svc_onehot,
            }

            X_features = np.array(
                [feature_row.get(col, 0.0) for col in FEATURE_COLUMNS],
                dtype=np.float64,
            ).reshape(1, -1)

            X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

            X_scaled = model2_input_scaler.transform(X_features)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            with torch.no_grad():
                pred_scaled_m2 = model2(X_tensor).numpy()

            replicas = float(model2_output_scaler.inverse_transform(pred_scaled_m2)[0][0])
            replicas = int(round(replicas))
            replicas = max(1, min(10, replicas))

            scaling_results.append(
                {
                    "service": service,
                    "predicted_replicas": replicas,
                    "service_rps": round(metrics["service_rps"], 2),
                    "cpu": round(metrics["cpu_usage_cores"], 3),
                    "memory_mb": round(metrics["memory_usage_bytes"] / (1024 * 1024), 1),
                    "latency_p95_ms": round(metrics["latency_p95_ms"], 2),
                }
            )

        # --------------------------------------------------
        # 4. Print results
        # --------------------------------------------------
        results_df = pd.DataFrame(scaling_results)
        log.info("\nPredicted scaling decisions:\n%s", results_df.to_string(index=False))

        time.sleep(PREDICTION_INTERVAL_SECONDS)

    # FIX #7: Distinguish recoverable network errors from unexpected model
    # errors. Network failures skip the cycle; anything else is re-raised so
    # it surfaces immediately rather than being silently swallowed.
    except requests.RequestException as exc:
        log.error("Prometheus unreachable — skipping cycle: %s", exc)
        time.sleep(PREDICTION_INTERVAL_SECONDS)

    except Exception as exc:
        log.exception("Unexpected error — halting: %s", exc)
        raise
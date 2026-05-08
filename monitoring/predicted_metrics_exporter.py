import time
import math
import logging
from pathlib import Path
from collections import deque, defaultdict

import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from prometheus_client import start_http_server, Gauge
import os
from pathlib import Path

# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# =========================================================
# Paths
# =========================================================

if os.name != "nt" and Path("/app").exists():
    PROJECT_ROOT = Path("/app")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL1_DIR = PROJECT_ROOT / "data" / "processed" / "two_model_dataset" / "model1_rps_forecast"
MODEL2_DIR = PROJECT_ROOT / "data" / "processed" / "model2_dataset_v2"

MODEL1_PATH = PROJECT_ROOT / "models" / "model1_rps_lstm.pth"
MODEL2_PATH = PROJECT_ROOT / "models" / "model2_capacity_nn_v2.pth"


# =========================================================
# Config
# =========================================================

PROMETHEUS_URL = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"
# PROMETHEUS_URL = "http://localhost:9090"

EXPORTER_PORT = 8000

WINDOW_SIZE = 12
PREDICTION_INTERVAL_SECONDS = 5

MIN_REPLICAS = 1
MAX_REPLICAS = 10

SMOOTHING_ALPHA = 0.3
SCALE_DOWN_COOLDOWN_SECONDS = 60

TRAINING_MAX_RPS = 63.1166

SERVICES = [
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "shippingservice",
]

EXCLUDE_FROM_SCALING = [
    # "redis-cart"
]


# =========================================================
# Runtime State
# =========================================================

frontend_rps_history = deque(maxlen=WINDOW_SIZE)
service_rps_history = defaultdict(lambda: deque(maxlen=3))

previous_smoothed_rps = None
last_replicas = {}
last_scale_down_time = {}


# =========================================================
# Prometheus Metrics Exposed to KEDA
# =========================================================

predicted_rps_gauge = Gauge(
    "predicted_requests_per_second",
    "Predicted frontend RPS 30 seconds ahead"
)

predicted_replicas_gauge = Gauge(
    "predicted_replicas",
    "Recommended predicted replicas per service",
    ["service"]
)

current_frontend_rps_gauge = Gauge(
    "current_frontend_rps",
    "Current frontend RPS"
)

model2_raw_prediction_gauge = Gauge(
    "model2_raw_predicted_replicas",
    "Raw Model 2 predicted replicas before rounding/stabilization",
    ["service"]
)


# =========================================================
# Models
# =========================================================

class RPSLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
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


class CapacityNNV2(nn.Module):
    def __init__(self, input_size=22):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


# =========================================================
# Prometheus Helpers
# =========================================================

def prometheus_query(query: str) -> float:
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=10
    )
    response.raise_for_status()

    data = response.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {data}")

    result = data["data"]["result"]
    if not result:
        return 0.0

    value = result[0]["value"][1]
    if value in ["NaN", "nan", "+Inf", "-Inf"]:
        return 0.0

    return float(value)


def get_frontend_rps() -> float:
    query = """
    sum(
      rate(
        istio_requests_total{
          destination_app="frontend",
          response_code!~"5..",
          request_operation!~"health|metrics",
          request_path!~"/health.*|/metrics.*|/ready.*|/live.*"
        }[1m]
      )
    )
    """
    return max(0.0, prometheus_query(query))


def get_service_rps(service: str) -> float:
    query = f"""
    sum(
      rate(
        istio_requests_total{{
          destination_app="{service}",
          request_operation!~"health|metrics",
          request_path!~"/health.*|/metrics.*|/ready.*|/live.*"
        }}[1m]
      )
    )
    """
    return max(0.0, prometheus_query(query))


def get_service_cpu(service: str) -> float:
    query = f"""
    sum(
      rate(
        container_cpu_usage_seconds_total{{
          namespace="default",
          pod=~"{service}.*",
          container!="",
          container!="POD"
        }}[1m]
      )
    )
    """
    return max(0.0, prometheus_query(query))


def get_service_memory_bytes(service: str) -> float:
    query = f"""
    sum(
      container_memory_working_set_bytes{{
        namespace="default",
        pod=~"{service}.*",
        container!="",
        container!="POD"
      }}
    )
    """
    return max(0.0, prometheus_query(query))


def get_service_latency_p95(service: str) -> float:
    query = f"""
    histogram_quantile(
      0.95,
      sum(
        rate(
          istio_request_duration_milliseconds_bucket{{
            destination_app="{service}"
          }}[1m]
        )
      ) by (le)
    )
    """
    return max(0.0, prometheus_query(query))


def get_service_latency_avg(service: str) -> float:
    query = f"""
    (
      sum(
        rate(
          istio_request_duration_milliseconds_sum{{
            destination_app="{service}"
          }}[1m]
        )
      )
      /
      sum(
        rate(
          istio_request_duration_milliseconds_count{{
            destination_app="{service}"
          }}[1m]
        )
      )
    )
    """
    return max(0.0, prometheus_query(query))


# =========================================================
# Prediction Helpers
# =========================================================

def smooth_rps(predicted_rps: float) -> float:
    global previous_smoothed_rps

    if previous_smoothed_rps is None:
        previous_smoothed_rps = predicted_rps
        return predicted_rps

    smoothed = (
        SMOOTHING_ALPHA * predicted_rps
        + (1 - SMOOTHING_ALPHA) * previous_smoothed_rps
    )

    previous_smoothed_rps = smoothed
    return smoothed


def clamp_replicas(value: int) -> int:
    return max(MIN_REPLICAS, min(MAX_REPLICAS, value))


def stabilize_replicas(service: str, recommended_replicas: int) -> int:
    now = time.time()
    previous = last_replicas.get(service, MIN_REPLICAS)

    if recommended_replicas > previous:
        last_replicas[service] = recommended_replicas
        last_scale_down_time[service] = now
        return recommended_replicas

    if recommended_replicas == previous:
        return previous

    last_down = last_scale_down_time.get(service, 0)

    if now - last_down >= SCALE_DOWN_COOLDOWN_SECONDS:
        last_replicas[service] = recommended_replicas
        last_scale_down_time[service] = now
        return recommended_replicas

    return previous


def predict_future_rps(model1, rps_scaler) -> float:
    recent_rps = np.array(frontend_rps_history).reshape(-1, 1)
    recent_scaled = rps_scaler.transform(recent_rps)

    x = recent_scaled.reshape(1, WINDOW_SIZE, 1)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model1(x_tensor).cpu().numpy()

    predicted_rps = rps_scaler.inverse_transform(pred_scaled)[0][0]
    predicted_rps = max(0.0, float(predicted_rps))

    return smooth_rps(predicted_rps)


def build_model2_row(
    service: str,
    predicted_rps: float,
    current_frontend_rps: float,
    current_service_rps: float,
    cpu_usage_cores: float,
    memory_usage_bytes: float,
    latency_p95_ms: float,
    latency_avg_ms: float,
    feature_columns: list[str]
) -> pd.DataFrame:

    service_history = list(service_rps_history[service])

    service_rps_lag_1 = service_history[-2] if len(service_history) >= 2 else current_service_rps

    frontend_history = list(frontend_rps_history)

    rps_lag_1 = frontend_history[-2] if len(frontend_history) >= 2 else current_frontend_rps
    rps_lag_2 = frontend_history[-3] if len(frontend_history) >= 3 else current_frontend_rps
    rps_lag_3 = frontend_history[-4] if len(frontend_history) >= 4 else current_frontend_rps

    rps_rolling_mean_3 = (
        sum(frontend_history[-3:]) / min(3, len(frontend_history))
        if frontend_history else current_frontend_rps
    )

    row = {
        "predicted_rps": predicted_rps,
        "frontend_rps": current_frontend_rps,
        "service_rps": current_service_rps,
        "rps_lag_1": rps_lag_1,
        "rps_lag_2": rps_lag_2,
        "rps_lag_3": rps_lag_3,
        "service_rps_lag_1": service_rps_lag_1,
        "rps_rolling_mean_3": rps_rolling_mean_3,
        "cpu_usage_cores": cpu_usage_cores,
        "memory_usage_bytes": memory_usage_bytes,
        "latency_p95_ms": latency_p95_ms,
        "latency_avg_ms": latency_avg_ms,
    }

    for svc in SERVICES:
        row[f"svc_{svc}"] = 1 if svc == service else 0

    df = pd.DataFrame([row])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]


def predict_replicas_for_service(
    model2,
    input_scaler,
    output_scaler,
    feature_df: pd.DataFrame
) -> float:

    x_raw = feature_df.values
    x_scaled = input_scaler.transform(x_raw)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_scaled = model2(x_tensor).cpu().numpy()

    y_real = output_scaler.inverse_transform(y_scaled)[0][0]
    return float(y_real)


# =========================================================
# Load Models
# =========================================================

logging.info("Loading models and scalers...")

rps_scaler = joblib.load(MODEL1_DIR / "rps_scaler.pkl")

model2_input_scaler = joblib.load(MODEL2_DIR / "input_scaler.pkl")
model2_output_scaler = joblib.load(MODEL2_DIR / "output_scaler.pkl")

with open(MODEL2_DIR / "feature_columns.txt", "r") as f:
    feature_columns = [line.strip() for line in f.readlines()]

model1 = RPSLSTMModel()
model1.load_state_dict(torch.load(MODEL1_PATH, map_location="cpu"))
model1.eval()

model2 = CapacityNNV2(input_size=len(feature_columns))
model2.load_state_dict(torch.load(MODEL2_PATH, map_location="cpu"))
model2.eval()

logging.info("Models loaded successfully.")
logging.info(f"Model 2 feature count: {len(feature_columns)}")


# =========================================================
# Main Loop
# =========================================================

def collect_initial_history():
    logging.info(f"Collecting initial RPS history: {WINDOW_SIZE} samples...")

    while len(frontend_rps_history) < WINDOW_SIZE:
        try:
            current_rps = get_frontend_rps()

            if current_rps <= 0:
                logging.info("Skipping zero-RPS sample.")
            else:
                frontend_rps_history.append(current_rps)
                logging.info(
                    f"Buffered RPS sample {len(frontend_rps_history)}/{WINDOW_SIZE}: {current_rps:.2f}"
                )

        except Exception as e:
            logging.warning(f"Failed to collect RPS sample: {e}")

        time.sleep(PREDICTION_INTERVAL_SECONDS)

    logging.info("Warmup complete. Starting prediction loop.")


def run_prediction_loop():
    while True:
        try:
            current_frontend_rps = get_frontend_rps()
            frontend_rps_history.append(current_frontend_rps)

            current_frontend_rps_gauge.set(current_frontend_rps)

            # =====================================================
            # Zero-traffic rule:
            # If there is no real traffic, publish minimum replicas.
            # This prevents old predictions from keeping services scaled up.
            # =====================================================
            if current_frontend_rps <= 0.1:
                logging.info("No active traffic detected. Resetting all services to minimum replicas.")

                predicted_rps_gauge.set(0)

                decisions = []

                for service in SERVICES:
                    last_replicas[service] = MIN_REPLICAS
                    last_scale_down_time[service] = time.time()

                    model2_raw_prediction_gauge.labels(service=service).set(MIN_REPLICAS)

                    if service not in EXCLUDE_FROM_SCALING:
                        predicted_replicas_gauge.labels(service=service).set(MIN_REPLICAS)

                    decisions.append({
                        "service": service,
                        "raw_prediction": MIN_REPLICAS,
                        "predicted_replicas": MIN_REPLICAS,
                        "service_rps": 0,
                        "cpu": 0,
                        "memory_mb": 0,
                        "latency_p95_ms": 0,
                        "latency_avg_ms": 0,
                    })

                logging.info("=" * 80)
                logging.info(f"Current frontend RPS: {current_frontend_rps:.2f}")
                logging.info("Predicted frontend RPS (+30s): 0.00")
                logging.info("\n" + pd.DataFrame(decisions).to_string(index=False))

                time.sleep(PREDICTION_INTERVAL_SECONDS)
                continue

            if current_frontend_rps > TRAINING_MAX_RPS:
                logging.warning(
                    f"Current RPS {current_frontend_rps:.2f} exceeds training max "
                    f"{TRAINING_MAX_RPS:.2f}; prediction may be unreliable."
                )

            predicted_rps = predict_future_rps(model1, rps_scaler)
            predicted_rps_gauge.set(predicted_rps)

            decisions = []

            for service in SERVICES:
                service_rps = get_service_rps(service)
                service_rps_history[service].append(service_rps)

                cpu = get_service_cpu(service)
                memory_bytes = get_service_memory_bytes(service)
                latency_p95 = get_service_latency_p95(service)
                latency_avg = get_service_latency_avg(service)

                feature_df = build_model2_row(
                    service=service,
                    predicted_rps=predicted_rps,
                    current_frontend_rps=current_frontend_rps,
                    current_service_rps=service_rps,
                    cpu_usage_cores=cpu,
                    memory_usage_bytes=memory_bytes,
                    latency_p95_ms=latency_p95,
                    latency_avg_ms=latency_avg,
                    feature_columns=feature_columns
                )

                raw_prediction = predict_replicas_for_service(
                    model2=model2,
                    input_scaler=model2_input_scaler,
                    output_scaler=model2_output_scaler,
                    feature_df=feature_df
                )

                rounded_replicas = int(round(raw_prediction))
                rounded_replicas = clamp_replicas(rounded_replicas)
                stable_replicas = stabilize_replicas(service, rounded_replicas)

                model2_raw_prediction_gauge.labels(service=service).set(raw_prediction)

                if service not in EXCLUDE_FROM_SCALING:
                    predicted_replicas_gauge.labels(service=service).set(stable_replicas)

                decisions.append({
                    "service": service,
                    "raw_prediction": round(raw_prediction, 3),
                    "predicted_replicas": stable_replicas,
                    "service_rps": round(service_rps, 2),
                    "cpu": round(cpu, 3),
                    "memory_mb": round(memory_bytes / (1024 * 1024), 1),
                    "latency_p95_ms": round(latency_p95, 2),
                    "latency_avg_ms": round(latency_avg, 2),
                })

            logging.info("=" * 80)
            logging.info(f"Current frontend RPS: {current_frontend_rps:.2f}")
            logging.info(f"Predicted frontend RPS (+30s): {predicted_rps:.2f}")
            logging.info("\n" + pd.DataFrame(decisions).to_string(index=False))

        except Exception as e:
            logging.exception(f"Prediction loop error: {e}")

        time.sleep(PREDICTION_INTERVAL_SECONDS)


if __name__ == "__main__":
    start_http_server(EXPORTER_PORT)
    logging.info(f"Predicted metrics exporter running on port {EXPORTER_PORT}")

    collect_initial_history()
    run_prediction_loop()
import time
import math
import requests
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from prometheus_client import start_http_server, Gauge


PROJECT_ROOT = Path(__file__).resolve().parent
#PROJECT_ROOT = Path(__file__).resolve().parents[1]


MODEL1_DIR = PROJECT_ROOT / "data/processed/two_model_dataset/model1_rps_forecast"
MODEL2_DIR = PROJECT_ROOT / "data/processed/two_model_dataset/model2_capacity_per_service"

MODEL1_PATH = PROJECT_ROOT / "models/model1_rps_lstm.pth"
MODEL2_PATH = PROJECT_ROOT / "models/model2_capacity_per_service_nn.pth"

PROMETHEUS_URL = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"
#PROMETHEUS_URL = "http://localhost:9090"

WINDOW_SIZE = 12
QUERY_MINUTES = 5
QUERY_STEP = "10s"

CPU_PER_POD = 0.5
MEMORY_PER_POD_MB = 512

SAFETY_FACTOR = 1.2
MIN_REPLICAS = 1
MAX_REPLICAS = 10

PREDICTION_INTERVAL_SECONDS = 10

# Smooth prediction to avoid sudden jumps
SMOOTHING_ALPHA = 0.3

# Prevent fast scale down
SCALE_DOWN_COOLDOWN_SECONDS = 60

EXCLUDE_FROM_SCALING = [
    "redis-cart"
]

previous_smoothed_rps = None
last_replicas = {}
last_scale_down_time = {}


predicted_rps_gauge = Gauge(
    "predicted_requests_per_second",
    "Predicted future RPS from Model 1"
)

predicted_service_cpu_gauge = Gauge(
    "predicted_service_cpu",
    "Predicted CPU cores required per service",
    ["service"]
)

predicted_service_memory_gauge = Gauge(
    "predicted_service_memory_mb",
    "Predicted memory MB required per service",
    ["service"]
)

predicted_replicas_gauge = Gauge(
    "predicted_replicas",
    "Recommended predicted replicas per service",
    ["service"]
)


class RPSLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class CapacityPerServiceNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)


def clamp_replicas(value):
    return max(MIN_REPLICAS, min(MAX_REPLICAS, value))


def get_service_name(target_column):
    if target_column.endswith("_cpu"):
        return target_column.replace("_cpu", "")
    if target_column.endswith("_memory"):
        return target_column.replace("_memory", "")
    return target_column


def get_recent_rps_from_prometheus():
    end = time.time()
    start = end - (QUERY_MINUTES * 60)

    # Business traffic only for frontend.
    # Health checks and metrics traffic are excluded.
    query = '''
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
    '''

    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={
            "query": query,
            "start": start,
            "end": end,
            "step": QUERY_STEP
        },
        timeout=10
    )

    response.raise_for_status()
    data = response.json()

    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {data}")

    result = data["data"]["result"]

    if not result:
        raise RuntimeError("No RPS data returned from Prometheus")

    values = result[0]["values"]

    rps_series = []
    for _, value in values:
        if value not in ["NaN", "nan", "+Inf", "-Inf"]:
            rps_series.append(float(value))

    if len(rps_series) < WINDOW_SIZE:
        raise RuntimeError(
            f"Not enough RPS points. Found {len(rps_series)}, need {WINDOW_SIZE}"
        )

    return rps_series[-WINDOW_SIZE:]


def smooth_rps(predicted_rps):
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


def stabilize_replicas(service, recommended_replicas):
    now = time.time()

    previous = last_replicas.get(service, MIN_REPLICAS)

    # Scale up immediately
    if recommended_replicas > previous:
        last_replicas[service] = recommended_replicas
        last_scale_down_time[service] = now
        return recommended_replicas

    # Keep same replicas
    if recommended_replicas == previous:
        return previous

    # Scale down only after cooldown
    last_down = last_scale_down_time.get(service, 0)

    if now - last_down >= SCALE_DOWN_COOLDOWN_SECONDS:
        last_replicas[service] = recommended_replicas
        last_scale_down_time[service] = now
        return recommended_replicas

    return previous


print("Loading models and scalers...")

rps_scaler = joblib.load(MODEL1_DIR / "rps_scaler.pkl")

capacity_input_scaler = joblib.load(MODEL2_DIR / "capacity_input_scaler.pkl")
capacity_output_scaler = joblib.load(MODEL2_DIR / "capacity_output_scaler.pkl")

with open(MODEL2_DIR / "target_columns.txt", "r") as f:
    target_columns = [line.strip() for line in f.readlines()]

model1 = RPSLSTMModel()
model1.load_state_dict(torch.load(MODEL1_PATH, map_location="cpu"))
model1.eval()

model2 = CapacityPerServiceNN(len(target_columns))
model2.load_state_dict(torch.load(MODEL2_PATH, map_location="cpu"))
model2.eval()

print("Models loaded successfully")


def get_prediction():
    rps_series = get_recent_rps_from_prometheus()

    recent_rps = np.array(rps_series).reshape(-1, 1)
    recent_scaled = rps_scaler.transform(recent_rps)

    x_rps = recent_scaled.reshape(1, WINDOW_SIZE, 1)
    x_rps_tensor = torch.tensor(x_rps, dtype=torch.float32)

    with torch.no_grad():
        predicted_rps_scaled = model1(x_rps_tensor).cpu().numpy()

    predicted_rps = rps_scaler.inverse_transform(predicted_rps_scaled)[0][0]
    predicted_rps = max(0.0, float(predicted_rps))

    predicted_rps = smooth_rps(predicted_rps)

    model2_input = np.array([
        [predicted_rps, predicted_rps ** 2]
    ])

    model2_input_scaled = capacity_input_scaler.transform(model2_input)
    model2_tensor = torch.tensor(model2_input_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction_scaled = model2(model2_tensor).cpu().numpy()

    prediction_real = capacity_output_scaler.inverse_transform(prediction_scaled)[0]

    services = {}

    for col, value in zip(target_columns, prediction_real):
        service = get_service_name(col)

        if service not in services:
            services[service] = {
                "cpu": 0.0,
                "memory_mb": 0.0,
                "replicas": MIN_REPLICAS
            }

        if col.endswith("_cpu"):
            services[service]["cpu"] = max(0.0, float(value))

        elif col.endswith("_memory"):
            # Keep this if your training memory target was in bytes.
            # If your model already outputs MB, remove the division.
            services[service]["memory_mb"] = max(0.0, float(value)) / (1024 * 1024)

    for service, values in services.items():
        predicted_cpu = values["cpu"]
        predicted_memory_mb = values["memory_mb"]

        required_cpu = predicted_cpu * SAFETY_FACTOR
        required_memory_mb = predicted_memory_mb * SAFETY_FACTOR

        replicas_by_cpu = math.ceil(required_cpu / CPU_PER_POD)
        replicas_by_memory = math.ceil(required_memory_mb / MEMORY_PER_POD_MB)

        recommended_replicas = max(
            replicas_by_cpu,
            replicas_by_memory,
            MIN_REPLICAS
        )

        recommended_replicas = clamp_replicas(recommended_replicas)
        recommended_replicas = stabilize_replicas(service, recommended_replicas)

        values["cpu"] = predicted_cpu
        values["memory_mb"] = predicted_memory_mb
        values["replicas"] = recommended_replicas

    return predicted_rps, services


if __name__ == "__main__":
    start_http_server(8000)
    print("Predicted metrics exporter running on port 8000")

    while True:
        try:
            predicted_rps, services = get_prediction()

            predicted_rps_gauge.set(predicted_rps)

            for service, values in services.items():
                predicted_service_cpu_gauge.labels(service=service).set(values["cpu"])
                predicted_service_memory_gauge.labels(service=service).set(values["memory_mb"])

                if service not in EXCLUDE_FROM_SCALING:
                    predicted_replicas_gauge.labels(service=service).set(values["replicas"])

            print(f"Predicted RPS={predicted_rps:.2f} | Services={services}")

        except Exception as e:
            print(f"Prediction error: {e}")

        time.sleep(PREDICTION_INTERVAL_SECONDS)
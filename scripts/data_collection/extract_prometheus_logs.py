import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta
from pathlib import Path

PROMETHEUS_URL = "http://localhost:9090"

parser = argparse.ArgumentParser()

parser.add_argument("--start", help="Start time (YYYY-MM-DD HH:MM:SS)")
parser.add_argument("--end", help="End time (YYYY-MM-DD HH:MM:SS)")
parser.add_argument("--run-name", default="run_1", help="Folder name for this experiment")
parser.add_argument("--step", default="5s", help="Prometheus query step, example: 5s, 15s, 30s")
parser.add_argument(
  "--rate-window",
  default="1m",
  help="Prometheus range selector window for rate/increase queries, example: 30s, 1m"
)

args = parser.parse_args()

if args.start and args.end:
    START_TIME = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    END_TIME = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
else:
    END_TIME = datetime.now()
    START_TIME = END_TIME - timedelta(minutes=30)

STEP = args.step
RATE_WINDOW = args.rate_window

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "prometheus_export" / args.run_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Using time range:")
print(f"Start : {START_TIME}")
print(f"End   : {END_TIME}")
print(f"Step  : {STEP}")
print(f"Rate window: {RATE_WINDOW}")
print(f"Output: {OUTPUT_DIR}")

COMMON_ISTIO_FILTER = """
reporter="destination",
destination_workload_namespace="default",
destination_workload!~"unknown|online-boutique-test.*|istio-gateway-istio"
"""

QUERIES = {
    "frontend_total_requests": f"""
sum(
  increase(
    istio_requests_total{{
      reporter="source",
      source_workload="istio-gateway-istio",
      destination_workload="frontend"
    }}[{RATE_WINDOW}]
  )
)
""",

    "frontend_rps_total": f"""
sum(
  rate(
    istio_requests_total{{
      reporter="source",
      source_workload="istio-gateway-istio",
      destination_workload="frontend"
    }}[{RATE_WINDOW}]
  )
)
""",

    "service_rps": f"""
sum by (destination_workload) (
  rate(
    istio_requests_total{{
      {COMMON_ISTIO_FILTER}
    }}[{RATE_WINDOW}]
  )
)
""",

    "service_latency_p95_ms": f"""
histogram_quantile(
  0.95,
  sum by (destination_workload, le) (
    rate(
      istio_request_duration_milliseconds_bucket{{
        {COMMON_ISTIO_FILTER}
      }}[{RATE_WINDOW}]
    )
  )
)
""",

    "service_latency_avg_ms": f"""
sum by (destination_workload) (
  rate(
    istio_request_duration_milliseconds_sum{{
      {COMMON_ISTIO_FILTER}
    }}[{RATE_WINDOW}]
  )
)
/
sum by (destination_workload) (
  rate(
    istio_request_duration_milliseconds_count{{
      {COMMON_ISTIO_FILTER}
    }}[{RATE_WINDOW}]
  )
)
""",

    "cpu_usage_cores": f"""
sum by (pod) (
  rate(
    container_cpu_usage_seconds_total{{
      namespace="default",
      container!="POD",
      container!="istio-proxy",
      pod!~"online-boutique-test.*"
    }}[{RATE_WINDOW}]
  )
)
""",

    "memory_usage_bytes": """
sum by (pod) (
  container_memory_working_set_bytes{
    namespace="default",
    container!="POD",
    container!="istio-proxy",
    pod!~"online-boutique-test.*"
  }
)
"""
}


def query_range(metric_name, query):
    url = f"{PROMETHEUS_URL}/api/v1/query_range"

    params = {
        "query": query,
        "start": START_TIME.timestamp(),
        "end": END_TIME.timestamp(),
        "step": STEP
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed for {metric_name}: {payload}")

    rows = []

    for series in payload["data"]["result"]:
        labels = series.get("metric", {})

        for timestamp, value in series.get("values", []):
            try:
                numeric_value = float(value)
            except ValueError:
                continue

            rows.append({
                "timestamp": datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S"),
                "metric": metric_name,
                "value": numeric_value,
                **labels
            })

    return pd.DataFrame(rows)


all_data = []

for metric_name, query in QUERIES.items():
    print(f"Extracting {metric_name}...")

    df = query_range(metric_name, query)

    output_file = OUTPUT_DIR / f"{metric_name}.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file} rows={len(df)}")

    if not df.empty:
        all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / "combined_metrics.csv", index=False)
else:
    print("Warning: no data returned from Prometheus.")

print("Done.")
print(f"Files saved in: {OUTPUT_DIR}")
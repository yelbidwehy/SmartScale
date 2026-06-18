from locust import HttpUser, task, constant,between
import json
import time
import os
import random
from collections import defaultdict

CHECKOUT_DATA = {
    "email": "test@example.com",
    "street_address": "123 Test Street",
    "zip_code": "12345",
    "city": "Riyadh",
    "state": "Riyadh",
    "country": "Saudi Arabia",
    "credit_card_number": "4111111111111111",
    "credit_card_expiration_month": "12",
    "credit_card_expiration_year": "2027",
    "credit_card_cvv": "123"
}

with open("/mnt/workload/boutique_workload_large.json") as f:
    WORKLOAD = sorted(json.load(f), key=lambda x: x["time"])

USER_WORKLOAD = defaultdict(list)
for req in WORKLOAD:
    USER_WORKLOAD[req.get("user")].append(req)

AVAILABLE_USERS = sorted(USER_WORKLOAD.keys())

# Each worker pod gets a different seed so the 4 workers do not all replay
# the exact same workload-user IDs in the same order.
WORKER_NAME = os.environ.get("HOSTNAME", "locust-worker")
WORKER_SEED = abs(hash(WORKER_NAME)) % 1000000
random.seed(WORKER_SEED)


class BoutiqueUser(HttpUser):
    wait_time = between(0.001, 0.01)  # 1-10ms between task checks

    def on_start(self):
        self.start_time = time.time()

        if not hasattr(self.environment, "user_counter"):
            self.environment.user_counter = 0

        self.user_id = self.environment.user_counter
        self.environment.user_counter += 1

        if not AVAILABLE_USERS:
            self.my_workload = []
            self.no_workload = True
            print("No workload users found in boutique_workload_large.json")
            return

        # Random mapping is better with multiple workers because each worker has
        # its own local user_counter. This avoids all workers only using user IDs 0..N.
        self.mapped_user = random.choice(AVAILABLE_USERS)
        self.my_workload = USER_WORKLOAD.get(self.mapped_user, [])

        self.index = 0
        self.no_workload = len(self.my_workload) == 0

        print(
            f"Worker {WORKER_NAME} | Locust user {self.user_id} "
            f"mapped to workload user {self.mapped_user}, "
            f"loaded {len(self.my_workload)} requests"
        )

    @task
    def replay_workload(self):
        if self.no_workload:
            return

        now = time.time() - self.start_time

        while self.index < len(self.my_workload):
            req = self.my_workload[self.index]

            if req["time"] > now:
                break

            method = req.get("method", "GET").upper()
            url = req.get("url", "/")
            data = req.get("data", {}) or {}

            if "checkout" in url and url.endswith("/"):
                url = url[:-1]

            if method == "POST" and "checkout" in url:
                data = CHECKOUT_DATA

            try:
                if method == "GET":
                    self.client.get(url, name=url)
                elif method == "POST":
                    self.client.post(url, data=data, name=url)
            except Exception as e:
                print(f"Request failed: {method} {url} - {e}")

            self.index += 1

        # Repeat the same workload forever. This is important for long experiments
        # where stage duration is longer than the original captured workload.
        if self.index >= len(self.my_workload):
            self.index = 0
            self.start_time = time.time()

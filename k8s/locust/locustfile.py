from locust import HttpUser, task, constant
from locust.exception import StopUser
import json
import time
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


class BoutiqueUser(HttpUser):
    wait_time = constant(0)

    def on_start(self):
        self.start_time = time.time()

        if not hasattr(self.environment, "user_counter"):
            self.environment.user_counter = 0

        self.user_id = self.environment.user_counter
        self.environment.user_counter += 1

        self.my_workload = USER_WORKLOAD.get(self.user_id, [])

        self.index = 0
        self.no_workload = len(self.my_workload) == 0

        print(f"User {self.user_id} loaded {len(self.my_workload)} requests")

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

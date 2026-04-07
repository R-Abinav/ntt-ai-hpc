import json

import requests

BASE = "http://127.0.0.1:5001"


def section(title):
    print(f"\n{'=' * 54}")
    print(f"  {title}")
    print("=" * 54)


# 1. GET /health
section("GET /health")
r = requests.get(f"{BASE}/health")
print(f"Status : {r.status_code}")
print(json.dumps(r.json(), indent=2))

# 2. POST /predict  n=1048576
section("POST /predict  n=1048576")
r = requests.post(f"{BASE}/predict", json={"n": 1048576})
print(f"Status : {r.status_code}")
body = r.json()
print(f"Recommended threads : {body['recommended_threads']}")
print(f"Predicted time      : {body['predicted_time_us']} us")
print("All predictions:")
print(f"  {'threads':>8}  {'predicted_time_us':>20}")
print(f"  {'-' * 8}  {'-' * 20}")
for entry in body["all_predictions"]:
    print(f"  {entry['threads']:>8}  {entry['predicted_time_us']:>20.2f}")

# 3. GET /optimal_map
section("GET /optimal_map")
r = requests.get(f"{BASE}/optimal_map")
print(f"Status : {r.status_code}")
table = r.json()
print(f"  {'n':>10}  {'recommended_threads':>20}")
print(f"  {'-' * 10}  {'-' * 20}")
for row in table:
    print(f"  {row['n']:>10}  {row['recommended_threads']:>20}")

# 4. POST /predict  n=999  (invalid — not a power of 2)
section("POST /predict  n=999  (error case)")
r = requests.post(f"{BASE}/predict", json={"n": 999})
print(f"Status : {r.status_code}")
print(json.dumps(r.json(), indent=2))

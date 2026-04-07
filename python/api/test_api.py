import json
import requests

BASE = "http://127.0.0.1:5001"

def section(title):
    print(f"\n{'=' * 54}\n  {title}\n{'=' * 54}")

# 1. GET /health
section("GET /health")
r = requests.get(f"{BASE}/health")
print(f"Status : {r.status_code}")
print(json.dumps(r.json(), indent=2))

# 2. POST /predict — MLP (default)
section("POST /predict  n=1048576  (mlp)")
r = requests.post(f"{BASE}/predict", json={"n": 1048576})
body = r.json()
print(f"Model used          : {body['model_used']}")
print(f"Recommended threads : {body['recommended_threads']}")
print(f"Predicted time      : {body['predicted_time_us']} us")

# 3. Switch active model to XGBoost
section("POST /active_model  → xgboost")
r = requests.post(f"{BASE}/active_model", json={"model": "xgboost"})
print(f"Status : {r.status_code}")
print(json.dumps(r.json(), indent=2))

# 4. POST /predict — now uses XGBoost by default
section("POST /predict  n=1048576  (xgboost via active model)")
r = requests.post(f"{BASE}/predict", json={"n": 1048576})
body = r.json()
print(f"Model used          : {body['model_used']}")
print(f"Recommended threads : {body['recommended_threads']}")
print(f"Predicted time      : {body['predicted_time_us']} us")

# 5. POST /predict — override back to MLP via query param
section("POST /predict  n=262144  ?model=mlp  (query override)")
r = requests.post(f"{BASE}/predict?model=mlp", json={"n": 262144})
body = r.json()
print(f"Model used          : {body['model_used']}")
print(f"Recommended threads : {body['recommended_threads']}")

# 6. GET /optimal_map — XGBoost
section("GET /optimal_map?model=xgboost")
r = requests.get(f"{BASE}/optimal_map?model=xgboost")
table = r.json()
print(f"  {'n':>10}  {'recommended_threads':>20}")
print(f"  {'-'*10}  {'-'*20}")
for row in table:
    print(f"  {row['n']:>10}  {row['recommended_threads']:>20}")

# 7. GET /compare
section("GET /compare?n=262144")
r = requests.get(f"{BASE}/compare?n=262144")
print(f"Status : {r.status_code}")
print(json.dumps(r.json(), indent=2))

# 8. Error case
section("POST /predict  n=999  (error case)")
r = requests.post(f"{BASE}/predict", json={"n": 999})
print(f"Status : {r.status_code}")
print(json.dumps(r.json(), indent=2))
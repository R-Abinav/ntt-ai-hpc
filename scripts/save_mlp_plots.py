"""
save_plots.py — regenerate all four notebook plots and save to docs/

Run from the project root:
    python scripts/save_plots.py
"""

import math
import os
import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "result.csv")
MODEL_PT = os.path.join(ROOT, "python", "api", "cost_model.pt")
SCALER = os.path.join(ROOT, "python", "api", "scaler.pkl")
OUT_DIR = os.path.join(ROOT, "docs")

os.makedirs(OUT_DIR, exist_ok=True)

# ── model definition (must match training) ────────────────────────────────────


class CostModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# ── load data ─────────────────────────────────────────────────────────────────

print(f"Loading data from {DATA}")
df = pd.read_csv(DATA)
print(f"  {len(df)} rows, sizes: {sorted(df['n'].unique())}")

# ── plot 1 — speedup curve ────────────────────────────────────────────────────

print("Generating plot 1: Speedup Curve ...")

fig, ax = plt.subplots(figsize=(9, 5))

for n in [8192, 65536, 262144, 1048576]:
    subset = df[df["n"] == n].sort_values("threads")
    if subset.empty:
        print(f"  WARNING: n={n} not found in data, skipping")
        continue
    t1_rows = subset[subset["threads"] == 1]
    if t1_rows.empty:
        print(f"  WARNING: no 1-thread row for n={n}, skipping")
        continue
    t1 = t1_rows["time_us"].values[0]
    ax.plot(subset["threads"], t1 / subset["time_us"], marker="o", label=f"n={n:,}")

ax.axhline(y=1, linestyle="--", color="gray", linewidth=1, label="baseline (1x)")
ax.set_xlabel("Thread Count")
ax.set_ylabel("Speedup (x)")
ax.set_title("OpenMP Speedup vs Thread Count")
ax.legend()
ax.grid(True)
plt.tight_layout()
out = os.path.join(OUT_DIR, "plot1_speedup.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved: {out}")

# ── plot 2 — threading overhead crossover ─────────────────────────────────────

print("Generating plot 2: Threading Overhead Crossover ...")

fig, ax = plt.subplots(figsize=(9, 5))

for t, label, color in [(1, "1 thread", "steelblue"), (4, "4 threads", "darkorange")]:
    sub = df[df["threads"] == t].sort_values("n")
    ax.plot(sub["n"], sub["time_us"], marker="o", label=label, color=color)

ax.set_xscale("log", base=2)
ax.set_xlabel("NTT Size n (log2 scale)")
ax.set_ylabel("Time (us)")
ax.set_title("Threading Overhead Crossover")

t1s = df[df["threads"] == 1].set_index("n")["time_us"]
t4s = df[df["threads"] == 4].set_index("n")["time_us"]
common = t1s.index.intersection(t4s.index)
diff = t4s[common] - t1s[common]
crossover_ns = diff[diff <= 0].index
if len(crossover_ns):
    cx = crossover_ns[0]
    ax.axvline(
        x=cx, linestyle=":", color="red", linewidth=1.5, label=f"crossover n={cx:,}"
    )

ax.legend()
ax.grid(True)
plt.tight_layout()
out = os.path.join(OUT_DIR, "plot2_crossover.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved: {out}")

# ── feature engineering (shared setup for plots 3 and 4) ─────────────────────

print("Preparing features and training model ...")

df["log2_n"] = np.log2(df["n"])
n_vals = df["n"].values

X = df[["log2_n", "threads"]].values.astype(np.float32)
y = np.log1p(df["time_us"].values).astype(np.float32)

scaler_local = StandardScaler()
X_scaled = scaler_local.fit_transform(X)

X_train, X_test, y_train, y_test, n_train, n_test = train_test_split(
    X_scaled, y, n_vals, test_size=0.2, random_state=42
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

opt_map = (
    df.loc[df.groupby("n")["time_us"].idxmin()].set_index("n")["threads"].to_dict()
)

# try loading saved model/scaler first; fall back to retraining if missing
model = CostModel()
if os.path.exists(MODEL_PT) and os.path.exists(SCALER):
    print(f"  Loading saved model from {MODEL_PT}")
    model.load_state_dict(torch.load(MODEL_PT, map_location="cpu", weights_only=True))
    with open(SCALER, "rb") as f:
        scaler_infer = pickle.load(f)
    train_losses, test_losses = None, None
else:
    print("  Saved model not found — retraining for 500 epochs ...")
    scaler_infer = scaler_local
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses, test_losses = [], []
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            tl = criterion(model(X_test_t), y_test_t).item()
        train_losses.append(loss.item())
        test_losses.append(tl)
        if epoch % 100 == 0:
            print(f"    epoch {epoch:4d}  train={loss.item():.4f}  test={tl:.4f}")

model.eval()

# ── plot 3 — training loss curve ─────────────────────────────────────────────

# always retrain a fresh instance so we have the loss history for the plot
print("Retraining to capture loss curve for plot 3 ...")

model_fresh = CostModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_fresh.parameters(), lr=0.001)
train_losses, test_losses = [], []

for epoch in range(1, 501):
    model_fresh.train()
    optimizer.zero_grad()
    loss = criterion(model_fresh(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()
    model_fresh.eval()
    with torch.no_grad():
        tl = criterion(model_fresh(X_test_t), y_test_t).item()
    train_losses.append(loss.item())
    test_losses.append(tl)

print("Generating plot 3: Training Loss Curve ...")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_losses, label="Train Loss")
ax.plot(test_losses, label="Test Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training Loss Curve")
ax.legend()
ax.grid(True)
plt.tight_layout()
out = os.path.join(OUT_DIR, "plot3_loss_curve.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved: {out}")

# ── plot 4 — predicted vs actual optimal thread count ────────────────────────

print("Generating plot 4: AI-Predicted vs Actual Optimal Thread Count ...")

CANDIDATE_THREADS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]
test_ns = sorted(set(n_test))

predicted_opts, actual_opts = [], []

with torch.no_grad():
    for n_val in test_ns:
        rows = np.array(
            [[np.log2(n_val), t] for t in CANDIDATE_THREADS], dtype=np.float32
        )
        preds = model(torch.tensor(scaler_infer.transform(rows))).squeeze().numpy()
        predicted_opts.append(CANDIDATE_THREADS[int(np.argmin(preds))])
        actual_opts.append(opt_map[n_val])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(test_ns, actual_opts, marker="o", linestyle="-", label="Actual optimal threads")
ax.plot(
    test_ns,
    predicted_opts,
    marker="x",
    linestyle="--",
    label="Predicted optimal threads",
)
ax.set_xscale("log", base=2)
ax.set_xlabel("NTT Size n (log2 scale)")
ax.set_ylabel("Optimal Thread Count")
ax.set_title("AI-Predicted vs Actual Optimal Thread Count")
ax.legend()
ax.grid(True)
plt.tight_layout()
out = os.path.join(OUT_DIR, "plot4_predicted_vs_actual.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved: {out}")

correct = sum(p == a for p, a in zip(predicted_opts, actual_opts))
print(f"  Accuracy: {correct}/{len(test_ns)} = {100 * correct / len(test_ns):.1f}%")

#done
print("\nAll plots saved to docs/")

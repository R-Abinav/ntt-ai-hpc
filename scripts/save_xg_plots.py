"""
save_plots_xgb.py — save XGBoost plots to docs/
Run from project root:
    python scripts/save_plots_xgb.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA      = os.path.join(ROOT, "results", "result.csv")
XGB_MODEL = os.path.join(ROOT, "python", "api", "cost_model_xgb.pkl")
SCALER    = os.path.join(ROOT, "python", "api", "scaler.pkl")
OUT_DIR   = os.path.join(ROOT, "docs")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA)
opt_map = df.loc[df.groupby("n")["time_us"].idxmin()].set_index("n")["threads"].to_dict()

with open(XGB_MODEL, "rb") as f:
    xgb_model = pickle.load(f)
with open(SCALER, "rb") as f:
    scaler = pickle.load(f)

CANDIDATE_THREADS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]
all_ns = sorted(df['n'].unique())

# ── plot 5 — feature importance ───────────────────────────────────────────────

print("Generating plot 5: XGBoost Feature Importance ...")
fig, ax = plt.subplots(figsize=(9, 5))
importance = xgb_model.feature_importances_
bars = ax.bar(['log2(n)', 'threads'], importance, color=['steelblue', 'orange'], width=0.4)
ax.bar_label(bars, fmt='%.3f', padding=4)
ax.set_title('XGBoost Feature Importance')
ax.set_ylabel('Importance Score')
ax.set_xlabel('Feature')
ax.set_ylim(0, max(importance) * 1.2)
ax.grid(axis='y')
plt.tight_layout()
out = os.path.join(OUT_DIR, "plot5_xgb_feature_importance.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved: {out}")

# ── plot 6 — predicted vs actual ──────────────────────────────────────────────

print("Generating plot 6: XGBoost Predicted vs Actual Optimal Thread Count ...")
xgb_opts, actual_opts = [], []
for n_val in all_ns:
    rows = np.array([[np.log2(n_val), t] for t in CANDIDATE_THREADS], dtype=np.float32)
    preds = xgb_model.predict(scaler.transform(rows))
    xgb_opts.append(CANDIDATE_THREADS[int(np.argmin(preds))])
    actual_opts.append(opt_map[n_val])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(all_ns, actual_opts, 'o-', label='Actual optimal threads', color='steelblue')
ax.plot(all_ns, xgb_opts, 'x--', label='XGB predicted optimal threads', color='green')
ax.set_xscale('log', base=2)
ax.set_xticks(all_ns)
ax.set_xticklabels([f'$2^{{{int(np.log2(n))}}}$' for n in all_ns], fontsize=8)
ax.set_title('XGBoost: Predicted vs Actual Optimal Thread Count')
ax.set_xlabel('NTT Size n (log2 scale)')
ax.set_ylabel('Optimal Thread Count')
ax.legend()
ax.grid(True)
correct = sum(p == a for p, a in zip(xgb_opts, actual_opts))
ax.annotate(f'Accuracy: {correct}/{len(all_ns)} = {100*correct/len(all_ns):.1f}%',
            xy=(0.02, 0.95), xycoords='axes fraction', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
plt.tight_layout()
out = os.path.join(OUT_DIR, "plot6_xgb_predicted_vs_actual.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved: {out}")

print("\nXGBoost plots saved to docs/")
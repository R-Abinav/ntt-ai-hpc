"""
add_xgb_cells.py — append three XGBoost cells to ntt_ai_book.ipynb
Run from the project root:
    python scripts/add_xgb_cells.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, "python", "ntt_ai_book.ipynb")

# ── new cells ─────────────────────────────────────────────────────────────────

CELL_M10 = {
    "cell_type": "markdown",
    "id": "m10",
    "metadata": {},
    "source": [
        "## 10 — XGBoost Cost Model\n",
        "Train an XGBRegressor on the same features, target, and train/test split as the MLP. "
        "XGBoost is a gradient-boosted tree ensemble — it makes no assumptions about the shape "
        "of the relationship between inputs and output, and often reaches lower error than a small "
        "MLP on tabular data with few features. Using the same split lets us compare test MSE directly.\n\n"
        "On macOS, PyTorch's training loop leaves OpenMP threads resident in the kernel process. "
        "XGBoost ships its own libomp and crashes when it tries to initialise a second OpenMP runtime "
        "in the same process. The cell below trains XGBoost in a fresh subprocess (no pre-loaded libomp), "
        "saves the model to disk, then loads it back — giving a clean result without touching any existing cells.",
    ],
}

CELL_C10 = {
    "cell_type": "code",
    "execution_count": None,
    "id": "c10",
    "metadata": {},
    "outputs": [],
    "source": [
        "import os, subprocess, sys, tempfile, pickle\n",
        "import numpy as np\n",
        "from sklearn.metrics import mean_squared_error\n",
        "\n",
        "# ── train XGBoost in a subprocess to avoid libomp conflict with PyTorch ──────\n",
        "# After PyTorch's training loop, its OpenMP thread pool is resident in this\n",
        "# process. XGBoost bundles its own libomp; importing it here causes a second\n",
        "# OpenMP runtime to initialise in the same process, which segfaults on macOS.\n",
        "# Running in a subprocess gives XGBoost a clean process with no pre-loaded OMP.\n",
        "\n",
        "# serialise the split arrays to a temp directory\n",
        "_tmp = tempfile.mkdtemp()\n",
        "_data_path  = os.path.join(_tmp, 'split.npz')\n",
        "_model_path = os.path.join(_tmp, 'xgb.pkl')\n",
        "np.savez(_data_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)\n",
        "\n",
        "_script = f'''\n",
        "import numpy as np, pickle\n",
        "from xgboost import XGBRegressor\n",
        'd = np.load(r"{_data_path}")\n',
        "xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,\n",
        "                   subsample=0.8, random_state=42, verbosity=0, nthread=1)\n",
        'xgb.fit(d["X_train"], d["y_train"])\n',
        'with open(r"{_model_path}", "wb") as f:\n',
        "    pickle.dump(xgb, f)\n",
        'preds = xgb.predict(d["X_test"])\n',
        'resid = d["y_test"] - preds\n',
        "mse   = float((resid ** 2).mean())\n",
        "print(mse)\n",
        "'''\n",
        "\n",
        "result = subprocess.run(\n",
        "    [sys.executable, '-c', _script],\n",
        "    capture_output=True, text=True\n",
        ")\n",
        "if result.returncode != 0:\n",
        "    raise RuntimeError(f'XGBoost subprocess failed:\\n{result.stderr}')\n",
        "\n",
        "with open(_model_path, 'rb') as f:\n",
        "    xgb_model = pickle.load(f)\n",
        "\n",
        "xgb_mse = float(result.stdout.strip())\n",
        "\n",
        "# MLP test MSE for comparison (safe — already in this process)\n",
        "model.eval()\n",
        "with torch.no_grad():\n",
        "    mlp_test_preds = model(X_test_t).squeeze().numpy()\n",
        "mlp_mse = mean_squared_error(y_test, mlp_test_preds)\n",
        "\n",
        "print(f'XGBoost  Test MSE: {xgb_mse:.4f}')\n",
        "print(f'MLP      Test MSE: {mlp_mse:.4f}')",
    ],
}

CELL_M11 = {
    "cell_type": "markdown",
    "id": "m11",
    "metadata": {},
    "source": [
        "## 11 — XGBoost vs MLP: Predicted Optimal Thread Comparison\n",
        "For every NTT size in the dataset, both models score all 13 candidate thread counts "
        "and pick the one with the lowest predicted log1p(time_us). The result is compared "
        "against the ground-truth optimum from the benchmark. A `*` marks a correct prediction. "
        "Accuracy is printed side by side so the two models can be compared directly.",
    ],
}

CELL_C11 = {
    "cell_type": "code",
    "execution_count": None,
    "id": "c11",
    "metadata": {},
    "outputs": [],
    "source": [
        "all_ns          = sorted(df['n'].unique())\n",
        "all_threads_list = sorted(df['threads'].unique())\n",
        "\n",
        "mlp_opts, xgb_opts, actual_opts_all = [], [], []\n",
        "\n",
        "model.eval()\n",
        "with torch.no_grad():\n",
        "    for n_val in all_ns:\n",
        "        rows        = np.array([[np.log2(n_val), t] for t in all_threads_list], dtype=np.float32)\n",
        "        rows_scaled = scaler.transform(rows)\n",
        "\n",
        "        mlp_preds = model(torch.tensor(rows_scaled)).squeeze().numpy()\n",
        "        xgb_preds = xgb_model.predict(rows_scaled)\n",
        "\n",
        "        mlp_opts.append(all_threads_list[int(np.argmin(mlp_preds))])\n",
        "        xgb_opts.append(all_threads_list[int(np.argmin(xgb_preds))])\n",
        "        actual_opts_all.append(opt_map[n_val])\n",
        "\n",
        "print('  {:>10}  {:>8}  {:>8}  {:>8}'.format('n', 'actual', 'mlp', 'xgb'))\n",
        "print('  ' + '-' * 44)\n",
        "for n_val, actual, mlp, xgb in zip(all_ns, actual_opts_all, mlp_opts, xgb_opts):\n",
        "    mlp_mark = '*' if mlp == actual else ' '\n",
        "    xgb_mark = '*' if xgb == actual else ' '\n",
        "    print(f'  {n_val:>10}  {actual:>8}  {mlp:>7}{mlp_mark}  {xgb:>7}{xgb_mark}')\n",
        "\n",
        "n_total  = len(all_ns)\n",
        "mlp_acc  = sum(p == a for p, a in zip(mlp_opts, actual_opts_all))\n",
        "xgb_acc  = sum(p == a for p, a in zip(xgb_opts, actual_opts_all))\n",
        "print(f'\\nMLP accuracy: {mlp_acc}/{n_total} = {100 * mlp_acc / n_total:.1f}%')\n",
        "print(f'XGB accuracy: {xgb_acc}/{n_total} = {100 * xgb_acc / n_total:.1f}%')",
    ],
}

CELL_M12 = {
    "cell_type": "markdown",
    "id": "m12",
    "metadata": {},
    "source": [
        "## 12 — Save XGBoost Model\n",
        "Persist the trained XGBRegressor with pickle so the Flask API can load it alongside "
        "the MLP and serve predictions from either model.",
    ],
}

CELL_C12 = {
    "cell_type": "code",
    "execution_count": None,
    "id": "c12",
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "\n",
        "xgb_path = '../python/api/cost_model_xgb.pkl'\n",
        "with open(xgb_path, 'wb') as f:\n",
        "    pickle.dump(xgb_model, f)\n",
        "\n",
        "size_kb = os.path.getsize(xgb_path) / 1024\n",
        "print(f'Saved: {xgb_path}')\n",
        "print(f'File size: {size_kb:.1f} KB')",
    ],
}

NEW_CELLS = [CELL_M10, CELL_C10, CELL_M11, CELL_C11, CELL_M12, CELL_C12]

# ── load, append, save ────────────────────────────────────────────────────────

with open(NB_PATH, "r", encoding="utf-8") as fh:
    nb = json.load(fh)

existing_ids = {c["id"] for c in nb["cells"]}
added = 0
for cell in NEW_CELLS:
    if cell["id"] in existing_ids:
        print(f"  SKIP  cell {cell['id']} already present")
    else:
        nb["cells"].append(cell)
        added += 1
        print(f"  ADD   cell {cell['id']} ({cell['cell_type']})")

with open(NB_PATH, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")

print(f"\n{added} cell(s) added — notebook now has {len(nb['cells'])} cells")
print(f"Saved: {NB_PATH}")

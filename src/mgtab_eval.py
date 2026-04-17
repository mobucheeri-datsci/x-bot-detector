import os
import sys
import json
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import xgboost as xgb

from src.config import checkpoints, data_raw
from src.train import compute_metrics, find_best_threshold


def main():
    base = os.path.join(data_raw, "mgtab", "MGTAB")
    if not os.path.exists(base):
        raise FileNotFoundError(f"missing {base}; download MGTAB first")

    features = torch.load(os.path.join(base, "features.pt"), weights_only=False).numpy().astype(np.float32)
    labels = torch.load(os.path.join(base, "labels_bot.pt"), weights_only=False).numpy().astype(np.int64)

    n = features.shape[0]
    print(f"loaded mgtab: {n:,} users, {features.shape[1]} features")
    print(f"bot label balance: {(labels == 1).sum():,} bots, {(labels == 0).sum():,} humans")

    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]
    x_test, y_test = features[test_idx], labels[test_idx]

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos
    print(f"train split: {len(train_idx):,} scale_pos_weight={scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=30,
        n_jobs=1,
        tree_method="hist",
    )

    start = time.time()
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    elapsed = time.time() - start
    print(f"trained in {elapsed:.1f}s, best iteration {model.best_iteration}")

    val_prob = model.predict_proba(x_val)[:, 1]
    threshold, _ = find_best_threshold(y_val, val_prob)
    print(f"tuned threshold: {threshold:.3f}")

    y_prob = model.predict_proba(x_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob, threshold=threshold)
    metrics.update({
        "model_name": "xgboost_mgtab",
        "model_type": "xgboost",
        "trained_on": "MGTAB",
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(features.shape[1]),
        "train_time_s": elapsed,
    })

    cm = metrics["confusion_matrix"]
    print(f"\ntest results (threshold={threshold:.3f}):")
    print(f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
    print(f"Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    out = os.path.join(checkpoints, "mgtab_eval_results.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved to {out}")
    print(f"\nfor reference, MGTAB published baselines (Liu et al. 2023, arxiv:2301.01123):")
    print(f"Random Forest: F1 around 0.84")
    print(f"GCN: F1 around 0.86")
    print(f"RGT: F1 around 0.89")

if __name__ == "__main__":
    main()
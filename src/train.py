import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import json
import time
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from src.config import (
    checkpoints, device, glove_file, max_seq_len,
    learning_rate, weight_decay, max_epochs, patience,
)
from src.data import load_processed, create_datasets, GloveVocab
from src.models import BiGRU_LSTM, CNN_BiLSTM


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "threshold": round(float(threshold), 4),
    }
    try: metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    except ValueError: metrics["roc_auc"] = 0.0
    try: metrics["pr_auc"] = round(average_precision_score(y_true, y_prob), 4)
    except ValueError: metrics["pr_auc"] = 0.0
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def collate_glove(batch, vocab):
    texts = [b["text"] for b in batch]
    return {
        "token_ids": vocab.tokenize_batch(texts, max_len=max_seq_len),
        "numeric": torch.stack([b["numeric"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def forward_batch(model, batch, dev):
    labels = batch["label"].to(dev)
    logits = model(input_ids=batch["token_ids"].to(dev), numeric=batch["numeric"].to(dev))
    return logits, labels


def train_neural(model, name, train_loader, val_loader, test_loader, pos_weight,
                 lr=learning_rate, max_epoch=max_epochs, early_stop=patience):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    os.makedirs(checkpoints, exist_ok=True)
    ckpt_path = os.path.join(checkpoints, f"{name}_best.pt")
    best_f1, no_improve, history = 0.0, 0, []

    print(f"\n{'='*60}")
    print(f"  Training: {name}  |  Device: {device}  |  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    start = time.time()
    for epoch in range(1, max_epoch + 1):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, labels = forward_batch(model, batch, device)
            loss = criterion(logits.squeeze(-1), labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits, labels = forward_batch(model, batch, device)
                all_logits.append(logits.squeeze(-1))
                all_labels.append(labels)

        val_probs = torch.sigmoid(torch.cat(all_logits)).cpu().numpy()
        val_labels = torch.cat(all_labels).cpu().numpy()
        val_m = compute_metrics(val_labels, val_probs)
        scheduler.step(val_m["f1"])

        history.append({"epoch": epoch, "train_loss": round(total_loss/n, 4),
                        "val_f1": val_m["f1"], "val_accuracy": val_m["accuracy"],
                        "lr": optimizer.param_groups[0]["lr"]})

        print(f"  Epoch {epoch:>3}/{max_epoch}  loss={total_loss/n:.4f}  val_f1={val_m['f1']:.4f}  val_acc={val_m['accuracy']:.4f}")

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> Saved best (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop:
                print(f"  >> Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s  |  Best F1: {best_f1:.4f}")

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()

    val_logits, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits, labels = forward_batch(model, batch, device)
            val_logits.append(logits.squeeze(-1))
            val_labels.append(labels)
    val_probs = torch.sigmoid(torch.cat(val_logits)).cpu().numpy()
    val_labels = torch.cat(val_labels).cpu().numpy()
    best_threshold, _ = find_best_threshold(val_labels, val_probs)
    print(f"  Tuned threshold: {best_threshold:.3f}")

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, labels = forward_batch(model, batch, device)
            all_logits.append(logits.squeeze(-1))
            all_labels.append(labels)

    y_prob = torch.sigmoid(torch.cat(all_logits)).cpu().numpy()
    y_true = torch.cat(all_labels).cpu().numpy()
    test_m = compute_metrics(y_true, y_prob, threshold=best_threshold)
    test_m.update({
        "model_name": name, "model_type": "neural",
        "train_time_s": elapsed, "epochs_trained": len(history),
        "num_params": sum(p.numel() for p in model.parameters()),
    })

    cm = test_m["confusion_matrix"]
    print(f"\n  Test Results for {name} (threshold={best_threshold:.3f}):")
    print(f"    F1={test_m['f1']:.4f}  Acc={test_m['accuracy']:.4f}  ROC-AUC={test_m['roc_auc']:.4f}")
    if len(cm) == 2:
        print(f"    Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    with open(os.path.join(checkpoints, f"{name}_results.json"), "w") as f:
        json.dump(test_m, f, indent=2)
    with open(os.path.join(checkpoints, f"{name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    np.save(os.path.join(checkpoints, f"{name}_y_true.npy"), y_true)
    np.save(os.path.join(checkpoints, f"{name}_y_prob.npy"), y_prob)

    return test_m


def train_xgboost(data, splits):
    import xgboost as xgb

    name = "xgboost"
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    numeric = data["numeric_features"].numpy()
    labels = data["labels"].numpy()
    user_ids = data["user_ids"]

    id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    train_idx = np.array([id_to_idx[uid] for uid in splits["train"]])
    val_idx   = np.array([id_to_idx[uid] for uid in splits.get("val", splits.get("valid"))])
    test_idx  = np.array([id_to_idx[uid] for uid in splits["test"]])

    x_train, y_train = numeric[train_idx], labels[train_idx]
    x_val,   y_val   = numeric[val_idx],   labels[val_idx]
    x_test,  y_test  = numeric[test_idx],  labels[test_idx]

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos
    print(f"  Class balance: {neg} humans, {pos} bots  |  scale_pos_weight={scale_pos_weight:.2f}")

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
    print(f"  Done in {elapsed:.1f}s  |  Best iteration: {model.best_iteration}")

    val_prob = model.predict_proba(x_val)[:, 1]
    best_threshold, _ = find_best_threshold(y_val, val_prob)
    print(f"  Tuned threshold: {best_threshold:.3f}")

    y_prob = model.predict_proba(x_test)[:, 1]
    test_m = compute_metrics(y_test, y_prob, threshold=best_threshold)
    test_m.update({
        "model_name": name, "model_type": "xgboost",
        "train_time_s": elapsed, "epochs_trained": int(model.best_iteration or 0),
        "num_params": model.n_estimators,
    })

    cm = test_m["confusion_matrix"]
    print(f"\n  Test Results for {name} (threshold={best_threshold:.3f}):")
    print(f"    F1={test_m['f1']:.4f}  Acc={test_m['accuracy']:.4f}  ROC-AUC={test_m['roc_auc']:.4f}")
    if len(cm) == 2:
        print(f"    Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    os.makedirs(checkpoints, exist_ok=True)
    model.save_model(os.path.join(checkpoints, f"{name}_best.json"))

    importance = dict(zip(
        [f"f{i}" for i in range(x_train.shape[1])],
        model.feature_importances_.tolist()
    ))

    with open(os.path.join(checkpoints, f"{name}_results.json"), "w") as f:
        json.dump(test_m, f, indent=2)
    with open(os.path.join(checkpoints, f"{name}_history.json"), "w") as f:
        json.dump([{"epoch": 1, "train_loss": 0.0, "val_f1": test_m["f1"],
                    "val_accuracy": test_m["accuracy"], "lr": 0.05}], f, indent=2)
    with open(os.path.join(checkpoints, f"{name}_feature_importance.json"), "w") as f:
        json.dump(importance, f, indent=2)
    np.save(os.path.join(checkpoints, f"{name}_y_true.npy"), y_test)
    np.save(os.path.join(checkpoints, f"{name}_y_prob.npy"), y_prob)

    return test_m


def train_xgboost_fused(data, splits):
    import xgboost as xgb
    from src.config import data_processed

    name = "xgboost_fused"
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    bio_path = os.path.join(data_processed, "bio_embeddings.npy")
    if not os.path.exists(bio_path):
        raise FileNotFoundError(f"missing {bio_path}; run python src/embed_bios.py first")
    bio = np.load(bio_path)
    numeric = data["numeric_features"].numpy()
    if bio.shape[0] != numeric.shape[0]:
        raise ValueError(f"row mismatch: numeric={numeric.shape[0]} bio={bio.shape[0]}")

    fused = np.concatenate([numeric, bio], axis=1).astype(np.float32)
    print(f"  fused features: {numeric.shape[1]} numeric + {bio.shape[1]} sbert = {fused.shape[1]}")

    labels = data["labels"].numpy()
    user_ids = data["user_ids"]
    id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    train_idx = np.array([id_to_idx[uid] for uid in splits["train"]])
    val_idx   = np.array([id_to_idx[uid] for uid in splits.get("val", splits.get("valid"))])
    test_idx  = np.array([id_to_idx[uid] for uid in splits["test"]])

    x_train, y_train = fused[train_idx], labels[train_idx]
    x_val,   y_val   = fused[val_idx],   labels[val_idx]
    x_test,  y_test  = fused[test_idx],  labels[test_idx]

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos
    print(f"  Class balance: {neg} humans, {pos} bots  |  scale_pos_weight={scale_pos_weight:.2f}")

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
    print(f"  Done in {elapsed:.1f}s  |  Best iteration: {model.best_iteration}")

    val_prob = model.predict_proba(x_val)[:, 1]
    best_threshold, _ = find_best_threshold(y_val, val_prob)
    print(f"  Tuned threshold: {best_threshold:.3f}")

    y_prob = model.predict_proba(x_test)[:, 1]
    test_m = compute_metrics(y_test, y_prob, threshold=best_threshold)
    test_m.update({
        "model_name": name, "model_type": "xgboost_fused",
        "train_time_s": elapsed, "epochs_trained": int(model.best_iteration or 0),
        "num_params": model.n_estimators,
        "num_features": fused.shape[1],
    })

    cm = test_m["confusion_matrix"]
    print(f"\n  Test Results for {name} (threshold={best_threshold:.3f}):")
    print(f"    F1={test_m['f1']:.4f}  Acc={test_m['accuracy']:.4f}  ROC-AUC={test_m['roc_auc']:.4f}")
    if len(cm) == 2:
        print(f"    Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    os.makedirs(checkpoints, exist_ok=True)
    model.save_model(os.path.join(checkpoints, f"{name}_best.json"))

    with open(os.path.join(checkpoints, f"{name}_results.json"), "w") as f:
        json.dump(test_m, f, indent=2)
    np.save(os.path.join(checkpoints, f"{name}_y_true.npy"), y_test)
    np.save(os.path.join(checkpoints, f"{name}_y_prob.npy"), y_prob)

    return test_m


def main():
    print("=" * 60)
    print("  Twitter/X Bot Detector: Training")
    print("=" * 60)

    try:
        data = load_processed()
    except FileNotFoundError as e:
        print(f"[!] {e}")
        return

    splits = data["splits"]
    train_ds, val_ds, test_ds = create_datasets(data, splits)

    train_labels = train_ds.labels.numpy()
    pos = (train_labels == 1).sum()
    neg = (train_labels == 0).sum()
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    print(f"\n[*] Class balance: {neg} humans, {pos} bots  |  pos_weight={pos_weight.item():.2f}")

    print("\n[*] Building vocabulary...")
    vocab = GloveVocab.build_from_corpus(train_ds.texts)
    os.makedirs(checkpoints, exist_ok=True)
    vocab.save(os.path.join(checkpoints, "vocab.json"))

    if os.path.exists(glove_file):
        embeddings = vocab.load_glove_embeddings()
    else:
        print(f"[!] GloVe not found, using random embeddings")
        embeddings = vocab.random_embeddings()

    glove_collate = partial(collate_glove, vocab=vocab)
    all_results = []

    print("\n" + "=" * 60 + "\n  MODEL 1: BiGRU-LSTM + GloVe\n" + "=" * 60)
    try:
        model = BiGRU_LSTM(vocab_size=vocab.vocab_size, pretrained_embeddings=embeddings)
        r = train_neural(model, "bigru_lstm",
                         DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=glove_collate),
                         DataLoader(val_ds, batch_size=64, collate_fn=glove_collate),
                         DataLoader(test_ds, batch_size=64, collate_fn=glove_collate),
                         pos_weight=pos_weight)
        all_results.append(r)
    except Exception as e:
        print(f"[!] BiGRU-LSTM failed: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 60 + "\n  MODEL 2: CNN + BiLSTM\n" + "=" * 60)
    try:
        model = CNN_BiLSTM(vocab_size=vocab.vocab_size, pretrained_embeddings=embeddings)
        r = train_neural(model, "cnn_bilstm",
                         DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=glove_collate),
                         DataLoader(val_ds, batch_size=64, collate_fn=glove_collate),
                         DataLoader(test_ds, batch_size=64, collate_fn=glove_collate),
                         pos_weight=pos_weight)
        all_results.append(r)
    except Exception as e:
        print(f"[!] CNN-BiLSTM failed: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 60 + "\n  MODEL 3: XGBoost (numeric features only)\n" + "=" * 60)
    try:
        r = train_xgboost(data, splits)
        all_results.append(r)
    except Exception as e:
        print(f"[!] XGBoost failed: {e}")
        import traceback; traceback.print_exc()

    if all_results:
        all_results.sort(key=lambda r: r["f1"], reverse=True)
        print(f"\n{'='*60}\n  COMPARISON\n{'='*60}")
        print(f"  {'Model':<25} {'F1':>6} {'Acc':>6} {'AUC':>6} {'Thresh':>7}")
        for r in all_results:
            best = " << BEST" if r == all_results[0] else ""
            print(f"  {r['model_name']:<25} {r['f1']:>6.4f} {r['accuracy']:>6.4f} {r['roc_auc']:>6.4f} {r['threshold']:>7.3f}{best}")

        with open(os.path.join(checkpoints, "best_model_info.json"), "w") as f:
            json.dump(all_results[0], f, indent=2)
        print(f"\n[+] Best model: {all_results[0]['model_name']}")
        print(f"    Start API: uvicorn api.app:app --reload")
    else:
        print("\n[!] No models trained.")


if __name__ == "__main__":
    main()

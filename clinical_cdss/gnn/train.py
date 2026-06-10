import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

from clinical_cdss.gnn.data import load_graph_data
from clinical_cdss.gnn.model import ClinicalHGAT


def _infer_dims(data):
    return {key: value.shape[1] for key, value in data.x_dict.items()}


def _metrics(y_true, y_prob, y_pred):
    out = {
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    specs = []
    for c in range(3):
        true_c = (np.array(y_true) == c)
        pred_c = (np.array(y_pred) == c)
        if not true_c.any():
            continue
        tn = ((~true_c) & (~pred_c)).sum()
        fp = ((~true_c) & pred_c).sum()
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    out["specificity"] = float(np.mean(specs)) if specs else 0.0
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except ValueError:
        out["auroc"] = 0.0
    return out


def train_one_fold(data, train_idx, val_idx, epochs=120, lr=1e-3, weight_decay=1e-4, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClinicalHGAT(_infer_dims(data), num_classes=3).to(device)
    labels = data.labels.to(device)
    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)

    class_counts = torch.bincount(labels[train_idx], minlength=3).float()
    class_weights = class_counts.sum() / class_counts.clamp_min(1.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = {"score": -1.0, "state": None, "metrics": None}
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)["logits"]
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)["logits"]
            prob = torch.softmax(logits[val_idx], dim=-1).detach().cpu().numpy()
            pred = logits[val_idx].argmax(dim=-1).detach().cpu().numpy()
            y_true = labels[val_idx].detach().cpu().numpy()
            metrics = _metrics(y_true, prob, pred)
            score = metrics["f1"]

        # Per-epoch logging every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={loss.item():.4f} | "
                  f"val_f1={score:.3f} | auroc={metrics.get('auroc', 0):.3f}")

        if score > best["score"]:
            best = {
                "score": score,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metrics": metrics,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Guard against best["state"] being None (should not occur in practice)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best["metrics"] or {}


def train_final_model(data, train_val_idx, epochs=120, lr=1e-3, weight_decay=1e-4, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClinicalHGAT(_infer_dims(data), num_classes=3).to(device)
    labels = data.labels.to(device)
    train_val_idx_tensor = torch.tensor(train_val_idx, dtype=torch.long, device=device)

    class_counts = torch.bincount(labels[train_val_idx_tensor], minlength=3).float()
    class_weights = class_counts.sum() / class_counts.clamp_min(1.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)["logits"]
        loss = criterion(logits[train_val_idx_tensor], labels[train_val_idx_tensor])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Final model epoch {epoch:3d}/{epochs} | loss={loss.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Final model early stopping at epoch {epoch}")
                break

    model.eval()
    return model


def train_cross_validation(folds=5, epochs=120, patience=20, model_out="models/clinical_hgat.pt"):
    print("Loading graph data from Neo4j...")
    data = load_graph_data()
    labels = data.labels.numpy()
    print(f"Loaded {len(labels)} patients | class distribution: {np.bincount(labels).tolist()}")

    if len(labels) < 5 or len(np.unique(labels)) < 2:
        raise ValueError("Need at least two classes and enough labeled patients to train HGAT.")

    # Split into 80% Train-Val and 20% independent Holdout Test set
    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.20,
        stratify=labels,
        random_state=42
    )
    train_val_labels = labels[train_val_idx]
    print(f"Split data: 80% Train-Val (N={len(train_val_idx)}) | 20% Holdout Test (N={len(test_idx)})")
    print(f"Train-Val class distribution: {np.bincount(train_val_labels).tolist()}")
    print(f"Holdout Test class distribution: {np.bincount(labels[test_idx]).tolist()}")

    min_class = np.bincount(train_val_labels).min()
    if min_class < 2:
        raise ValueError("Each class needs at least two patients in Train-Val for stratified cross-validation.")
    n_splits = min(folds, int(min_class))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    for fold, (train_sub_idx, val_sub_idx) in enumerate(splitter.split(np.zeros(len(train_val_labels)), train_val_labels), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        # Map sub-indices back to original indices
        original_train_idx = train_val_idx[train_sub_idx]
        original_val_idx = train_val_idx[val_sub_idx]
        model, metrics = train_one_fold(data, original_train_idx, original_val_idx, epochs=epochs, patience=patience)
        fold_metrics.append(metrics)
        print(f"Fold {fold} best: {metrics}")

    avg = {
        key: float(np.mean([m[key] for m in fold_metrics if key in m]))
        for key in fold_metrics[0]
    }
    print(f"\nCross-validation average (on Train-Val folds): {avg}")

    print("\nTraining final model on 80% Train-Val data...")
    final_model = train_final_model(data, train_val_idx, epochs=epochs, patience=patience)

    # Independent Evaluation on Holdout Test Set
    print("\nEvaluating final model on independent Holdout Test Set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.eval()
    with torch.no_grad():
        logits = final_model(data)["logits"]
        prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred = logits.argmax(dim=-1).detach().cpu().numpy()
        
        test_prob = prob[test_idx]
        test_pred = pred[test_idx]
        test_y_true = labels[test_idx]
        
        test_metrics = _metrics(test_y_true, test_prob, test_pred)
    
    print(f"Independent Holdout Test Metrics: {test_metrics}")

    test_patient_ids = [data.patient_ids[i] for i in test_idx]

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": final_model.state_dict(),
        "in_dim_dict": _infer_dims(data),
        "metrics": avg,
        "test_metrics": test_metrics,
        "test_patient_ids": test_patient_ids,
        "training_note": "Final model trained on Train/Val patient set (80%) and evaluated on independent Holdout Test patient set (20%).",
    }, out_path)
    print(f"Saved final model to {out_path}")
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--out", default="models/clinical_hgat.pt")
    args = parser.parse_args()
    train_cross_validation(args.folds, args.epochs, args.patience, args.out)

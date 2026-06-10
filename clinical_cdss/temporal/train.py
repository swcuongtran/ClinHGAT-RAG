import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import GroupKFold, train_test_split
from torch import nn

from clinical_cdss.temporal.data import load_temporal_prefix_data
from clinical_cdss.temporal.model import TemporalDiseaseForecaster


def _metrics(y_true, y_prob, y_pred):
    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    negatives = np.array(y_true) == 0
    out["specificity"] = float((np.array(y_pred)[negatives] == 0).mean()) if negatives.any() else 0.0
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auroc"] = 0.0
    return out


def train_final_model(data, train_val_idx, epochs=120):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalDiseaseForecaster(
        daily_dim=data.sequences.shape[-1],
        static_dim=data.static_features.shape[-1],
        hidden=32,
        heads=2,
        layers=1,
        dropout=0.4,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    train_val_labels = data.labels[train_val_idx]
    counts = torch.bincount(train_val_labels, minlength=2).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    train_val_idx_tensor = torch.tensor(train_val_idx, dtype=torch.long, device=device)
    labels = data.labels.to(device)
    sequences = data.sequences.to(device)
    static = data.static_features.to(device)
    masks = data.masks.to(device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(sequences[train_val_idx_tensor], static[train_val_idx_tensor], masks[train_val_idx_tensor])
        loss = criterion(output["logits"], labels[train_val_idx_tensor])
        loss.backward()
        optimizer.step()

    model.eval()
    return model.state_dict()


def train_temporal_forecaster(epochs=120, folds=5, model_out="models/temporal_forecaster.pt"):
    # Attempt to import StratifiedGroupKFold locally to avoid dependency issues
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:
        StratifiedGroupKFold = None

    data = load_temporal_prefix_data()
    labels = data.labels.numpy()
    groups = np.array(data.patient_ids)
    
    # Extract unique patient IDs and their respective labels
    patient_labels = {}
    for patient_id, label in zip(data.patient_ids, labels):
        patient_labels.setdefault(patient_id, int(label))
    
    unique_patient_ids = np.array(list(patient_labels.keys()))
    unique_labels = np.array([patient_labels[pid] for pid in unique_patient_ids])

    if len(unique_patient_ids) < 5 or len(np.unique(unique_labels)) < 2:
        raise ValueError("Need enough labeled patients from both classes for temporal training.")

    # Patient-level Group Split: 20% holdout test patients
    train_val_patients, test_patients = train_test_split(
        unique_patient_ids,
        test_size=0.20,
        stratify=unique_labels,
        random_state=42
    )
    
    train_val_set = set(train_val_patients)
    test_set = set(test_patients)
    
    train_val_idx = np.array([i for i, pid in enumerate(data.patient_ids) if pid in train_val_set])
    test_idx = np.array([i for i, pid in enumerate(data.patient_ids) if pid in test_set])

    print(f"Group Split: 80% Train-Val ({len(train_val_patients)} patients, {len(train_val_idx)} prefix samples)")
    print(f"Group Split: 20% Holdout Test ({len(test_patients)} patients, {len(test_idx)} prefix samples)")

    train_val_labels_subset = labels[train_val_idx]
    train_val_groups = groups[train_val_idx]
    train_val_sequences = data.sequences[train_val_idx]

    # Calculate CV splits on the Train-Val subset only
    patient_labels_tv = {pid: patient_labels[pid] for pid in train_val_patients}
    group_class_counts = np.bincount(list(patient_labels_tv.values()), minlength=2)
    min_group_class = int(group_class_counts.min())
    
    if min_group_class < 2:
        raise ValueError("Each class needs at least two patients in Train-Val for grouped cross-validation.")

    n_splits = min(folds, len(train_val_patients), min_group_class)
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(train_val_sequences, train_val_labels_subset, train_val_groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(train_val_sequences, train_val_labels_subset, train_val_groups)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_score = -1.0
    fold_metrics = []
    for fold, (train_sub_idx, val_sub_idx) in enumerate(split_iter, 1):
        # Map sub-indices back to the original index in the full dataset
        original_train_idx = train_val_idx[train_sub_idx]
        original_val_idx = train_val_idx[val_sub_idx]

        model = TemporalDiseaseForecaster(
            daily_dim=data.sequences.shape[-1],
            static_dim=data.static_features.shape[-1],
            hidden=32,
            heads=2,
            layers=1,
            dropout=0.4,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        train_labels = data.labels[original_train_idx]
        counts = torch.bincount(train_labels, minlength=2).float()
        weights = counts.sum() / counts.clamp_min(1.0)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))

        train_idx_t = torch.tensor(original_train_idx, dtype=torch.long, device=device)
        val_idx_t = torch.tensor(original_val_idx, dtype=torch.long, device=device)
        labels_t = data.labels.to(device)
        sequences = data.sequences.to(device)
        static = data.static_features.to(device)
        masks = data.masks.to(device)

        best_fold = {"score": -1.0, "state": None, "metrics": None}
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(sequences[train_idx_t], static[train_idx_t], masks[train_idx_t])
            loss = criterion(output["logits"], labels_t[train_idx_t])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                output = model(sequences[val_idx_t], static[val_idx_t], masks[val_idx_t])
                prob = torch.softmax(output["logits"], dim=-1)[:, 1].cpu().numpy()
                pred = output["logits"].argmax(dim=-1).cpu().numpy()
                y_true = labels_t[val_idx_t].cpu().numpy()
                metrics = _metrics(y_true, prob, pred)
                if metrics["f1"] > best_fold["score"]:
                    best_fold = {
                        "score": metrics["f1"],
                        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "metrics": metrics,
                    }

        fold_metrics.append(best_fold["metrics"])
        print(f"Fold {fold}: {best_fold['metrics']}")
        if best_fold["score"] > best_score:
            best_score = best_fold["score"]

    avg = {key: float(np.mean([m[key] for m in fold_metrics if key in m])) for key in fold_metrics[0]}
    print(f"Average (on Train-Val folds): {avg}")

    print("\nTraining final model on 80% Train-Val data (all patient prefix samples)...")
    final_model_state = train_final_model(data, train_val_idx, epochs=epochs)

    # Independent Evaluation on Holdout Test Set
    print("\nEvaluating final model on independent Holdout Test Set...")
    final_model = TemporalDiseaseForecaster(
        daily_dim=data.sequences.shape[-1],
        static_dim=data.static_features.shape[-1],
        hidden=32,
        heads=2,
        layers=1,
        dropout=0.4,
    ).to(device)
    final_model.load_state_dict(final_model_state)
    final_model.eval()
    
    with torch.no_grad():
        test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=device)
        sequences = data.sequences.to(device)
        static = data.static_features.to(device)
        masks = data.masks.to(device)
        labels_t = data.labels.to(device)

        output = final_model(sequences[test_idx_t], static[test_idx_t], masks[test_idx_t])
        prob = torch.softmax(output["logits"], dim=-1)[:, 1].cpu().numpy()
        pred = output["logits"].argmax(dim=-1).cpu().numpy()
        y_true = labels_t[test_idx_t].cpu().numpy()
        
        test_metrics = _metrics(y_true, prob, pred)
        
    print(f"Independent Holdout Test Metrics (temporal prefixes): {test_metrics}")

    out = Path(model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": final_model_state,
        "daily_dim": data.sequences.shape[-1],
        "static_dim": data.static_features.shape[-1],
        "hidden": 32,
        "heads": 2,
        "layers": 1,
        "dropout": 0.4,
        "feature_names": data.feature_names,
        "static_feature_names": data.static_feature_names,
        "mean": data.mean,
        "std": data.std,
        "static_mean": data.static_mean,
        "static_std": data.static_std,
        "symptom_names": data.symptom_names,
        "safe_concept_names": data.safe_concept_names,
        "metrics": avg,
        "test_metrics": test_metrics,
        "test_patient_ids": [str(p) for p in test_patients],
        "training_note": "Final model trained on Train/Val patient set (80%) and evaluated on independent Holdout Test patient set (20%).",
    }, out)
    print(f"Saved temporal forecaster to {out}")
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default="models/temporal_forecaster.pt")
    args = parser.parse_args()
    train_temporal_forecaster(args.epochs, args.folds, args.out)

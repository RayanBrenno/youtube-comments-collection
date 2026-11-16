"""Métricas de classificação (scikit-learn, cálculo local)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(preds, labels, average: str) -> dict:
    """accuracy + f1 + precision + recall a partir de predições já decididas."""
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average=average, zero_division=0),
        "precision": precision_score(labels, preds, average=average, zero_division=0),
        "recall": recall_score(labels, preds, average=average, zero_division=0),
    }


def save_per_class_report(preds, labels, label_names: list[str], out_dir: Path, model_slug: str) -> None:
    """Salva classification_report (txt) + matriz de confusão (csv) por modelo."""
    from sklearn.metrics import classification_report, confusion_matrix

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = list(range(len(label_names)))

    report = classification_report(
        labels, preds, labels=idx, target_names=label_names, digits=4, zero_division=0
    )
    (out_dir / f"{model_slug}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(labels, preds, labels=idx)
    df_cm = pd.DataFrame(cm, index=[f"real_{n}" for n in label_names],
                         columns=[f"pred_{n}" for n in label_names])
    df_cm.to_csv(out_dir / f"{model_slug}_confusion_matrix.csv")
    print(f">>> Relatório por classe salvo em {out_dir}/{model_slug}_*")

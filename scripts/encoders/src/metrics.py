"""Métricas de qualidade (via `scikit-learn`) e de custo (params, tamanho, latência)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(preds, labels, average: str) -> dict:
    """accuracy + f1 + precision + recall a partir de predições já decididas.

    Usa scikit-learn (cálculo local, sem chamada de rede) em vez de `evaluate.load()`,
    que baixa o script de cada métrica do Hub a cada chamada — inviável aqui porque
    compute_classification_metrics roda a cada época/modelo.
    """
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average=average, zero_division=0),
        "precision": precision_score(labels, preds, average=average, zero_division=0),
        "recall": recall_score(labels, preds, average=average, zero_division=0),
    }


def build_compute_metrics(average: str):
    """Retorna a função compute_metrics para o Trainer (converte logits -> preds)."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_classification_metrics(preds, labels, average)

    return compute_metrics


def save_per_class_report(preds, labels, label_names: list[str], out_dir: Path, model_slug: str) -> None:
    """Salva classification_report (txt) + matriz de confusão (csv) por modelo.

    É o que mostra ONDE o modelo erra. Na tarefa de 3 classes, a confusão
    neutro<->positivo costuma concentrar quase todo o erro, e isso não aparece
    num F1 agregado.
    """
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


def model_cost_metrics(model, saved_dir: str | Path) -> dict:
    """Nº de parâmetros (milhões) e tamanho em disco (MB) do modelo salvo."""
    num_params = model.num_parameters()
    size_bytes = sum(
        f.stat().st_size for f in Path(saved_dir).rglob("*") if f.is_file()
    )
    return {
        "num_params_M": round(num_params / 1e6, 2),
        "size_mb": round(size_bytes / 1e6, 2),
    }


@torch.no_grad()
def measure_latency(model, tokenized_test, n_samples: int, batch_size: int = 1) -> float:
    """Latência média de inferência por amostra (ms), com warm-up.

    Mede o forward puro (sem tokenização), uma amostra por vez por padrão, que é o
    cenário típico de inferência online — o que mais importa para tomada de decisão.
    """
    device = next(model.parameters()).device
    model.eval()

    n = min(n_samples, len(tokenized_test))
    keep = [c for c in ("input_ids", "attention_mask", "token_type_ids") if c in tokenized_test.column_names]
    subset = tokenized_test.select(range(n)).with_format("torch", columns=keep)

    def _forward(i):
        inputs = {k: subset[i][k].unsqueeze(0).to(device) for k in keep}
        model(**inputs)

    # Warm-up (não cronometrado) para estabilizar CUDA/caches.
    for i in range(min(3, n)):
        _forward(i)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for i in range(n):
        _forward(i)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return round((elapsed / n) * 1000, 2)  # ms por amostra

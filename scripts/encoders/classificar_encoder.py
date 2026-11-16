"""Classifica o dataset 2 (teste) com um encoder fine-tuned salvo (só inferência).

É daqui que saem as predições por comentário e as métricas reportadas no TCC.
Usa só a coluna `text`; compara com `sentimento` e salva:
  <saida.csv>                       uma linha por comentário (pred + confiança)
  <saida sem .csv>_relatorio.txt    acurácia, F1/precisão/recall macro, relatório por
                                    classe e matriz de confusão
Não sobrescreve um arquivo de saída existente.

Uso:
    python classificar_encoder.py <pasta_do_modelo> <saida.csv>

Exemplo:
    python classificar_encoder.py modelos/etapa2/xlm-roberta-base \\
        resultados/etapa2/predicoes/xlm-roberta-base.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATASET = "dataset/dataset2_teste.csv"
LABELS = ["negativo", "neutro", "positivo"]


def relatorio(y_true: list[str], preds: list[str]) -> str:
    """Texto com as métricas de um modelo (o mesmo que vai para o _relatorio.txt)."""
    cm = pd.DataFrame(
        confusion_matrix(y_true, preds, labels=LABELS),
        index=[f"real_{n}" for n in LABELS],
        columns=[f"pred_{n}" for n in LABELS],
    )
    return (
        f"N={len(y_true)}\n"
        f"Accuracy={accuracy_score(y_true, preds):.4f}\n"
        f"F1-macro={f1_score(y_true, preds, average='macro'):.4f}\n"
        f"Precision-macro={precision_score(y_true, preds, average='macro'):.4f}\n"
        f"Recall-macro={recall_score(y_true, preds, average='macro'):.4f}\n\n"
        f"{classification_report(y_true, preds, labels=LABELS, digits=4)}\n"
        f"Matriz de confusão (linhas = real, colunas = predito):\n{cm.to_string()}\n"
    )


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir, out_path = sys.argv[1], Path(sys.argv[2])
    rel_path = out_path.with_name(out_path.stem + "_relatorio.txt")
    for p in (out_path, rel_path):
        if p.exists():
            raise SystemExit(f"{p} já existe — não sobrescrevo predições coletadas.")

    df = pd.read_csv(DATASET).dropna(subset=["text", "sentimento"])
    texts = df["text"].astype(str).tolist()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    id2label = model.config.id2label

    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=256, padding=True,
                      return_tensors="pt").to(dev)
            c, idx = model(**enc).logits.softmax(-1).max(-1)
            preds += [id2label[int(j)] for j in idx.cpu()]
            confs += c.cpu().tolist()

    df["pred"] = preds
    df["confianca_modelo"] = confs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["comment_id", "text", "sentimento", "pred", "confianca_modelo"]].to_csv(out_path, index=False)

    texto = relatorio(df["sentimento"].tolist(), preds)
    rel_path.write_text(texto, encoding="utf-8")
    print(f"device={dev}  -> {out_path}\n{texto}")


if __name__ == "__main__":
    main()

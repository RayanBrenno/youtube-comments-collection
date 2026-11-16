"""Classificação com os encoders fine-tuned do TCC (scripts/encoders/modelos/etapa2/).

Faz a mesma inferência de scripts/encoders/classificar_encoder.py, que gerou as métricas do TCC:
lotes de 64, truncamento em 256 tokens, fp32, e a confiança é a maior probabilidade do
softmax. O rótulo vem do id2label gravado no config.json do modelo
(negativo / neutro / positivo).

Cada modelo é carregado na primeira vez que é usado e fica em memória (GPU se houver).
torch e transformers só são importados aqui, então o backend sobe rápido e funciona
só com o Claude se os modelos não estiverem disponíveis.
"""
from __future__ import annotations

import threading
from pathlib import Path

BATCH_SIZE = 64
MAX_LENGTH = 256

_load_lock = threading.Lock()
_cache: dict[str, tuple] = {}  # pasta -> (tokenizer, model, device, lock de inferência)


def carregar(pasta: Path) -> tuple:
    """Carrega (uma única vez) o tokenizer e o modelo salvos em `pasta`."""
    chave = str(pasta)
    if chave in _cache:
        return _cache[chave]

    with _load_lock:
        if chave not in _cache:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(chave)
            model = AutoModelForSequenceClassification.from_pretrained(chave).to(device).eval()
            _cache[chave] = (tokenizer, model, device, threading.Lock())
            print(f">>> Encoder carregado: {pasta.name} ({device})")
    return _cache[chave]


def classificar(textos: list[str], pasta: Path) -> list[tuple[str, float]]:
    """Retorna (rótulo, confiança) para cada texto, na mesma ordem."""
    if not textos:
        return []

    import torch

    tokenizer, model, device, infer_lock = carregar(pasta)
    id2label = model.config.id2label
    resultados: list[tuple[str, float]] = []

    with infer_lock, torch.no_grad():
        for i in range(0, len(textos), BATCH_SIZE):
            enc = tokenizer(
                textos[i:i + BATCH_SIZE],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            conf, idx = model(**enc).logits.softmax(-1).max(-1)
            resultados += [
                (id2label[int(j)], round(float(c), 4))
                for j, c in zip(idx.cpu(), conf.cpu())
            ]
    return resultados

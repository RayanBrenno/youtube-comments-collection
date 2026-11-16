"""Carrega o CSV de comentários e tipa a coluna de rótulo como ClassLabel."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import ClassLabel, Dataset

from .config import Config


@dataclass
class RawData:
    train: Dataset  # só usado se llm.num_fewshot > 0 (exemplos no prompt)
    test: Dataset
    num_labels: int
    label_names: list[str]


def load_raw_data(cfg: Config) -> RawData:
    """Sem treino: o arquivo inteiro é o conjunto avaliado.

    Os rótulos ficam em ordem alfabética (negativo, neutro, positivo), que também é a
    ordem em que aparecem no prompt.
    """
    df = pd.read_csv(cfg.dataset.path)
    label_column = cfg.dataset.label_column
    label_names = sorted(set(df[label_column]))
    df[label_column] = df[label_column].map({name: i for i, name in enumerate(label_names)})
    if df[label_column].isna().any():
        raise ValueError(f"Rótulos fora de {label_names} na coluna '{label_column}'.")
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.cast_column(label_column, ClassLabel(names=label_names))
    return RawData(train=ds, test=ds, num_labels=len(label_names), label_names=label_names)

"""Carregamento, split e tokenização do dataset.

O dataset bruto é carregado UMA vez (e fica fixo: mesmo split + seed para todos os
modelos). A tokenização, porém, é refeita por modelo, pois cada um tem seu vocabulário.

Splits
------
train / val / test. O `val` é opcional (dataset.val_size) e existe para a SELEÇÃO de
modelo (early stopping, melhor checkpoint). O `test` só é tocado na avaliação final —
usar o test para escolher o checkpoint inflaria as métricas reportadas.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import ClassLabel, Dataset

from .config import Config


@dataclass
class RawData:
    train: Dataset
    test: Dataset
    num_labels: int
    label_names: list[str]
    val: Optional[Dataset] = None


def _num_labels(ds: Dataset, label_column: str) -> tuple[int, list[str]]:
    feature = ds.features[label_column]
    if isinstance(feature, ClassLabel):
        return feature.num_classes, list(feature.names)
    # Fallback: rótulos inteiros sem metadados de ClassLabel.
    labels = sorted(set(ds[label_column]))
    return len(labels), [str(v) for v in labels]


def _from_pandas_with_labels(df: pd.DataFrame, label_column: str, label_names: list[str]) -> Dataset:
    """Converte rótulos string -> índice no pandas antes de tipar como ClassLabel.

    `cast_column` com ClassLabel não converte strings em versões recentes do `datasets`
    (ArrowInvalid: "Failed to parse string ... as int64"), então o mapeamento é explícito.
    """
    df = df.copy()
    df[label_column] = df[label_column].map({name: i for i, name in enumerate(label_names)})
    if df[label_column].isna().any():
        raise ValueError(f"Rótulos fora de {label_names} na coluna '{label_column}'.")
    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.cast_column(label_column, ClassLabel(names=label_names))


def _read_local_df(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato de dataset local não suportado: {path.suffix}")


def _load_local_file(cfg: Config, path_str: Optional[str] = None,
                     label_names: Optional[list[str]] = None) -> Dataset:
    """Carrega um dataset local (.xlsx/.csv) e tipa a coluna de rótulo como ClassLabel."""
    df = _read_local_df(path_str or cfg.dataset.path)
    label_column = cfg.dataset.label_column
    label_names = label_names or sorted(set(df[label_column]))
    return _from_pandas_with_labels(df, label_column, label_names)


def load_raw_data(cfg: Config) -> RawData:
    """Carrega o dataset, garante splits train/val/test e aplica max_samples."""
    label_column = cfg.dataset.label_column
    val = None

    ds = _load_local_file(cfg)
    if cfg.eval_only and cfg.dataset.eval_full:
        # Sem treino, não há o que separar: avalia o modelo no dataset inteiro.
        train, test = ds, ds
    elif cfg.dataset.test_path:
        # Test set em arquivo separado: usa como está, mesmos rótulos do train.
        train = ds
        test = _load_local_file(cfg, cfg.dataset.test_path, ds.features[label_column].names)
    else:
        # Split ESTRATIFICADO para preservar a proporção das classes no teste.
        split = ds.train_test_split(
            test_size=cfg.dataset.test_size,
            seed=cfg.seed,
            stratify_by_column=label_column,
        )
        train, test = split["train"], split["test"]

    num_labels, label_names = _num_labels(train, label_column)

    # Subconjunto opcional para smoke test (embaralha com a seed antes de cortar).
    if cfg.dataset.max_samples:
        n_train = cfg.dataset.max_samples
        n_test = max(1, cfg.dataset.max_samples // 4)
        train = train.shuffle(seed=cfg.seed).select(range(min(n_train, len(train))))
        test = test.shuffle(seed=cfg.seed).select(range(min(n_test, len(test))))

    # Split de validação retirado do TRAIN (o test permanece intocado até o fim).
    if cfg.dataset.val_size > 0:
        inner = train.train_test_split(
            test_size=cfg.dataset.val_size,
            seed=cfg.seed,
            stratify_by_column=label_column,
        )
        train, val = inner["train"], inner["test"]

    return RawData(
        train=train, test=test, val=val, num_labels=num_labels, label_names=label_names
    )


def tokenize_dataset(raw: RawData, tokenizer, cfg: Config):
    """Tokeniza train/val/test para um modelo específico, padronizando 'labels'."""
    text_column = cfg.dataset.text_column
    label_column = cfg.dataset.label_column

    def _tok(batch):
        enc = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=cfg.training.max_length,
        )
        enc["labels"] = batch[label_column]
        return enc

    def _apply(ds: Optional[Dataset]):
        if ds is None:
            return None
        remove = [c for c in ds.column_names if c not in ("labels",)]
        return ds.map(_tok, batched=True, remove_columns=remove)

    return _apply(raw.train), _apply(raw.val), _apply(raw.test)

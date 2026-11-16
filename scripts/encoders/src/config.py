"""Carregamento e validação dos configs YAML (pasta configs/)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DatasetConfig:
    name: str
    path: str  # arquivo local (.xlsx/.csv) de treino
    text_column: str = "text"
    label_column: str = "label"
    max_samples: Optional[int] = None
    test_size: float = 0.2  # só usado se test_path não for definido
    val_size: float = 0.0  # fração do TRAIN separada p/ early stopping (0 = desligado)
    test_path: Optional[str] = None  # test set local pronto (ignora test_size)
    eval_full: bool = True  # eval_only: True = avalia no arquivo inteiro; False = só no split de teste


@dataclass
class TrainingConfig:
    num_train_epochs: float = 2
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_length: int = 256

    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    label_smoothing_factor: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    precision: str = "auto"  # auto (bf16 > fp16 > fp32 conforme a GPU) | bf16 | fp16 | fp32
    gradient_checkpointing: bool = False
    group_by_length: bool = True  # agrupa textos de tamanho similar: menos padding
    dataloader_num_workers: int = 0  # 0 é o seguro no Windows (sem fork)

    # Seleção de modelo por early stopping (exige dataset.val_size > 0).
    eval_strategy: str = "epoch"  # epoch | steps | no
    eval_steps: int = 500
    early_stopping_patience: int = 0  # 0 = desligado
    metric_for_best_model: str = "f1"

    # Reexecução automática com batch menor se a GPU estourar a memória.
    # O batch efetivo é mantido dobrando o gradient_accumulation_steps.
    oom_retry: bool = True
    min_batch_size: int = 4


@dataclass
class Config:
    seed: int
    dataset: DatasetConfig
    models: list[str]
    training: TrainingConfig
    metric_average: str = "binary"
    latency_samples: int = 100
    output_dir: str = "resultados"
    eval_only: bool = False  # True = só avalia modelos já fine-tuned (pula o treino)
    save_models: bool = False  # True = persiste cada modelo treinado em disco p/ reuso
    models_dir: str = "modelos"  # pasta onde os modelos treinados são salvos
    per_class_report: bool = False  # salva matriz de confusão + relatório por classe

    @property
    def results_csv(self) -> Path:
        return Path(self.output_dir) / "results.csv"

    @property
    def reports_dir(self) -> Path:
        return Path(self.output_dir) / "reports"


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        seed=int(raw["seed"]),
        dataset=DatasetConfig(**raw["dataset"]),
        models=list(raw.get("models") or []),
        training=TrainingConfig(**raw["training"]),
        metric_average=raw.get("metric_average", "binary"),
        latency_samples=int(raw.get("latency_samples", 100)),
        output_dir=raw.get("output_dir", "resultados"),
        eval_only=bool(raw.get("eval_only", False)),
        save_models=bool(raw.get("save_models", False)),
        models_dir=raw.get("models_dir", "modelos"),
        per_class_report=bool(raw.get("per_class_report", False)),
    )

    # Validações básicas para falhar cedo com mensagem clara.
    if not cfg.models:
        raise ValueError("Nada a avaliar: a lista 'models' está vazia.")
    if cfg.metric_average not in {"binary", "weighted", "macro", "micro"}:
        raise ValueError(
            f"metric_average inválido: {cfg.metric_average!r}. "
            "Use 'binary' (2 classes) ou 'weighted'/'macro'/'micro' (multiclasse)."
        )
    if cfg.training.precision not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError(f"training.precision inválido: {cfg.training.precision!r}.")
    if cfg.training.early_stopping_patience > 0 and cfg.dataset.val_size <= 0:
        raise ValueError(
            "early_stopping_patience > 0 exige dataset.val_size > 0 — sem split de "
            "validação a seleção de modelo olharia o test set (vazamento)."
        )
    if not 0.0 <= cfg.dataset.val_size < 1.0:
        raise ValueError(f"dataset.val_size deve estar em [0, 1): {cfg.dataset.val_size}")
    return cfg

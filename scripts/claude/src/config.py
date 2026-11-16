"""Carregamento do config YAML (pasta configs/)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DatasetConfig:
    name: str
    path: str  # CSV com os comentários a classificar
    text_column: str = "text"
    label_column: str = "sentimento"
    id_column: Optional[str] = None  # id estável por linha (predições por amostra / retomada)


@dataclass
class LLMModelConfig:
    id: str
    price_in: float = 0.0   # US$ por 1M tokens de entrada
    price_out: float = 0.0  # US$ por 1M tokens de saída
    # Raciocínio enviado à API. Padrão: desligado. null no YAML = não envia o parâmetro
    # (modelos em que ele não pode ser desligado, ex.: claude-opus-5-5; aí só reduz o effort).
    thinking: Optional[dict] = field(default_factory=lambda: {"type": "disabled"})
    effort: Optional[str] = None      # output_config.effort (None = padrão do modelo)
    max_tokens: Optional[int] = None  # sobrepõe llm.max_tokens (ex.: espaço para o thinking)


@dataclass
class LLMConfig:
    num_fewshot: int = 0
    concurrency: int = 8
    max_tokens: int = 16
    dry_run: bool = False  # True = só estima o custo via count_tokens (gratuito), sem classificar
    limit_samples: Optional[int] = None  # piloto: subconjunto estratificado
    max_cost_usd: Optional[float] = None  # teto de gasto por modelo (cancela o restante)
    models: list[LLMModelConfig] = field(default_factory=list)


@dataclass
class Config:
    seed: int
    dataset: DatasetConfig
    llm: LLMConfig
    output_dir: str = "resultados"
    per_class_report: bool = False  # salva matriz de confusão + relatório por classe
    # Média usada no `f1`/`precision`/`recall` que o llm.py registra. O results.csv
    # traz também macro e weighted separadamente (ver run_claude.py).
    metric_average: str = "weighted"

    @property
    def results_csv(self) -> Path:
        return Path(self.output_dir) / "results.csv"

    @property
    def reports_dir(self) -> Path:
        return Path(self.output_dir) / "reports"


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw_llm = dict(raw["llm"])
    models = [LLMModelConfig(**m) for m in raw_llm.pop("models", [])]
    cfg = Config(
        seed=int(raw["seed"]),
        dataset=DatasetConfig(**raw["dataset"]),
        llm=LLMConfig(**raw_llm, models=models),
        output_dir=raw.get("output_dir", "resultados"),
        per_class_report=bool(raw.get("per_class_report", False)),
    )
    if not cfg.llm.models:
        raise ValueError("Nada a avaliar: llm.models está vazio.")
    return cfg

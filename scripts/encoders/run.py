"""Entry point: fine-tuning + avaliação dos encoders de um config.

Carrega o dataset uma vez, faz o fine-tuning e a avaliação de cada modelo do config
e salva as métricas em <output_dir>/results.csv.

Uso:
    python run.py --config configs/etapa1_cardiff.yaml
    python run.py --config configs/etapa2.yaml
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_config
from src.data import load_raw_data
from src.train import run_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning e avaliação de encoders.")
    parser.add_argument("--config", required=True, help="Caminho do config YAML (pasta configs/).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config: dataset={cfg.dataset.name} | modelos={cfg.models} | seed={cfg.seed}")

    raw = load_raw_data(cfg)
    val_txt = f" / {len(raw.val)} validação" if raw.val is not None else ""
    print(
        f"Dataset carregado: {len(raw.train)} treino{val_txt} / {len(raw.test)} teste | "
        f"{raw.num_labels} classes ({raw.label_names})"
    )

    rows = []
    for model_name in cfg.models:
        try:
            rows.append(run_one(model_name, raw, cfg))
        except Exception as exc:  # não derruba o estudo inteiro por causa de um modelo
            print(f"!!! Falha no modelo {model_name}: {exc}")

    if not rows:
        raise SystemExit("Nenhum modelo concluiu com sucesso — nada a reportar.")

    df = pd.DataFrame(rows)
    cfg.results_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.results_csv, index=False)
    print(f"\nResultados salvos em {cfg.results_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

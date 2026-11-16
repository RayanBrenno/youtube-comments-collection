"""Classifica o dataset com os modelos Claude do config e salva as métricas.

Uso:
    python run_claude.py --config configs/claude_5_5.yaml

Saídas em <output_dir>/:
    predictions/<modelo>.csv   uma linha por comentário (resposta, tokens, latência, custo)
    reports/<modelo>_*         relatório por classe e matriz de confusão
    results.csv                uma linha por modelo
"""
from __future__ import annotations

import argparse

import pandas as pd
from dotenv import load_dotenv

# Carrega o .env (ANTHROPIC_API_KEY) antes de qualquer uso da API. override=True: o .env
# do projeto vence uma chave antiga que esteja no ambiente do sistema.
load_dotenv(override=True)

from src.config import load_config
from src.data import load_raw_data
from src.llm import estimate_llm_cost, run_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Classificação com Claude via API.")
    parser.add_argument("--config", required=True, help="Caminho do config YAML (pasta configs/).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = load_raw_data(cfg)
    print(f"Config: dataset={cfg.dataset.name} | {len(raw.test)} comentários | "
          f"classes {raw.label_names} | modelos={[m.id for m in cfg.llm.models]}")

    # dry_run: só estima o custo (count_tokens é gratuito) e sai sem classificar.
    if cfg.llm.dry_run:
        for llm_model in cfg.llm.models:
            try:
                print(estimate_llm_cost(llm_model, raw, cfg))
            except Exception as exc:
                print(f"!!! Falha na estimativa do LLM {llm_model.id}: {exc}")
        return

    rows = []
    for llm_model in cfg.llm.models:
        try:
            rows.append(run_llm(llm_model, raw, cfg))
        except Exception as exc:  # um modelo com erro não derruba os outros
            print(f"!!! Falha no LLM {llm_model.id}: {exc}")

    if not rows:
        raise SystemExit("Nenhum modelo concluiu com sucesso — nada a reportar.")

    df = pd.DataFrame(rows)
    cfg.results_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.results_csv, index=False)
    print(f"\nResultados salvos em {cfg.results_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

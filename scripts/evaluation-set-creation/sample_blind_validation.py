"""
Sorteia a amostra de validação às cegas: SAMPLE_PER_THEME comentários de
cada tema do dataset (`dataset_validacao/dataset_validacao.csv`), com semente
fixa, para o autor classificar antes de ver os rótulos das LLMs.

O arquivo gerado (`amostra_cega.csv`) tem só comment_id, texto e uma coluna
`sentimento` vazia, sem tema, canal ou qualquer outra informação, e as
linhas são embaralhadas para os temas não aparecerem agrupados.

Uso:
    python evaluation-set-creation/sample_blind_validation.py
"""

import csv
import random
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset_validacao"
DATASET = DATASET_DIR / "dataset_validacao.csv"
OUTPUT = DATASET_DIR / "amostra_cega.csv"

SEED = 42
SAMPLE_PER_THEME = 20


def main():
    if OUTPUT.exists():
        raise SystemExit(
            f"{OUTPUT.name} já existe. Apague-o manualmente se quiser sortear de novo "
            "(isso descarta qualquer classificação já feita nele)."
        )

    with open(DATASET, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(f"{SEED}-amostra-cega")
    sample = []
    for tema in sorted({row["tema"] for row in rows}):
        theme_rows = sorted((r for r in rows if r["tema"] == tema), key=lambda r: r["comment_id"])
        sample += rng.sample(theme_rows, SAMPLE_PER_THEME)
    rng.shuffle(sample)

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["comment_id", "text", "sentimento"])
        writer.writerows((row["comment_id"], row["text"], "") for row in sample)

    print(f"Amostra cega: {len(sample)} comentários → {OUTPUT}")


if __name__ == "__main__":
    main()

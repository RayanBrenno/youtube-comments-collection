"""
Prepara os lotes de anotação para serem colados no chat das LLMs.

Cada lote (`dataset_validacao/llm/lotes/lote_XX.txt`) contém o prompt de
`prompt_anotacao.md` (o bloco entre as duas primeiras linhas `---`) seguido
de BATCH_SIZE comentários em JSON. Os comentários são embaralhados com
semente fixa, para os lotes misturarem temas e vídeos, e recebem ids curtos
(c0001, c0002, ...) para o modelo não errar ao copiá-los na resposta. A
correspondência id curto → comment_id fica em `llm/ids.csv`.

Também cria `llm/respostas/<modelo>.txt` vazios, onde as respostas de cada
modelo devem ser coladas (uma após a outra) para collect_llm_labels.py ler.

Uso:
    python evaluation-set-creation/prepare_llm_batches.py
"""

import csv
import json
import random
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset_validacao"
DATASET = DATASET_DIR / "dataset_validacao.csv"
PROMPT_FILE = DATASET_DIR / "prompt_anotacao.md"
LLM_DIR = DATASET_DIR / "llm"
BATCHES_DIR = LLM_DIR / "lotes"
RESPONSES_DIR = LLM_DIR / "respostas"
IDS_FILE = LLM_DIR / "ids.csv"

SEED = 42
BATCH_SIZE = 50
MODELS = ["gpt", "deepseek", "gemini"]


def load_prompt() -> str:
    """O prompt é o bloco entre as duas primeiras linhas '---' do .md."""
    lines = PROMPT_FILE.read_text(encoding="utf-8").splitlines()
    separators = [i for i, line in enumerate(lines) if line.strip() == "---"]
    return "\n".join(lines[separators[0] + 1 : separators[1]]).strip()


def batch_message(prompt: str, items: list[dict]) -> str:
    comments = ",\n".join(json.dumps(item, ensure_ascii=False) for item in items)
    return f"{prompt}\n\nClassifique os comentários a seguir:\n[\n{comments}\n]\n"


def main():
    if IDS_FILE.exists():
        raise SystemExit(
            f"{IDS_FILE} já existe. Os lotes já foram gerados; apague a pasta llm/ "
            "manualmente só se quiser recomeçar a anotação do zero."
        )

    with open(DATASET, encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["comment_id"])
    random.Random(f"{SEED}-lotes-llm").shuffle(rows)

    prompt = load_prompt()
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    with open(IDS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "comment_id", "lote"])
        for start in range(0, len(rows), BATCH_SIZE):
            batch_number = start // BATCH_SIZE + 1
            items = []
            for offset, row in enumerate(rows[start : start + BATCH_SIZE]):
                short_id = f"c{start + offset + 1:04d}"
                items.append({"id": short_id, "texto": row["text"]})
                writer.writerow([short_id, row["comment_id"], batch_number])
            (BATCHES_DIR / f"lote_{batch_number:02d}.txt").write_text(
                batch_message(prompt, items), encoding="utf-8"
            )

    for model in MODELS:
        (RESPONSES_DIR / f"{model}.txt").touch()

    total_batches = -(-len(rows) // BATCH_SIZE)
    print(f"{len(rows)} comentários em {total_batches} lotes de até {BATCH_SIZE} → {BATCHES_DIR}")
    print(f"Cole as respostas em: {', '.join(f'respostas/{m}.txt' for m in MODELS)}")


if __name__ == "__main__":
    main()

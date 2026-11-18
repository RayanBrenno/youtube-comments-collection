"""
Lê as respostas das LLMs coladas em `dataset_validacao/llm/respostas/<modelo>.txt`
(uma resposta após a outra, em qualquer formato que contenha o array JSON),
confere se cada modelo classificou todos os comentários e consolida os
rótulos.

Enquanto faltar algum comentário em algum modelo, gera
`llm/refazer_<modelo>.txt` (prompt + só os comentários faltantes) para colar
de novo no chat daquele modelo.

Com os 3 modelos completos, gera em `dataset_validacao/`:
- divergencias.csv        → comentários sem unanimidade (fora da amostra cega),
                            só id e texto, para classificação manual sem ver
                            os rótulos das LLMs. Preencha `sentimento` e salve
                            como divergencias_classificado.csv (ou .xlsx).
- dataset_rotulado.csv    → dataset com os rótulos de cada LLM, o humano e o final
- metricas_anotacao.json  → concordância entre LLMs e Kappa de Cohen contra a
                            amostra cega

Regra do rótulo final: amostra cega → rótulo dos dois anotadores humanos onde
concordam, consenso da adjudicação (adjudicacao.xlsx) onde discordam;
unanimidade das 3 LLMs → rótulo delas; demais → classificação manual
(divergencias_classificado.csv).

Uso:
    python evaluation-set-creation/collect_llm_labels.py
"""

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path

from prepare_llm_batches import MODELS, batch_message, load_prompt

DATASET_DIR = Path(__file__).resolve().parent / "dataset_validacao"
DATASET = DATASET_DIR / "dataset_validacao.csv"
BLIND_FILE = DATASET_DIR / "amostra_cega_classificado.csv"
BLIND_FILE_2 = DATASET_DIR / "amostra_cega_classificado_2.csv"
ADJUDICATION_FILE = DATASET_DIR / "adjudicacao.xlsx"
DIVERGENT_FILE = DATASET_DIR / "divergencias.csv"
DIVERGENT_LABELED_FILE = DATASET_DIR / "divergencias_classificado.csv"
LLM_DIR = DATASET_DIR / "llm"
IDS_FILE = LLM_DIR / "ids.csv"
RESPONSES_DIR = LLM_DIR / "respostas"

LABELS = {"positivo", "negativo", "neutro"}


OBJECT_PATTERN = re.compile(r"\{[^{}]*\}")


def extract_objects(text: str) -> list[dict]:
    """
    Encontra cada objeto JSON do texto colado, um por um. Aceita texto e
    ```json ao redor, respostas cortadas no meio e a marcação que algumas
    interfaces adicionam ao copiar (barra no fim da linha, &#x20; etc.).
    """
    text = html.unescape(text).replace("\\\n", "\n")
    objects = []
    for match in OBJECT_PATTERN.finditer(text):
        try:
            value = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def parse_responses(model: str, valid_ids: set[str]) -> tuple[dict[str, str], Counter]:
    """Devolve {id curto: rótulo} e a contagem de problemas encontrados."""
    text = (RESPONSES_DIR / f"{model}.txt").read_text(encoding="utf-8")
    labels, problems = {}, Counter()

    for item in extract_objects(text):
        short_id = str(item.get("id", "")).strip()
        label = str(item.get("sentimento", "")).strip().lower()
        if short_id not in valid_ids:
            problems["id desconhecido"] += 1
        elif label not in LABELS:
            problems["rótulo inválido"] += 1
        else:
            if short_id in labels and labels[short_id] != label:
                problems["respondido 2x com rótulos diferentes (vale o último)"] += 1
            labels[short_id] = label

    return labels, problems


def majority(votes: list[str]) -> str | None:
    label, count = Counter(votes).most_common(1)[0]
    return label if count >= 2 else None


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[k] * count_b[k] for k in LABELS) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def load_labeled_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {
            r["comment_id"]: r["sentimento"].strip().lower()
            for r in csv.DictReader(f) if r["sentimento"].strip()
        }


def load_labeled_xlsx(path: Path, label_column: str) -> dict[str, str]:
    """Rótulos válidos de uma planilha (primeira aba) com comment_id e `label_column`."""
    if not path.exists():
        return {}
    from openpyxl import load_workbook

    sheet = load_workbook(path, read_only=True, data_only=True).worksheets[0]
    header, *rows = list(sheet.iter_rows(values_only=True))
    labels = {}
    for row in rows:
        values = dict(zip(header, row))
        label = str(values.get(label_column) or "").strip().lower()
        if values.get("comment_id") and label in LABELS:
            labels[str(values["comment_id"]).strip()] = label
    return labels


def load_adjudication() -> dict[str, str]:
    """Rótulos de consenso preenchidos em adjudicacao.xlsx (coluna `consenso`)."""
    return load_labeled_xlsx(ADJUDICATION_FILE, "consenso")


def load_blind_gold() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """
    Devolve (gabarito, anotador 1, anotador 2) da amostra cega. Onde os dois
    anotadores concordam, vale o rótulo deles; onde discordam, o consenso da
    adjudicação (vazio enquanto não for preenchido).
    """
    first, second = load_labeled_csv(BLIND_FILE), load_labeled_csv(BLIND_FILE_2)
    if not second:
        return first, first, {}
    consensus = load_adjudication()
    gold = {
        cid: label if second.get(cid) == label else consensus.get(cid, "")
        for cid, label in first.items()
    }
    return gold, first, second


def main():
    with open(IDS_FILE, encoding="utf-8") as f:
        id_rows = list(csv.DictReader(f))
    to_comment = {r["id"]: r["comment_id"] for r in id_rows}

    with open(DATASET, encoding="utf-8") as f:
        dataset = list(csv.DictReader(f))
    texts = {r["comment_id"]: r["text"] for r in dataset}

    llm_labels, complete = {}, True
    for model in MODELS:
        labels, problems = parse_responses(model, set(to_comment))
        missing = [i for i in to_comment if i not in labels]
        print(f"- {model}: {len(labels)}/{len(to_comment)} classificados"
              + "".join(f" | {p}: {n}" for p, n in problems.items()))

        redo_file = LLM_DIR / f"refazer_{model}.txt"
        if missing:
            complete = False
            items = [{"id": i, "texto": texts[to_comment[i]]} for i in missing]
            redo_file.write_text(batch_message(load_prompt(), items), encoding="utf-8")
            print(f"    faltam {len(missing)} → cole {redo_file.name} numa conversa nova do {model}")
        elif redo_file.exists():
            redo_file.unlink()

        llm_labels[model] = {to_comment[i]: label for i, label in labels.items()}

    if not complete:
        print("\nAnotação incompleta: complete os modelos acima e rode de novo.")
        return

    blind, annotator_1, annotator_2 = load_blind_gold()
    manual = load_labeled_csv(DIVERGENT_LABELED_FILE) or load_labeled_xlsx(
        DIVERGENT_LABELED_FILE.with_suffix(".xlsx"), "sentimento"
    )

    output, divergent = [], []
    for row in dataset:
        cid = row["comment_id"]
        votes = [llm_labels[m][cid] for m in MODELS]
        if cid in blind:
            origin, final = "cego", blind[cid]
        elif len(set(votes)) == 1:
            origin, final = "unanime", votes[0]
        else:
            origin, final = "divergente", manual.get(cid, "")
            divergent.append(row)

        out = {k: v for k, v in row.items() if k != "sentimento"}
        out.update({f"llm_{m}": llm_labels[m][cid] for m in MODELS})
        out.update({
            "rotulo_anotador_1": annotator_1.get(cid, ""),
            "rotulo_anotador_2": annotator_2.get(cid, ""),
            "rotulo_humano_cego": blind.get(cid, ""),
            "rotulo_manual": manual.get(cid, ""),
            "rotulo_final": final,
            "origem": origin,
        })
        output.append(out)

    if not DIVERGENT_FILE.exists():
        with open(DIVERGENT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["comment_id", "text", "sentimento"])
            writer.writerows((r["comment_id"], r["text"], "") for r in divergent)
        print(f"\n{len(divergent)} divergências para classificar → {DIVERGENT_FILE.name}")

    with open(DATASET_DIR / "dataset_rotulado.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)

    all_votes = {cid: [llm_labels[m][cid] for m in MODELS] for cid in texts}
    unanimous = sum(len(set(v)) == 1 for v in all_votes.values())

    blind_ids = [cid for cid in blind if cid in texts and blind[cid]]
    human = [blind[cid] for cid in blind_ids]
    kappas = {m: round(cohen_kappa(human, [llm_labels[m][c] for c in blind_ids]), 3) for m in MODELS}
    with_majority = [(blind[c], majority(all_votes[c])) for c in blind_ids if majority(all_votes[c])]
    kappas["maioria_3_llms"] = round(cohen_kappa(*map(list, zip(*with_majority))), 3)

    pending = sum(1 for r in output if not r["rotulo_final"])
    both = [cid for cid in annotator_1 if cid in annotator_2]
    metrics = {
        "kappa_entre_anotadores_humanos_pre_adjudicacao": (
            round(cohen_kappa([annotator_1[c] for c in both], [annotator_2[c] for c in both]), 3)
            if both else None
        ),
        "amostra_cega_pendente_adjudicacao": sum(1 for cid in blind if not blind[cid]),
        "comentarios": len(texts),
        "unanimes_3_llms": unanimous,
        "unanimes_pct": round(100 * unanimous / len(texts), 1),
        "divergentes_fora_amostra_cega": len(divergent),
        "kappa_cohen_vs_amostra_cega": kappas,
        "amostra_cega_sem_maioria": len(blind_ids) - len(with_majority),
        "distribuicao_rotulo_final": dict(Counter(r["rotulo_final"] for r in output if r["rotulo_final"])),
        "rotulos_finais_pendentes": pending,
    }
    with open(DATASET_DIR / "metricas_anotacao.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nKappa entre os anotadores humanos (pré-adjudicação): "
          f"{metrics['kappa_entre_anotadores_humanos_pre_adjudicacao']}")
    if metrics["amostra_cega_pendente_adjudicacao"]:
        print(f"⚠ {metrics['amostra_cega_pendente_adjudicacao']} comentários da amostra cega "
              "aguardam consenso em adjudicacao.xlsx (fora do Kappa abaixo)")
    print(f"Unanimidade das 3 LLMs: {unanimous}/{len(texts)} ({metrics['unanimes_pct']}%)")
    print("Kappa de Cohen contra o gabarito da amostra cega:")
    for name, kappa in kappas.items():
        print(f"  {name}: {kappa}")
    print(f"Rótulos finais pendentes (divergências sem classificação manual): {pending}")


if __name__ == "__main__":
    main()

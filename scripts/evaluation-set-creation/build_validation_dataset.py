"""
Monta o dataset de validação a partir dos comentários brutos coletados por
collect_comments.py (`dataset_validacao/comentarios_brutos/*.jsonl`).

1. Filtros objetivos, aplicados nesta ordem sobre todos os comentários:
   - contém link (spam/autopromoção)
   - menos de MIN_WORDS palavras (só emoji, "kkkk", só menção etc.)
   - mais de MAX_CHARS caracteres
   - texto duplicado (mantém só a ocorrência mais antiga do corpus)
   - idioma não português: confiança de português < MIN_PT_CONFIDENCE
     (lingua, restrito a pt/es/en). O limite é baixo de propósito: o
     detector confunde português informal ("que casa lindaaa", "SHOW DE
     BOLA") com espanhol/inglês, e excluir esses textos enviesaria o dataset.
     Os estrangeiros que passarem são repostos pela reserva na anotação.
2. Sorteio por vídeo com semente fixa (SEED): os COMMENTS_PER_VIDEO
   primeiros entram no dataset e os RESERVE_PER_VIDEO seguintes formam a
   lista de reserva, usada só para repor comentários invalidados depois por
   critério objetivo.
3. Exclusões da revisão manual (EXCLUSIONS_FILE: comment_id, motivo), como
   comentários estrangeiros que passaram pelo filtro de idioma. São
   removidos depois do sorteio, sem refazê-lo: os comentários seguintes
   sobem uma posição e o primeiro da reserva do vídeo entra no lugar.

Gera em `dataset_validacao/`:
- dataset_validacao.csv     → os comentários do dataset (sentimento vazio)
- reserva.csv               → a lista de reserva, na ordem do sorteio
- exportacao_metadados.json → semente, critérios e contagem por filtro

Uso:
    python evaluation-set-creation/build_validation_dataset.py
"""

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from lingua import Language, LanguageDetectorBuilder

from common import flatten_text

DATASET_DIR = Path(__file__).resolve().parent / "dataset_validacao"
MANIFEST = DATASET_DIR / "videos_selecionados.csv"
RAW_DIR = DATASET_DIR / "comentarios_brutos"
EXCLUSIONS_FILE = DATASET_DIR / "exclusoes_revisao.csv"

SEED = 42
COMMENTS_PER_VIDEO = 20
RESERVE_PER_VIDEO = 6
MIN_WORDS = 3
MAX_CHARS = 1000
MIN_PT_CONFIDENCE = 0.05

URL_PATTERN = re.compile(r"https?://|www\.|\b\S+\.(com|net|org|ly|gg|br)(/|\b)", re.I)
MENTION_PATTERN = re.compile(r"@\S+")
WORD_PATTERN = re.compile(r"[^\W\d_]+")  # sequências de letras (qualquer alfabeto)

DETECTOR_LANGUAGES = [Language.PORTUGUESE, Language.SPANISH, Language.ENGLISH]

OUTPUT_FIELDS = [
    "comment_id", "text", "tema", "category_id", "video_id", "canal",
    "video_publicado_em", "comentario_publicado_em", "likes", "reply_count",
    "n_caracteres", "n_palavras", "confianca_pt", "ordem_sorteio", "sentimento",
]


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(MENTION_PATTERN.sub(" ", text)))


def load_comments(video: dict) -> list[dict]:
    with open(RAW_DIR / f"{video['video_id']}.jsonl", encoding="utf-8") as f:
        comments = [json.loads(line) for line in f]
    for comment in comments:
        comment["text"] = flatten_text(comment["text"])
        comment["video"] = video
    return comments


def apply_filters(comments: list[dict]) -> tuple[list[dict], Counter]:
    """Devolve os comentários aprovados e a contagem de exclusões por filtro."""
    excluded = Counter()

    def keep(items, reason, predicate):
        kept = [c for c in items if predicate(c)]
        excluded[reason] += len(items) - len(kept)
        return kept

    comments = keep(comments, "contém link", lambda c: not URL_PATTERN.search(c["text"]))
    comments = keep(comments, f"menos de {MIN_WORDS} palavras",
                    lambda c: count_words(c["text"]) >= MIN_WORDS)
    comments = keep(comments, f"mais de {MAX_CHARS} caracteres",
                    lambda c: len(c["text"]) <= MAX_CHARS)

    seen = set()

    def first_occurrence(comment):
        key = comment["text"].lower()
        if key in seen:
            return False
        seen.add(key)
        return True

    comments.sort(key=lambda c: (c["published_at"], c["comment_id"]))
    comments = keep(comments, "texto duplicado", first_occurrence)

    detector = LanguageDetectorBuilder.from_languages(*DETECTOR_LANGUAGES).build()
    confidences = detector.compute_language_confidence_in_parallel(
        [c["text"] for c in comments], Language.PORTUGUESE
    )
    for comment, confidence in zip(comments, confidences):
        comment["confianca_pt"] = confidence
    comments = keep(comments, "idioma não português",
                    lambda c: c["confianca_pt"] >= MIN_PT_CONFIDENCE)

    return comments, excluded


def draw(comments: list[dict], video_id: str) -> list[dict]:
    """Embaralha os comentários do vídeo com semente derivada de SEED + video_id."""
    ordered = sorted(comments, key=lambda c: c["comment_id"])
    random.Random(f"{SEED}-{video_id}").shuffle(ordered)
    return ordered


def load_exclusions() -> dict[str, str]:
    if not EXCLUSIONS_FILE.exists():
        return {}
    with open(EXCLUSIONS_FILE, encoding="utf-8") as f:
        return {row["comment_id"]: row["motivo"] for row in csv.DictReader(f)}


def to_row(comment: dict, position: int) -> dict:
    video = comment["video"]
    return {
        "comment_id": comment["comment_id"],
        "text": comment["text"],
        "tema": video["tema"],
        "category_id": video["category_id"],
        "video_id": video["video_id"],
        "canal": video["canal"],
        "video_publicado_em": video["publicado_em"],
        "comentario_publicado_em": comment["published_at"],
        "likes": comment["likes"],
        "reply_count": comment["reply_count"],
        "n_caracteres": len(comment["text"]),
        "n_palavras": count_words(comment["text"]),
        "confianca_pt": round(comment["confianca_pt"], 3),
        "ordem_sorteio": position,
        "sentimento": "",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        videos = list(csv.DictReader(f))

    all_comments = [c for video in videos for c in load_comments(video)]
    print(f"Comentários brutos: {len(all_comments)} de {len(videos)} vídeos")

    approved, excluded = apply_filters(all_comments)
    for reason, count in excluded.items():
        print(f"  - {reason}: {count}")
    print(f"Aprovados nos filtros: {len(approved)}\n")

    by_video: dict[str, list[dict]] = {}
    for comment in approved:
        by_video.setdefault(comment["video"]["video_id"], []).append(comment)

    exclusions = load_exclusions()
    applied_exclusions = 0

    dataset, reserve, per_video = [], [], {}
    needed = COMMENTS_PER_VIDEO + RESERVE_PER_VIDEO
    for video in videos:
        video_id = video["video_id"]
        pool = by_video.get(video_id, [])
        drawn = []
        for comment in draw(pool, video_id):
            if len(drawn) == needed:
                break
            if comment["comment_id"] in exclusions:
                applied_exclusions += 1
                continue
            drawn.append(comment)
        rows = [to_row(c, i) for i, c in enumerate(drawn, start=1)]
        dataset += rows[:COMMENTS_PER_VIDEO]
        reserve += rows[COMMENTS_PER_VIDEO:]

        total = sum(1 for c in all_comments if c["video"]["video_id"] == video_id)
        per_video[video_id] = {
            "tema": video["tema"],
            "brutos": total,
            "aprovados_filtros": len(pool),
            "sorteados": len(drawn),
        }
        if len(drawn) < needed:
            print(f"⚠ {video['tema']} | {video['titulo'][:50]}: só {len(drawn)} aprovados")

    write_csv(DATASET_DIR / "dataset_validacao.csv", dataset)
    write_csv(DATASET_DIR / "reserva.csv", reserve)

    metadata = {
        "semente": SEED,
        "sorteio": "random.Random(f'{semente}-{video_id}').shuffle sobre os aprovados "
                   "ordenados por comment_id",
        "comentarios_por_video": COMMENTS_PER_VIDEO,
        "reserva_por_video": RESERVE_PER_VIDEO,
        "filtros": {
            "ordem": list(excluded.keys()),
            "min_palavras": MIN_WORDS,
            "max_caracteres": MAX_CHARS,
            "duplicados": "texto idêntico (sem diferenciar maiúsculas); mantém o mais antigo",
            "idioma": f"lingua-language-detector (pt/es/en); exclui confiança pt < {MIN_PT_CONFIDENCE}",
            "revisao_manual": "comentários com confiança pt < 0,5 revisados; excluídos os "
                              "escritos em outro idioma ou sem sentido (exclusoes_revisao.csv)",
        },
        "totais": {
            "brutos": len(all_comments),
            "excluidos_por_filtro": dict(excluded),
            "aprovados_filtros": len(approved),
            "excluidos_revisao_manual": applied_exclusions,
            "dataset": len(dataset),
            "reserva": len(reserve),
        },
        "por_video": per_video,
    }
    with open(DATASET_DIR / "exportacao_metadados.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    per_theme = Counter(row["tema"] for row in dataset)
    print("Dataset por tema:")
    for tema, count in per_theme.items():
        print(f"  {tema}: {count}")
    print(f"\nDataset: {len(dataset)} comentários | reserva: {len(reserve)}")
    print(f"Arquivos em: {DATASET_DIR}")


if __name__ == "__main__":
    main()

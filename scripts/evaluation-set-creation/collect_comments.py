"""
Coleta todos os comentários de primeiro nível dos vídeos do manifesto
(`dataset_validacao/videos_selecionados.csv`, gerado por select_videos.py)
e salva um snapshot bruto por vídeo em arquivos locais, sem passar pelo
banco da aplicação:

- comentarios_brutos/<video_id>.jsonl → um comentário por linha:
  comment_id, text, likes, reply_count, published_at, updated_at
  (o nome do autor não é salvo)
- coleta_comentarios.json → data da coleta e contagens de cada vídeo

Todos os comentários são baixados porque a API não oferece ordem aleatória
(só por data ou relevância): o sorteio da exportação precisa do conjunto
completo de cada vídeo para não enviesar a amostra.

O arquivo de um vídeo só é gravado depois que todas as páginas foram
baixadas. Se a cota acabar no meio, basta rodar de novo: vídeos já salvos
são pulados e o vídeo interrompido é baixado desde o início.

Custo de cota: 1 unidade a cada 100 comentários (commentThreads.list).

Uso:
    python evaluation-set-creation/collect_comments.py
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from googleapiclient.errors import HttpError

from common import get_youtube_client

DATASET_DIR = Path(__file__).resolve().parent / "dataset_validacao"
MANIFEST = DATASET_DIR / "videos_selecionados.csv"
RAW_DIR = DATASET_DIR / "comentarios_brutos"
METADATA_FILE = DATASET_DIR / "coleta_comentarios.json"


def fetch_comment_counts(youtube, video_ids: list[str]) -> dict[str, int]:
    """Total de comentários informado pelo YouTube (inclui respostas)."""
    counts = {}
    for i in range(0, len(video_ids), 50):
        response = youtube.videos().list(
            part="statistics",
            id=",".join(video_ids[i : i + 50]),
            maxResults=50,
        ).execute()
        for item in response.get("items", []):
            counts[item["id"]] = int(item.get("statistics", {}).get("commentCount", 0))
    return counts


def fetch_all_comments(youtube, video_id: str) -> list[dict]:
    """Todos os comentários de primeiro nível, do mais recente ao mais antigo."""
    comments = []
    page_token = None

    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=page_token,
            textFormat="plainText",
            order="time",
        ).execute()

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]
            snippet = top["snippet"]
            comments.append({
                "comment_id": top["id"],
                "text": snippet.get("textDisplay", ""),
                "likes": snippet.get("likeCount", 0),
                "reply_count": item["snippet"].get("totalReplyCount", 0),
                "published_at": snippet.get("publishedAt"),
                "updated_at": snippet.get("updatedAt"),
            })

        page_token = response.get("nextPageToken")
        if not page_token:
            return comments


def save_comments(video_id: str, comments: list[dict]) -> None:
    """Grava num arquivo temporário e renomeia, para nunca deixar arquivo pela metade."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = RAW_DIR / f"{video_id}.jsonl.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        for comment in comments:
            f.write(json.dumps(comment, ensure_ascii=False) + "\n")
    tmp_file.replace(RAW_DIR / f"{video_id}.jsonl")


def load_metadata() -> dict:
    if METADATA_FILE.exists():
        with open(METADATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"videos": {}}


def save_metadata(metadata: dict) -> None:
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def is_quota_error(e: HttpError) -> bool:
    return e.resp.status in (403, 429) and b"quota" in e.content.lower()


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    pending = [row for row in manifest if not (RAW_DIR / f"{row['video_id']}.jsonl").exists()]
    print(f"Manifesto: {len(manifest)} vídeos | já coletados: {len(manifest) - len(pending)} "
          f"| pendentes: {len(pending)}\n")

    metadata = load_metadata()
    youtube, counts = None, {}
    if pending:
        youtube = get_youtube_client()
        try:
            counts = fetch_comment_counts(youtube, [row["video_id"] for row in pending])
        except HttpError as e:
            if is_quota_error(e):
                sys.exit("Cota diária da API esgotada. Rode novamente depois que ela renovar.")
            raise

    for index, row in enumerate(pending, start=1):
        video_id = row["video_id"]
        label = f"[{index}/{len(pending)}] {row['tema']} | {row['titulo'][:50]}"

        if video_id not in counts:
            metadata["videos"][video_id] = {"tema": row["tema"], "erro": "vídeo indisponível"}
            save_metadata(metadata)
            print(f"{label}\n    ✗ vídeo indisponível (removido ou privado)", flush=True)
            continue

        try:
            comments = fetch_all_comments(youtube, video_id)
        except HttpError as e:
            if is_quota_error(e):
                sys.exit(
                    f"{label}\n    ✗ cota diária da API esgotada.\n"
                    "Os vídeos anteriores ficaram salvos; rode novamente depois "
                    "que a cota renovar para continuar deste vídeo."
                )
            if e.resp.status == 403 and b"commentsDisabled" in e.content:
                metadata["videos"][video_id] = {"tema": row["tema"], "erro": "comentários desativados"}
                save_metadata(metadata)
                print(f"{label}\n    ✗ comentários desativados", flush=True)
                continue
            raise

        save_comments(video_id, comments)
        metadata["videos"][video_id] = {
            "tema": row["tema"],
            "category_id": row["category_id"],
            "coletado_em": datetime.now(timezone.utc).isoformat(),
            "comentarios_primeiro_nivel": len(comments),
            "comentarios_informados_youtube": counts[video_id],
        }
        save_metadata(metadata)
        print(f"{label}\n    ✓ {len(comments)} comentários de primeiro nível "
              f"(YouTube informa {counts[video_id]} contando respostas)", flush=True)

    collected = [v for v in metadata["videos"].values() if "erro" not in v]
    failed = {vid: v for vid, v in metadata["videos"].items() if "erro" in v}
    print(f"\nVídeos coletados: {len(collected)}/{len(manifest)} | "
          f"comentários: {sum(v['comentarios_primeiro_nivel'] for v in collected)}")
    for vid, v in failed.items():
        print(f"⚠ {v['tema']} | {vid}: {v['erro']}")
    print(f"Arquivos em: {RAW_DIR}")


if __name__ == "__main__":
    main()

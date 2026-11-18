"""
Seleciona, de forma reproduzível, os vídeos que compõem o dataset de
validação: VIDEOS_PER_CATEGORY vídeos para cada uma das categorias oficiais
do YouTube em CATEGORIES.

Para cada categoria, percorre os resultados de `search.list` (regionCode=BR,
relevanceLanguage=pt, janela temporal fixa, order=viewCount) na ordem da
busca e aplica os critérios de inclusão/exclusão até completar a cota da
categoria. As páginas de busca (até MAX_SEARCH_PAGES) só são pedidas à API
quando os candidatos já obtidos não bastam, e ficam salvas em
`dataset_validacao/cache/<category_id>.json`: execuções seguintes reutilizam
o cache e só gastam cota se precisarem de páginas novas.

Para refazer a busca na API (novo snapshot), apague a pasta `cache/`.

Gera em `dataset_validacao/`:
- videos_selecionados.csv   → os vídeos aprovados (manifesto da coleta)
- candidatos_avaliados.csv  → todo candidato avaliado + motivo da exclusão
- coleta_metadados.json     → parâmetros, critérios e data da coleta

Custo de cota: ~102 unidades por página de busca (cota diária padrão: 10.000).

Uso:
    python evaluation-set-creation/select_videos.py
"""

import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from googleapiclient.errors import HttpError

from common import get_youtube_client

OUTPUT_DIR = Path(__file__).resolve().parent / "dataset_validacao"
CACHE_DIR = OUTPUT_DIR / "cache"

# Taxonomia oficial de categorias do YouTube (snippet.categoryId)
CATEGORIES = {
    "25": "Notícias e Política",
    "17": "Esportes",
    "20": "Games",
    "10": "Música",
    "24": "Entretenimento",
    "28": "Ciência e Tecnologia",
    "27": "Educação",
    "23": "Comédia",
    "22": "Pessoas e Blogs",
    "26": "Estilo e Tutoriais",
}

VIDEOS_PER_CATEGORY = 5
MAX_SEARCH_PAGES = 5  # 50 resultados por página → até 250 candidatos/categoria

# Filtro de duração da própria API (`videoDuration`): "medium" = 4 a 20 min,
# o que elimina Shorts já na busca. Música usa "any", pois clipes costumam
# ter menos de 4 min; nela os Shorts são identificados pelo formato vertical.
DEFAULT_SEARCH_DURATION = "medium"
SEARCH_DURATION = {"10": "any"}

PUBLISHED_AFTER = "2025-01-01T00:00:00Z"
PUBLISHED_BEFORE = "2026-10-01T00:00:00Z"
MIN_COMMENTS = 300
MAX_COMMENTS = 10_000
MIN_VIEWS = 10_000
SHORT_MAX_SECONDS = 180  # Shorts: até 3 min e formato vertical
PLAYER_MAX_WIDTH = 1280  # necessário para a API devolver embedWidth/embedHeight
ACCEPTED_LANGUAGES = {"pt", "pt-br"}

# Tópicos musicais da classificação automática do YouTube
# (topicDetails.topicCategories). Fora da categoria Música, vídeos com algum
# desses tópicos são excluídos: a categoria é declarada por quem publica e
# muitos clipes aparecem em Entretenimento ou Pessoas e Blogs.
MUSIC_CATEGORY_ID = "10"
MUSIC_TOPICS = {
    "Music", "Christian_music", "Classical_music", "Country_music",
    "Electronic_music", "Hip_hop_music", "Independent_music", "Jazz",
    "Music_of_Asia", "Music_of_Latin_America", "Pop_music", "Reggae",
    "Rhythm_and_blues", "Rock_music", "Soul_music",
}

CANDIDATE_FIELDS = [
    "category_id", "tema", "posicao_busca", "video_id", "titulo", "canal",
    "channel_id", "publicado_em", "views", "comments", "duracao_s",
    "orientacao", "idioma", "pais_canal", "topicos", "status", "motivo",
]
SELECTED_FIELDS = [
    f for f in CANDIDATE_FIELDS if f not in ("status", "motivo")
] + ["url"]


def parse_duration(iso: str) -> int:
    """Converte duração ISO 8601 (ex.: PT1H2M3S, P1DT2H) em segundos."""
    match = re.fullmatch(
        r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", iso or ""
    )
    if not match:
        return 0
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def orientation(video: dict) -> str:
    """'vertical', 'horizontal' ou '' (dimensões do player indisponíveis)."""
    player = video.get("player", {})
    width, height = player.get("embedWidth"), player.get("embedHeight")
    if not width or not height:
        return ""
    return "vertical" if int(height) > int(width) else "horizontal"


def topics(video: dict) -> list[str]:
    """Tópicos do YouTube do vídeo, ex.: ['Music', 'Entertainment']."""
    urls = video.get("topicDetails", {}).get("topicCategories", [])
    return [url.rsplit("/", 1)[-1] for url in urls]


def search_params(category_id: str) -> dict:
    return {
        "type": "video",
        "videoCategoryId": category_id,
        "regionCode": "BR",
        "relevanceLanguage": "pt",
        "publishedAfter": PUBLISHED_AFTER,
        "publishedBefore": PUBLISHED_BEFORE,
        "order": "viewCount",
        "videoDuration": SEARCH_DURATION.get(category_id, DEFAULT_SEARCH_DURATION),
    }


# ---------------------------------------------------------------------------
# Etapa 1: conjunto de candidatos (API ou cache)
# ---------------------------------------------------------------------------

def search_page(youtube, category_id: str, page_token: str | None) -> dict:
    return youtube.search().list(
        part="id",
        maxResults=50,
        pageToken=page_token,
        **search_params(category_id),
    ).execute()


def fetch_videos(youtube, video_ids: list[str]) -> dict[str, dict]:
    response = youtube.videos().list(
        part="snippet,statistics,contentDetails,status,liveStreamingDetails,player,topicDetails",
        id=",".join(video_ids),
        maxWidth=PLAYER_MAX_WIDTH,
        maxResults=50,
    ).execute()
    return {item["id"]: item for item in response.get("items", [])}


def fetch_channel_countries(youtube, channel_ids: list[str]) -> dict[str, str]:
    response = youtube.channels().list(
        part="snippet",
        id=",".join(set(channel_ids)),
        maxResults=50,
    ).execute()
    return {
        item["id"]: item["snippet"].get("country", "")
        for item in response.get("items", [])
    }


def fetch_page(youtube, category_id: str, page_token: str | None) -> tuple[list[dict], str | None]:
    """Busca uma página de resultados e devolve (vídeos, token da próxima página)."""
    response = search_page(youtube, category_id, page_token)
    video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
    if not video_ids:
        return [], None

    videos = fetch_videos(youtube, video_ids)
    countries = fetch_channel_countries(
        youtube, [v["snippet"]["channelId"] for v in videos.values()]
    )

    page = []
    for video_id in video_ids:
        video = videos.get(video_id)
        if video is None:  # removido/privado entre a busca e o videos.list
            continue
        video["channel_country"] = countries.get(video["snippet"]["channelId"], "")
        video.setdefault("topicDetails", {})  # a API omite o campo quando vazio
        page.append(video)

    return page, response.get("nextPageToken")


def ensure_topic_details(youtube_factory, category_id: str, state: dict) -> None:
    """Completa os tópicos de candidatos em cache salvos antes desse campo existir."""
    missing = [v for v in state["videos"] if "topicDetails" not in v]
    if not missing:
        return

    for i in range(0, len(missing), 50):
        batch = missing[i : i + 50]
        response = youtube_factory().videos().list(
            part="topicDetails",
            id=",".join(v["id"] for v in batch),
            maxResults=50,
        ).execute()
        found = {item["id"]: item.get("topicDetails", {}) for item in response.get("items", [])}
        for video in batch:
            video["topicDetails"] = found.get(video["id"], {})

    save_cache_state(category_id, state)


def load_cache_state(category_id: str) -> dict:
    """Carrega o cache da categoria (ou um estado vazio) e valida os parâmetros."""
    cache_file = CACHE_DIR / f"{category_id}.json"
    params = search_params(category_id)

    if not cache_file.exists():
        return {
            "coletado_em": datetime.now(timezone.utc).isoformat(),
            "parametros_busca": params,
            "paginas_buscadas": 0,
            "proximo_token": None,
            "completo": False,
            "videos": [],
        }

    with open(cache_file, encoding="utf-8") as f:
        state = json.load(f)
    if state.get("parametros_busca") != params:
        sys.exit(
            f"O cache {cache_file} foi gerado com outros parâmetros de busca.\n"
            "Mova ou apague a pasta cache/ para buscar novamente."
        )
    return state


def fetch_next_page(youtube_factory, category_id: str, state: dict) -> None:
    """Busca a próxima página de candidatos e grava o progresso no cache."""
    page, next_token = fetch_page(youtube_factory(), category_id, state["proximo_token"])
    state["videos"] += page
    state["paginas_buscadas"] += 1
    state["proximo_token"] = next_token
    state["completo"] = not next_token or state["paginas_buscadas"] >= MAX_SEARCH_PAGES
    save_cache_state(category_id, state)


def save_cache_state(category_id: str, state: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{category_id}.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Etapa 2: seleção (offline)
# ---------------------------------------------------------------------------

def exclusion_reason(video: dict, category_id: str, used_channels: set[str]) -> str | None:
    """Devolve o motivo da exclusão, ou None se o vídeo atende aos critérios."""
    snippet = video["snippet"]
    stats = video.get("statistics", {})
    country = video.get("channel_country", "")
    language = (
        snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage") or ""
    ).lower()

    if snippet.get("categoryId") != category_id:
        return "categoria divergente"
    if video.get("liveStreamingDetails") or snippet.get("liveBroadcastContent") != "none":
        return "live/transmissão"
    if video.get("status", {}).get("madeForKids"):
        return "conteúdo infantil"
    if parse_duration(video["contentDetails"].get("duration")) <= SHORT_MAX_SECONDS:
        if orientation(video) == "vertical":
            return "Short (vertical, <= 3 min)"
        if orientation(video) == "":
            return "formato não verificável (<= 3 min)"
    if category_id != MUSIC_CATEGORY_ID and MUSIC_TOPICS.intersection(topics(video)):
        return "tópico musical fora da categoria Música"
    if language and language not in ACCEPTED_LANGUAGES:
        return f"idioma não pt-BR ({language})"
    if country and country != "BR":
        return f"canal não brasileiro ({country})"
    if not language and not country:
        return "idioma e país não verificáveis"
    if "commentCount" not in stats:
        return "comentários desativados"
    if int(stats["commentCount"]) < MIN_COMMENTS:
        return f"menos de {MIN_COMMENTS} comentários"
    if int(stats["commentCount"]) > MAX_COMMENTS:
        return f"mais de {MAX_COMMENTS} comentários"
    if int(stats.get("viewCount", 0)) < MIN_VIEWS:
        return f"menos de {MIN_VIEWS} visualizações"
    if snippet["channelId"] in used_channels:
        return "canal já selecionado no dataset"
    return None


def to_row(video: dict, category_id: str, position: int) -> dict:
    snippet, stats = video["snippet"], video.get("statistics", {})
    return {
        "category_id": category_id,
        "tema": CATEGORIES[category_id],
        "posicao_busca": position,
        "video_id": video["id"],
        "titulo": snippet.get("title", ""),
        "canal": snippet.get("channelTitle", ""),
        "channel_id": snippet["channelId"],
        "publicado_em": snippet.get("publishedAt", ""),
        "views": stats.get("viewCount", ""),
        "comments": stats.get("commentCount", ""),
        "duracao_s": parse_duration(video["contentDetails"].get("duration")),
        "orientacao": orientation(video),
        "idioma": snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage") or "",
        "pais_canal": video.get("channel_country", ""),
        "topicos": ";".join(topics(video)),
    }


def select_for_category(youtube_factory, category_id: str, state: dict,
                        used_channels: set[str]) -> tuple[list[dict], list[dict]]:
    """
    Avalia os candidatos na ordem da busca até completar VIDEOS_PER_CATEGORY.
    Novas páginas só são buscadas na API quando os candidatos em cache acabam,
    o que dá o mesmo resultado de buscar todas e gasta só a cota necessária.

    `used_channels` é compartilhado entre as categorias (um canal por dataset),
    então a ordem de CATEGORIES define a prioridade em caso de conflito.
    """
    selected, evaluated = [], []
    position = 0

    while len(selected) < VIDEOS_PER_CATEGORY:
        if position == len(state["videos"]):
            if state["completo"]:
                break
            fetch_next_page(youtube_factory, category_id, state)
            ensure_topic_details(youtube_factory, category_id, state)
            continue

        video = state["videos"][position]
        position += 1
        reason = exclusion_reason(video, category_id, used_channels)
        row = to_row(video, category_id, position)
        row["status"] = "excluido" if reason else "selecionado"
        row["motivo"] = reason or ""
        evaluated.append(row)

        if reason is None:
            selected.append(row)
            used_channels.add(video["snippet"]["channelId"])

    return selected, evaluated


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    client = None

    def youtube_factory():
        nonlocal client
        if client is None:
            client = get_youtube_client()
        return client

    OUTPUT_DIR.mkdir(exist_ok=True)

    all_selected, all_evaluated, collection_dates, pages_fetched = [], [], {}, {}
    used_channels: set[str] = set()
    for category_id, tema in CATEGORIES.items():
        state = load_cache_state(category_id)
        try:
            ensure_topic_details(youtube_factory, category_id, state)
            selected, evaluated = select_for_category(
                youtube_factory, category_id, state, used_channels
            )
        except HttpError as e:
            if e.resp.status in (403, 429) and b"quota" in e.content.lower():
                sys.exit(
                    f"\nCota diária da API esgotada ao buscar '{tema}'.\n"
                    "O progresso ficou salvo em cache/ (página a página); "
                    "rode novamente depois que a cota renovar."
                )
            raise
        collection_dates[category_id] = state["coletado_em"]
        pages_fetched[category_id] = state["paginas_buscadas"]
        all_selected += selected
        all_evaluated += evaluated

        aviso = "" if len(selected) == VIDEOS_PER_CATEGORY else "  ⚠ cota incompleta"
        print(f"- {tema}: {len(selected)}/{VIDEOS_PER_CATEGORY} selecionados "
              f"({len(evaluated)} candidatos avaliados, "
              f"{state['paginas_buscadas']} página(s) de busca){aviso}")

        reasons = Counter(row["motivo"] for row in evaluated if row["motivo"])
        for reason, count in reasons.most_common():
            print(f"    {count:>4}  {reason}")

    for row in all_selected:
        row["url"] = f"https://www.youtube.com/watch?v={row['video_id']}"

    write_csv(OUTPUT_DIR / "candidatos_avaliados.csv", all_evaluated, CANDIDATE_FIELDS)
    write_csv(OUTPUT_DIR / "videos_selecionados.csv", all_selected, SELECTED_FIELDS)

    metadata = {
        "coletado_em_por_categoria": collection_dates,
        "categorias": CATEGORIES,
        "parametros_busca": {
            "endpoint": "search.list",
            "por_categoria": {cid: search_params(cid) for cid in CATEGORIES},
            "max_paginas_por_categoria": MAX_SEARCH_PAGES,
            "paginas_buscadas_por_categoria": pages_fetched,
        },
        "criterios": {
            "videos_por_categoria": VIDEOS_PER_CATEGORY,
            "comentarios_min": MIN_COMMENTS,
            "comentarios_max": MAX_COMMENTS,
            "views_min": MIN_VIEWS,
            "short": f"duração <= {SHORT_MAX_SECONDS}s e formato vertical (player.embedHeight > embedWidth)",
            "idiomas_aceitos": sorted(ACCEPTED_LANGUAGES),
            "pais_canal": "BR",
            "um_video_por_canal_no_dataset": True,
            "exclui_topicos_musicais_fora_de_musica": sorted(MUSIC_TOPICS),
            "exclui": ["lives", "conteúdo infantil", "Shorts", "comentários desativados"],
        },
        "totais": {
            "selecionados": len(all_selected),
            "avaliados": len(all_evaluated),
        },
    }
    with open(OUTPUT_DIR / "coleta_metadados.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nTotal selecionado: {len(all_selected)} vídeos "
          f"({len(all_evaluated)} candidatos avaliados)")
    print(f"Arquivos gerados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

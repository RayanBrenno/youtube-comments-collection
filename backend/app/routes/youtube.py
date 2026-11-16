from fastapi import APIRouter, HTTPException

from app.schemas.youtube import YoutubeURLRequest

from app.services.youtube_service import (
    get_code_url,
    get_video_info,
    get_all_comments,
)

from app.services.ia_service import (
    classificar_comentarios,
    classificador_padrao,
    listar_classificadores,
    resolver_classificador,
    resumo,
)

from app.youtube_repository import (
    filter_new_comments,
    get_comments_by_video_id,
    get_dashboard_summary,
    get_video_by_code_url_and_user,
    get_video_by_id_and_user,
    get_videos_by_user,
    insert_comments,
    insert_video_info,
    update_video_info,
)

router = APIRouter(prefix="/youtube", tags=["youtube"])


def _format_comments(saved_comments: list[dict]) -> list[dict]:
    return [
        {
            "author": comment["author"],
            "text": comment["text"],
            "likes": comment["likes"],
            "published_at": comment["published_at"],
            "intencao": comment["intencao"],
            "score": comment["score"],
        }
        for comment in saved_comments
    ]


def _collect_classify_and_save(
    code_url: str,
    user_id: str,
    existing_video: dict | None,
    requested_classifier: str | None,
) -> tuple[str, dict, str]:
    """Coleta vídeo + comentários, classifica só os novos e salva.

    Vídeo novo: usa o classificador pedido (ou o padrão).
    Vídeo já salvo: usa o classificador com que ele foi consultado da primeira vez.

    A classificação roda antes de gravar qualquer coisa: se ela falhar (ex.: chave da
    Anthropic inválida), o banco não fica com um vídeo sem comentários.

    Retorna (video_id, video_info, classificador).
    """
    if existing_video:
        classifier = existing_video.get("classificador") or classificador_padrao()
    else:
        classifier = requested_classifier or classificador_padrao()
    resolver_classificador(classifier)  # 400 logo, antes de gastar cota do YouTube

    video_info = get_video_info(code_url)
    comments = get_all_comments(code_url)

    if existing_video:
        video_id = str(existing_video["_id"])
        new_comments = filter_new_comments(comments, video_id)
    else:
        video_id = None
        new_comments = comments

    classified = classificar_comentarios(new_comments, classifier)

    if video_id:
        update_video_info(video_info, video_id)
    else:
        video_id = insert_video_info(video_info, user_id, classifier)
    insert_comments(classified, video_id)

    return video_id, video_info, classifier


@router.get("/classificadores")
def fetch_classifiers():
    """Classificadores que o usuário pode escolher e se cada um está disponível."""
    return listar_classificadores()


@router.post("/check-video")
def check_video(payload: YoutubeURLRequest):
    video_code_url = get_code_url(str(payload.url))

    existing_video = get_video_by_code_url_and_user(
        code_url=video_code_url,
        user_id=payload.user_id,
    )

    if not existing_video:
        return {
            "exists": False,
            "video": None,
        }

    return {
        "exists": True,
        "video": {
            "id": str(existing_video["_id"]),
            "code_url": existing_video.get("code_url"),
            "title": existing_video.get("title"),
            "channel": existing_video.get("channel"),
            "thumbnail_url": existing_video.get("thumbnail_url"),
            "last_updated_at": existing_video.get("last_updated_at"),
            "consulted_at": existing_video.get("consulted_at"),
            "views": existing_video.get("views"),
            "likes": existing_video.get("likes"),
            "comments": existing_video.get("comments"),
            "classificador": resumo(existing_video.get("classificador")),
        },
    }


@router.get("/history/{user_id}")
def fetch_video_history(user_id: str):
    return get_videos_by_user(user_id)


@router.get("/dashboard/{user_id}")
def fetch_dashboard_summary(user_id: str):
    return get_dashboard_summary(user_id)


@router.post("/full-data")
def fetch_full_video_data(payload: YoutubeURLRequest):
    code_url = get_code_url(str(payload.url))
    existing_video = get_video_by_code_url_and_user(code_url, payload.user_id)

    saved_video_id, video_info, classifier = _collect_classify_and_save(
        code_url=code_url,
        user_id=payload.user_id,
        existing_video=existing_video,
        requested_classifier=payload.classificador,
    )

    saved_comments = get_comments_by_video_id(saved_video_id)
    comments_response = _format_comments(saved_comments)

    return {
        "video_id": saved_video_id,
        "video": video_info,
        "classificador": resumo(classifier),
        "total_comments": len(comments_response),
        "comments": comments_response,
    }


@router.post("/videos/{video_id}/update")
def update_saved_video(video_id: str, user_id: str):
    old_video = get_video_by_id_and_user(
        video_id=video_id,
        user_id=user_id,
    )

    if not old_video:
        raise HTTPException(
            status_code=404,
            detail="Vídeo não encontrado para este usuário.",
        )

    code_url = old_video.get("code_url")

    if not code_url:
        raise HTTPException(
            status_code=400,
            detail="Vídeo salvo não possui code_url.",
        )

    saved_video_id, new_video_info, classifier = _collect_classify_and_save(
        code_url=code_url,
        user_id=user_id,
        existing_video=old_video,
        requested_classifier=None,
    )

    saved_comments = get_comments_by_video_id(saved_video_id)
    comments_response = _format_comments(saved_comments)

    old_views = int(old_video.get("views") or 0)
    old_likes = int(old_video.get("likes") or 0)
    old_comments_count = int(old_video.get("comments") or 0)

    new_views = int(new_video_info.get("views") or 0)
    new_likes = int(new_video_info.get("likes") or 0)
    new_comments_count = int(new_video_info.get("comments") or 0)

    return {
        "video_id": str(saved_video_id),
        "video": new_video_info,
        "classificador": resumo(classifier),
        "total_comments": len(comments_response),
        "comments": comments_response,
        "previous": {
            "views": old_views,
            "likes": old_likes,
            "comments": old_comments_count,
            "last_updated_at": old_video.get("last_updated_at"),
        },
        "current": {
            "views": new_views,
            "likes": new_likes,
            "comments": new_comments_count,
        },
        "diff": {
            "views": new_views - old_views,
            "likes": new_likes - old_likes,
            "comments": new_comments_count - old_comments_count,
        },
    }

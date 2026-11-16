from typing import Optional

from pydantic import BaseModel, HttpUrl


class YoutubeURLRequest(BaseModel):
    url: HttpUrl
    user_id: str
    # id de um classificador de GET /youtube/classificadores; None = o padrão do .env.
    # Ignorado se o vídeo já existe: os comentários novos usam o classificador do vídeo.
    classificador: Optional[str] = None

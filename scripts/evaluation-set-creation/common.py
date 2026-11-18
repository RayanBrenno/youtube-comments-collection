"""
Utilitários compartilhados pelos scripts de criação do conjunto de avaliação.

Centraliza o carregamento da chave da YouTube Data API v3, lida do .env do
backend, e a normalização de texto dos comentários.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ENV = Path(__file__).resolve().parent.parent / "backend" / ".env"


def get_youtube_client():
    """Carrega o .env do backend e devolve o cliente da YouTube Data API v3."""
    from googleapiclient.discovery import build

    load_dotenv(BACKEND_ENV)

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        sys.exit(f"YOUTUBE_API_KEY não encontrada em {BACKEND_ENV}")

    return build("youtube", "v3", developerKey=api_key)


def flatten_text(text: str) -> str:
    """Colapsa quebras de linha e espaços múltiplos em um único espaço."""
    return " ".join((text or "").split())

"""Carrega o backend/.env antes de qualquer módulo do app.

Este arquivo roda no primeiro `import app...`, então todos os os.getenv() dos módulos
(SECRET_KEY, MONGO_URL, YOUTUBE_API_KEY, ANTHROPIC_API_KEY...) já enxergam o .env,
independentemente da ordem dos imports e da pasta de onde o uvicorn foi iniciado.

override=True: o valor do .env vence uma variável antiga que esteja no ambiente do sistema
(ex.: outra ANTHROPIC_API_KEY exportada no shell).
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

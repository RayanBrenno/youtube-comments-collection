import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.routes.autenticacao import router as auth_router
from app.routes.youtube import router as youtube_router
from app.services.ia_service import precarregar_padrao


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Carrega o encoder padrão (ex.: BERTimbau Large) antes da primeira consulta.
    await run_in_threadpool(precarregar_padrao)
    yield


app = FastAPI(
    title="Projeto Coletar Comentários do YouTube",
    version="1.0.0",
    lifespan=lifespan,
)


def _allowed_origins() -> list[str]:
    """Origens do CORS a partir de API_FRONT_URL (uma ou várias, separadas por vírgula).

    Localmente, localhost e 127.0.0.1 são a mesma máquina mas origens diferentes para o
    navegador; as duas formas são liberadas para o front funcionar em qualquer uma.
    """
    raw = os.getenv("API_FRONT_URL") or "http://localhost:5173"
    origins: set[str] = set()
    for origin in (o.strip().rstrip("/") for o in raw.split(",")):
        if not origin:
            continue
        origins.add(origin)
        if "://localhost" in origin:
            origins.add(origin.replace("://localhost", "://127.0.0.1"))
        elif "://127.0.0.1" in origin:
            origins.add(origin.replace("://127.0.0.1", "://localhost"))
    return sorted(origins)


# Erros não tratados viram um 500 em JSON DENTRO do CORS. Sem isso, a resposta de erro sai
# sem os cabeçalhos de CORS e o navegador mostra um "erro de CORS" enganoso no lugar do erro
# real. Registrado antes do CORSMiddleware para ficar por dentro dele.
@app.middleware("http")
async def unhandled_errors_as_json(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Erro interno no servidor: {exc}"},
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(youtube_router)

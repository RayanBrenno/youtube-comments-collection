"""Classificação de sentimento dos comentários (negativo / neutro / positivo).

Registro dos classificadores que o usuário pode escolher:

- encoders fine-tuned do TCC, rodando localmente (pasta MODELOS_DIR, padrão
  scripts/encoders/modelos/etapa2). O backend só lê os modelos prontos: não depende
  de nenhum código de scripts/;
- modelos Claude 5.5 via API, disponíveis quando ANTHROPIC_API_KEY está no .env.

A acurácia mostrada é a do TCC no dataset 2 (500 comentários do YouTube rotulados à mão):
encoders em scripts/encoders/README.md, Claude em scripts/claude/README.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from app.services import classificador_claude, classificador_encoder

BACKEND_DIR = Path(__file__).resolve().parents[2]
MODELOS_DIR_PADRAO = BACKEND_DIR.parent / "scripts" / "encoders" / "modelos" / "etapa2"
CLASSIFICADOR_PADRAO = "bertimbau-large"


@dataclass(frozen=True)
class Classificador:
    id: str
    nome: str
    tipo: str  # "encoder" | "claude"
    alvo: str  # encoder: pasta dentro de MODELOS_DIR; claude: id do modelo na API
    acuracia_tcc: float
    f1_macro_tcc: float


CLASSIFICADORES: dict[str, Classificador] = {
    c.id: c
    for c in [
        Classificador("bertimbau-large", "BERTimbau Large", "encoder",
                      "neuralmind__bert-large-portuguese-cased", 0.788, 0.7799),
        Classificador("bertimbau-base", "BERTimbau Base", "encoder",
                      "neuralmind__bert-base-portuguese-cased", 0.780, 0.7658),
        Classificador("xlm-roberta", "XLM-RoBERTa Base", "encoder",
                      "xlm-roberta-base", 0.784, 0.7726),
        Classificador("distilbert", "DistilBERT multilíngue", "encoder",
                      "distilbert-base-multilingual-cased", 0.624, 0.5902),
        Classificador("claude-haiku-5-5", "Claude Haiku 5.5", "claude",
                      "claude-haiku-5-5", 0.8333, 0.8244),
        Classificador("claude-sonnet-5-5", "Claude Sonnet 5.5", "claude",
                      "claude-sonnet-5-5", 0.8380, 0.8289),
        Classificador("claude-opus-5-5", "Claude Opus 5.5", "claude",
                      "claude-opus-5-5", 0.8600, 0.8463),
    ]
}


def modelos_dir() -> Path:
    valor = os.getenv("MODELOS_DIR")
    if not valor:
        return MODELOS_DIR_PADRAO
    p = Path(valor).expanduser()
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


def _pasta_encoder(c: Classificador) -> Path:
    return modelos_dir() / c.alvo


def motivo_indisponivel(c: Classificador) -> str | None:
    """None se o classificador pode ser usado; senão, o motivo."""
    if c.tipo == "encoder":
        if not (_pasta_encoder(c) / "config.json").is_file():
            return f"Modelo não encontrado em {_pasta_encoder(c)} (rode o fine-tuning em scripts/encoders/)."
        return None
    if not classificador_claude.chave_configurada():
        return "Defina ANTHROPIC_API_KEY no backend/.env."
    return None


def classificador_padrao() -> str:
    padrao = os.getenv("CLASSIFICADOR_PADRAO") or CLASSIFICADOR_PADRAO
    return padrao if padrao in CLASSIFICADORES else CLASSIFICADOR_PADRAO


def listar_classificadores() -> dict:
    return {
        "padrao": classificador_padrao(),
        "classificadores": [
            {
                "id": c.id,
                "nome": c.nome,
                "tipo": c.tipo,
                "acuracia_tcc": c.acuracia_tcc,
                "f1_macro_tcc": c.f1_macro_tcc,
                "disponivel": motivo_indisponivel(c) is None,
                "motivo": motivo_indisponivel(c),
            }
            for c in CLASSIFICADORES.values()
        ],
    }


def resumo(classificador_id: str | None) -> dict | None:
    """{id, nome, tipo} para devolver ao front junto com o vídeo."""
    c = CLASSIFICADORES.get(classificador_id or "")
    if not c:
        return None
    return {"id": c.id, "nome": c.nome, "tipo": c.tipo}


def resolver_classificador(classificador_id: str | None) -> Classificador:
    """Valida o classificador pedido (ou o padrão) e devolve-o; 400 se não puder ser usado."""
    cid = classificador_id or classificador_padrao()
    c = CLASSIFICADORES.get(cid)
    if not c:
        raise HTTPException(status_code=400, detail=f"Classificador desconhecido: {cid}.")
    motivo = motivo_indisponivel(c)
    if motivo:
        raise HTTPException(status_code=400, detail=f"{c.nome} indisponível: {motivo}")
    return c


def precarregar_padrao() -> None:
    """Carrega o encoder padrão na subida do backend, para a 1ª consulta não esperar."""
    c = CLASSIFICADORES[classificador_padrao()]
    if c.tipo == "encoder" and motivo_indisponivel(c) is None:
        try:
            classificador_encoder.carregar(_pasta_encoder(c))
        except Exception as exc:  # não impede o backend de subir
            print(f"!!! Não foi possível pré-carregar {c.nome}: {exc}")


def classificar_comentarios(comments: list[dict], classificador_id: str) -> list[dict]:
    """Devolve os comentários com intencao, score e classificador preenchidos."""
    c = resolver_classificador(classificador_id)
    textos = [str(comment.get("text") or "") for comment in comments]

    try:
        if c.tipo == "encoder":
            resultados = classificador_encoder.classificar(textos, _pasta_encoder(c))
        else:
            resultados = classificador_claude.classificar(textos, c.alvo)
    except classificador_claude.ClaudeIndisponivel as exc:
        raise HTTPException(status_code=502, detail=f"{c.nome}: {exc}") from exc

    return [
        {**comment, "intencao": label, "score": score, "classificador": c.id}
        for comment, (label, score) in zip(comments, resultados)
    ]

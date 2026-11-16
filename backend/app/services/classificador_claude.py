"""Classificação zero-shot com os modelos Claude 5.5 via API.

Reproduz a configuração do experimento do TCC (scripts/claude/src/llm.py e
configs/claude_5_5.yaml), para que o sistema classifique como no experimento:

- uma chamada por comentário, 8 em paralelo;
- o mesmo prompt de sistema, com os rótulos em ordem alfabética;
- só o texto do comentário na mensagem do usuário, cortado em 1.500 caracteres;
- thinking / effort / max_tokens por modelo iguais aos do experimento;
- sem parâmetros de amostragem e sem fallback para outro modelo (cada resposta tem que vir
  do modelo escolhido);
- a mesma leitura da resposta: rótulo idêntico, senão rótulo contido na resposta.

O Claude devolve só o rótulo, sem probabilidade, então o score fica vazio (None).
Respostas sem rótulo válido, recusas (stop_reason "refusal") e chamadas que falharem depois
das novas tentativas do SDK ficam com intenção None ("não classificada" no front).

Requer ANTHROPIC_API_KEY no backend/.env.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

LABELS = ["negativo", "neutro", "positivo"]

SYSTEM_PROMPT = "\n".join([
    "You are a precise text classification model.",
    f"Classify each input into EXACTLY one of these labels: {', '.join(LABELS)}.",
    "Respond with ONLY the label word, exactly as written, and nothing else.",
])

MAX_CHARS = 1500
CONCORRENCIA = 8

# Opções de requisição por modelo, iguais às de configs/claude_5_5.yaml do experimento.
OPCOES_POR_MODELO: dict[str, dict] = {
    "claude-haiku-5-5": {
        "max_tokens": 16,
        "thinking": {"type": "disabled"},
    },
    "claude-sonnet-5-5": {
        "max_tokens": 16,
        "thinking": {"type": "between_tools"},  # forma de desligar nesse modelo
    },
    "claude-opus-5-5": {
        "max_tokens": 1024,  # espaço para o raciocínio, que não pode ser desligado
        "output_config": {"effort": "low"},
    },
}


class ClaudeIndisponivel(RuntimeError):
    """Erro que impede classificar qualquer comentário (chave inválida, modelo errado...)."""


_client = None
_client_lock = threading.Lock()


def chave_configurada() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            import anthropic

            if not chave_configurada():
                raise ClaudeIndisponivel("ANTHROPIC_API_KEY não configurada no backend/.env.")
            # O SDK já repete sozinho em limite de requisições (429), erro do servidor e
            # falha de conexão; 4 tentativas extras dão folga em vídeos com muitos comentários.
            _client = anthropic.Anthropic(max_retries=4)
    return _client


def parse_label(out_text: str) -> str | None:
    """Extrai o rótulo da resposta livre do modelo (mesma regra do experimento)."""
    s = out_text.strip().strip('".\'').lower()
    if s in LABELS:
        return s
    for label in LABELS:
        if label in s:
            return label
    return None


def _classificar_um(client, modelo: str, texto: str) -> str | None:
    import anthropic

    texto = texto[:MAX_CHARS]
    if not texto.strip():
        return None  # a API não aceita mensagem vazia

    try:
        resp = client.messages.create(
            model=modelo,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": texto}],
            **OPCOES_POR_MODELO[modelo],
        )
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
        raise ClaudeIndisponivel(f"Chave da API da Anthropic recusada: {exc.message}") from exc
    except anthropic.NotFoundError as exc:
        raise ClaudeIndisponivel(f"Modelo {modelo} não encontrado na API: {exc.message}") from exc
    except anthropic.BadRequestError as exc:
        raise ClaudeIndisponivel(f"Requisição recusada pela API: {exc.message}") from exc
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        # Já passou pelas novas tentativas do SDK: deixa só este comentário sem rótulo.
        print(f"!!! Claude ({modelo}) falhou num comentário: {exc}")
        return None

    if resp.stop_reason == "refusal":
        return None
    out_text = next((b.text for b in resp.content if b.type == "text"), "")
    return parse_label(out_text)


def classificar(textos: list[str], modelo: str) -> list[tuple[str | None, None]]:
    """Retorna (rótulo ou None, None) para cada texto, na mesma ordem."""
    if not textos:
        return []

    client = _get_client()
    with ThreadPoolExecutor(max_workers=CONCORRENCIA) as pool:
        futures = [pool.submit(_classificar_um, client, modelo, t) for t in textos]
        try:
            labels = [f.result() for f in futures]
        except ClaudeIndisponivel:
            for f in futures:
                f.cancel()
            raise
    return [(label, None) for label in labels]

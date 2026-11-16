"""Classificação de sentimento com LLMs Claude via API (zero-shot).

Cada comentário é classificado por uma chamada à API (1 por requisição), com execução
concorrente. Além da qualidade, registra latência (round-trip) e custo em US$
(tokens x preço).

Cada predição é gravada em output_dir/predictions/<modelo>.csv assim que chega: uma
rodada interrompida (ou o piloto com llm.limit_samples) é retomada sem pagar de novo
pelos exemplos já classificados. llm.max_cost_usd limita o gasto por modelo e
llm.dry_run só estima o custo via count_tokens (gratuito), sem classificar nada.

Requer a variável de ambiente ANTHROPIC_API_KEY (billing separado do Claude Pro).
"""
from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .config import Config, LLMModelConfig
from .data import RawData
from .metrics import compute_classification_metrics, save_per_class_report

# Reviews podem ser longas; truncar limita o custo de tokens de entrada por chamada.
_MAX_CHARS = 1500

# Saída típica: só o rótulo (~2-4 tokens). Usado apenas na estimativa do dry_run.
_EST_OUT_TOKENS = 4

_PRED_COLUMNS = [
    "id", "text", "label", "pred", "raw_output", "stop_reason",
    "input_tokens", "output_tokens", "latency_ms", "cost_usd",
]


def _system_prompt(raw: RawData, cfg: Config) -> str:
    labels = ", ".join(raw.label_names)
    lines = [
        "You are a precise text classification model.",
        f"Classify each input into EXACTLY one of these labels: {labels}.",
        "Respond with ONLY the label word, exactly as written, and nothing else.",
    ]
    if cfg.llm.num_fewshot > 0:
        k = min(cfg.llm.num_fewshot, len(raw.train))
        demos = raw.train.shuffle(seed=cfg.seed).select(range(k))
        lines.append("\nExamples:")
        for ex in demos:
            text = str(ex[cfg.dataset.text_column])[:_MAX_CHARS]
            label = raw.label_names[ex[cfg.dataset.label_column]]
            lines.append(f"Input: {text}\nLabel: {label}")
    return "\n".join(lines)


def _parse_label(out_text: str, label_names: list[str]) -> str | None:
    """Extrai o rótulo da resposta livre do modelo, tolerante a variações."""
    s = out_text.strip().strip('".\'').lower()
    lowered = {name.lower(): name for name in label_names}
    if s in lowered:                       # match exato
        return lowered[s]
    for low, name in lowered.items():      # rótulo aparece dentro da resposta
        if low in s:
            return name
    return None


def _client():
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY não definida. Configure a chave da API antes de usar 'llm'."
        )
    return anthropic.Anthropic()


def _eval_set(raw: RawData, cfg: Config):
    """Test set com um id estável por linha (+ subconjunto estratificado no piloto)."""
    test = raw.test
    id_col = cfg.dataset.id_column
    ids = [str(v) for v in test[id_col]] if id_col else [str(i) for i in range(len(test))]
    test = test.add_column("_llm_id", ids)
    k = cfg.llm.limit_samples
    if k and k < len(test):
        test = test.train_test_split(
            test_size=int(k), seed=cfg.seed, stratify_by_column=cfg.dataset.label_column
        )["test"]
    return test


def _request_options(model_cfg: LLMModelConfig) -> dict:
    """thinking/effort do modelo. Padrão: sem raciocínio — modelos que pensam por padrão
    (ex.: claude-sonnet-5) gastariam o max_tokens curto no thinking e voltariam sem rótulo."""
    opts = {}
    if model_cfg.thinking is not None:
        opts["thinking"] = model_cfg.thinking
    if model_cfg.effort is not None:
        opts["output_config"] = {"effort": model_cfg.effort}
    return opts


def _classify_one(client, model_cfg: LLMModelConfig, system: str, text: str, max_tokens: int):
    """Classifica um texto. Retorna (out_text, stop_reason, in_tokens, out_tokens, elapsed_s)."""
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model_cfg.id,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": text[:_MAX_CHARS]}],
        **_request_options(model_cfg),
    )
    elapsed = time.perf_counter() - t0
    out_text = next((b.text for b in resp.content if b.type == "text"), "")
    return out_text, resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens, elapsed


def estimate_llm_cost(model_cfg: LLMModelConfig, raw: RawData, cfg: Config) -> dict:
    """Estima o custo de rodar o modelo no test set via count_tokens (sem custo).

    Conta os tokens de entrada de uma amostra estratificada de até 100 exemplos e
    extrapola pela média. Também valida chave e id do modelo antes de gastar algo.
    """
    client = _client()
    system = _system_prompt(raw, cfg)
    test = _eval_set(raw, cfg)
    n = len(test)
    sample = test
    if n > 100:
        sample = test.train_test_split(
            test_size=100, seed=cfg.seed, stratify_by_column=cfg.dataset.label_column
        )["test"]
    texts = [str(t)[:_MAX_CHARS] for t in sample[cfg.dataset.text_column]]

    def _count(text: str) -> int:
        return client.messages.count_tokens(
            model=model_cfg.id, system=system, messages=[{"role": "user", "content": text}],
            **_request_options(model_cfg),
        ).input_tokens

    with ThreadPoolExecutor(max_workers=cfg.llm.concurrency) as pool:
        counts = list(pool.map(_count, texts))

    est_in = round(n * sum(counts) / len(counts))
    cost = (est_in * model_cfg.price_in + n * _EST_OUT_TOKENS * model_cfg.price_out) / 1e6
    max_tokens = model_cfg.max_tokens or cfg.llm.max_tokens
    worst = (est_in * model_cfg.price_in + n * max_tokens * model_cfg.price_out) / 1e6
    return {
        "model": model_cfg.id,
        "n_samples": n,
        "est_input_tokens": est_in,
        "est_cost_usd": round(cost, 4),
        "worst_cost_usd": round(worst, 4),
    }


def run_llm(model_cfg: LLMModelConfig, raw: RawData, cfg: Config) -> dict:
    """Avalia um LLM sobre o dataset. Retorna uma linha de resultados."""
    print(f"\n{'=' * 70}\n>>> LLM: {model_cfg.id}  [via API, zero/few-shot]\n{'=' * 70}")
    client = _client()
    system = _system_prompt(raw, cfg)
    label_names = raw.label_names
    label_to_id = {name: i for i, name in enumerate(label_names)}

    test = _eval_set(raw, cfg)
    ids = list(test["_llm_id"])
    texts = test[cfg.dataset.text_column]
    labels = list(test[cfg.dataset.label_column])

    slug = model_cfg.id.replace("/", "__")
    pred_path = Path(cfg.output_dir) / "predictions" / f"{slug}.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not pred_path.exists() or pred_path.stat().st_size == 0
    done = None if new_file else pd.read_csv(pred_path, dtype={"id": str})
    done_ids = set(done["id"]) if done is not None else set()
    spent = float(done["cost_usd"].sum()) if done is not None else 0.0

    todo = [i for i, rid in enumerate(ids) if rid not in done_ids]
    print(f"    {len(ids) - len(todo)} de {len(ids)} já classificados em {pred_path} "
          f"(US$ {spent:.4f}); {len(todo)} a classificar.")

    max_tokens = model_cfg.max_tokens or cfg.llm.max_tokens

    def _task(i: int):
        return i, _classify_one(client, model_cfg, system, str(texts[i]), max_tokens)

    errors = 0
    with open(pred_path, "a", newline="", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=cfg.llm.concurrency) as pool:
        writer = csv.DictWriter(fh, fieldnames=_PRED_COLUMNS)
        if new_file:
            writer.writeheader()
        futures = [pool.submit(_task, i) for i in todo]
        for fut in as_completed(futures):
            if fut.cancelled():
                continue
            try:
                i, (out_text, stop_reason, tin, tout, elapsed) = fut.result()
            except Exception as exc:  # não derruba o estudo; fica de fora e é refeito na retomada
                errors += 1
                if errors <= 3:
                    print(f"    ! falha numa amostra: {exc}")
                continue
            cost = (tin * model_cfg.price_in + tout * model_cfg.price_out) / 1e6
            spent += cost
            writer.writerow({
                "id": ids[i], "text": texts[i], "label": label_names[labels[i]],
                "pred": _parse_label(out_text, label_names) or "", "raw_output": out_text,
                "stop_reason": stop_reason, "input_tokens": tin, "output_tokens": tout,
                "latency_ms": round(1000 * elapsed, 2), "cost_usd": cost,
            })
            fh.flush()
            if cfg.llm.max_cost_usd is not None and spent >= cfg.llm.max_cost_usd:
                pending = sum(f.cancel() for f in futures)
                if pending:
                    print(f"    !!! teto de US$ {cfg.llm.max_cost_usd} atingido: "
                          f"{pending} amostras canceladas.")

    # Métricas sobre tudo o que já foi classificado neste eval set (inclui retomadas).
    preds = pd.read_csv(pred_path, dtype={"id": str}, keep_default_na=False)
    preds = preds[preds["id"].isin(set(ids))].drop_duplicates("id", keep="last")
    valid = preds[preds["pred"].isin(label_to_id)]
    if valid.empty:
        raise RuntimeError(f"{model_cfg.id}: nenhuma predição válida (erros={errors}).")
    missing = len(ids) - len(preds)
    if len(valid) < len(ids):
        print(f"    aviso: {len(ids) - len(valid)} de {len(ids)} amostras sem predição válida "
              f"({len(preds) - len(valid)} rótulos inválidos, {missing} sem resposta; ignoradas).")
    y_pred = valid["pred"].map(label_to_id).tolist()
    y_true = valid["label"].map(label_to_id).tolist()

    # Métricas só sobre as respostas válidas (as inválidas ficam contadas à parte).
    macro = compute_classification_metrics(y_pred, y_true, "macro")
    weighted = compute_classification_metrics(y_pred, y_true, "weighted")
    if cfg.per_class_report:
        save_per_class_report(y_pred, y_true, label_names, cfg.reports_dir, slug)
    in_tokens = int(preds["input_tokens"].sum())
    out_tokens = int(preds["output_tokens"].sum())
    cost = (in_tokens * model_cfg.price_in + out_tokens * model_cfg.price_out) / 1e6

    row = {
        "model": model_cfg.id,
        "n_comentarios": len(ids),
        "n_validos": len(valid),
        "n_invalidos": len(ids) - len(valid),
        "accuracy": round(macro["accuracy"], 4),
        "f1_macro": round(macro["f1"], 4),
        "precision_macro": round(macro["precision"], 4),
        "recall_macro": round(macro["recall"], 4),
        "f1_weighted": round(weighted["f1"], 4),
        "precision_weighted": round(weighted["precision"], 4),
        "recall_weighted": round(weighted["recall"], 4),
        "latency_ms": round(float(preds["latency_ms"].mean()), 2),
        "cost_usd": round(cost, 4),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
    }
    print(f">>> {model_cfg.id} concluído: {row}")
    return row

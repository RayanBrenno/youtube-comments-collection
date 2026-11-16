"""Fine-tuning + avaliação de UM modelo, devolvendo um dict achatado de métricas."""
from __future__ import annotations

import gc
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import Config
from .data import RawData, tokenize_dataset
from .metrics import (
    build_compute_metrics,
    measure_latency,
    model_cost_metrics,
    save_per_class_report,
)


def _safe_name(model_name: str) -> str:
    """Converte 'org/modelo' num nome de pasta válido (ex.: org__modelo).

    Modelo local (pasta já fine-tuned, ex.: modelos/etapa1_cardiff/xlm-roberta-base) -> só o
    nome da pasta, para o 2º fine-tuning salvar em models_dir/xlm-roberta-base."""
    if Path(model_name).is_dir():
        return Path(model_name).name
    return model_name.replace("/", "__")


def _resolve_precision(precision: str) -> tuple[bool, bool]:
    """Decide (bf16, fp16) a partir do config e do hardware disponível.

    bf16 é preferido a fp16 em GPUs Ampere+ (compute capability >= 8.0): mesma
    economia de memória/velocidade, mas com o range de expoente do fp32, o que
    elimina o overflow de loss que o fp16 sofre em alguns modelos (o xlm-roberta
    da rodada anterior estourou a loss em 10378 — sintoma clássico disso).
    """
    if not torch.cuda.is_available() or precision == "fp32":
        return False, False
    if precision == "bf16":
        return True, False
    if precision == "fp16":
        return False, True
    # auto
    if torch.cuda.is_bf16_supported():
        return True, False
    return False, True


def _is_oom(exc: Exception) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_args(cfg: Config, tmp_dir: str, batch_size: int, grad_accum: int, has_val: bool):
    t = cfg.training
    bf16, fp16 = _resolve_precision(t.precision)

    # Sem split de validação não há como selecionar checkpoint sem olhar o test set,
    # então o early stopping / load_best_model_at_end ficam desligados.
    use_best = has_val and t.early_stopping_patience > 0 and not cfg.eval_only
    if cfg.eval_only:
        eval_strategy = "no"
    elif has_val:
        eval_strategy = t.eval_strategy
    else:
        eval_strategy = "no"

    kwargs = dict(
        output_dir=tmp_dir,
        eval_strategy=eval_strategy,
        save_strategy=eval_strategy if use_best else "no",
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        label_smoothing_factor=t.label_smoothing_factor,
        max_grad_norm=t.max_grad_norm,
        optim=t.optim,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=t.gradient_checkpointing,
        group_by_length=t.group_by_length,
        dataloader_num_workers=t.dataloader_num_workers,
        logging_steps=50,
        report_to="none",
        seed=cfg.seed,
        disable_tqdm=False,
    )
    if eval_strategy == "steps":
        kwargs["eval_steps"] = t.eval_steps
        if use_best:
            kwargs["save_steps"] = t.eval_steps
    if use_best:
        kwargs.update(
            load_best_model_at_end=True,
            metric_for_best_model=t.metric_for_best_model,
            greater_is_better=True,
            save_total_limit=1,
        )
    return TrainingArguments(**kwargs), use_best


def _train_and_eval(
    model_name: str, raw: RawData, cfg: Config, tmp_dir: str,
    batch_size: int, grad_accum: int,
) -> dict:
    """Uma tentativa completa de treino+avaliação com um dado batch size."""
    set_seed(cfg.seed)  # reprodutibilidade por modelo (e por tentativa)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, val_ds, test_ds = tokenize_dataset(raw, tokenizer, cfg)

    # Em eval_only, respeita o num_labels do modelo já treinado (não força o do dataset).
    model_kwargs = {} if cfg.eval_only else {"num_labels": raw.num_labels}
    if not cfg.eval_only:
        # Nomes legíveis gravados no config.json do modelo salvo — assim o modelo
        # exportado devolve "positivo" em vez de "LABEL_2" num pipeline().
        model_kwargs["id2label"] = dict(enumerate(raw.label_names))
        model_kwargs["label2id"] = {n: i for i, n in enumerate(raw.label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)

    args, use_best = _build_args(cfg, tmp_dir, batch_size, grad_accum, val_ds is not None)
    callbacks = (
        [EarlyStoppingCallback(early_stopping_patience=cfg.training.early_stopping_patience)]
        if use_best else []
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # seleção de modelo olha o VAL, nunca o test
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_compute_metrics(cfg.metric_average),
        callbacks=callbacks,
    )

    if cfg.eval_only:
        train_time_s = 0.0
    else:
        eff = batch_size * grad_accum
        print(
            f">>> batch={batch_size} x grad_accum={grad_accum} (efetivo={eff}) | "
            f"bf16={args.bf16} fp16={args.fp16} | early_stopping={'on' if use_best else 'off'}"
        )
        t0 = time.perf_counter()
        trainer.train()
        train_time_s = round(time.perf_counter() - t0, 1)

    # --- avaliação FINAL no test set (intocado até aqui) -----------------------
    pred = trainer.predict(test_ds, metric_key_prefix="test")
    eval_metrics = pred.metrics

    if cfg.per_class_report:
        save_per_class_report(
            np.argmax(pred.predictions, axis=-1),
            pred.label_ids,
            raw.label_names,
            cfg.reports_dir,
            _safe_name(model_name),
        )

    # Métricas no val, só para registrar o gap val/test (sinal de overfitting).
    val_f1 = float("nan")
    if val_ds is not None and not cfg.eval_only:
        val_f1 = round(trainer.evaluate(val_ds).get("eval_f1", float("nan")), 4)

    # Define onde salvar o modelo: pasta persistente (reuso/exportação) ou temporária
    # (só para medir o tamanho em disco). Persistir só faz sentido quando houve treino.
    if cfg.save_models and not cfg.eval_only:
        saved_dir = Path(cfg.models_dir) / _safe_name(model_name)
    else:
        saved_dir = Path(tmp_dir) / "saved"

    # Salva modelo + tokenizer juntos, para recarregar depois com from_pretrained.
    trainer.save_model(str(saved_dir))
    tokenizer.save_pretrained(str(saved_dir))
    cost = model_cost_metrics(model, saved_dir)
    if cfg.save_models and not cfg.eval_only:
        print(f">>> Modelo treinado salvo em: {saved_dir}")

    # Latência medida na CPU/GPU com o modelo já em eval: cenário de inferência online.
    latency_ms = measure_latency(model, test_ds, cfg.latency_samples)

    row = {
        "model": model_name,
        "accuracy": round(eval_metrics.get("test_accuracy", float("nan")), 4),
        "f1": round(eval_metrics.get("test_f1", float("nan")), 4),
        "precision": round(eval_metrics.get("test_precision", float("nan")), 4),
        "recall": round(eval_metrics.get("test_recall", float("nan")), 4),
        "eval_loss": round(eval_metrics.get("test_loss", float("nan")), 4),
        "val_f1": val_f1,
        "train_time_s": train_time_s,
        "latency_ms": latency_ms,
        "num_params_M": cost["num_params_M"],
        "size_mb": cost["size_mb"],
        "batch_size": batch_size,
        "grad_accum": grad_accum,
    }

    del trainer, model
    _free_gpu()
    return row


def run_one(model_name: str, raw: RawData, cfg: Config) -> dict:
    """Treina (ou não), avalia e mede custo de um modelo. Retorna uma linha de resultados.

    Em `cfg.eval_only=True`, o treino é pulado: o modelo (que deve já estar fine-tuned
    para a tarefa) é apenas avaliado. `train_time_s` fica 0.0 nesse caso.

    Se a GPU estourar a memória, reduz o batch pela metade e dobra o
    gradient_accumulation_steps — o batch EFETIVO (e portanto a dinâmica de treino,
    que é o que precisa ser idêntico entre modelos) fica preservado.
    """
    mode = "eval-only (sem treino)" if cfg.eval_only else "fine-tuning + avaliação"
    print(f"\n{'=' * 70}\n>>> Modelo: {model_name}  [{mode}]\n{'=' * 70}")

    batch_size = cfg.training.per_device_train_batch_size
    grad_accum = cfg.training.gradient_accumulation_steps

    while True:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                return _train_and_eval(model_name, raw, cfg, tmp_dir, batch_size, grad_accum)
        except Exception as exc:
            _free_gpu()
            can_retry = (
                cfg.training.oom_retry
                and _is_oom(exc)
                and batch_size // 2 >= cfg.training.min_batch_size
            )
            if not can_retry:
                raise
            batch_size //= 2
            grad_accum *= 2
            print(
                f"!!! OOM na GPU. Repetindo com batch={batch_size}, "
                f"grad_accum={grad_accum} (batch efetivo inalterado)."
            )

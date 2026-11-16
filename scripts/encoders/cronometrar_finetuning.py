"""Refaz o fine-tuning da Etapa 1 (Cardiff) e da Etapa 2, cronometrando o tempo TOTAL
de cada modelo (do lançamento do comando até ele terminar), em cronometragem/.

Cada modelo roda num processo separado (`python run.py --config <yaml de 1 modelo>`),
então o tempo inclui iniciar o Python, imports, carregar dados e modelo, treinar,
avaliar no teste, medir latência e salvar o modelo.
A receita de treino é a mesma dos configs de configs/; só mudam as pastas de saída.
A Etapa 2 parte dos modelos da Etapa 1 desta mesma execução (cronometragem/etapa1_cardiff/).

Saída: cronometragem/tempos_finetuning.csv (gravado a cada modelo) + modelos, logs e
resultados em cronometragem/. Não altera modelos/, resultados/ nem configs/.

Uso: python cronometrar_finetuning.py
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
OUT = Path("cronometragem")
ETAPAS = [
    ("etapa1_cardiff", "configs/etapa1_cardiff.yaml"),
    ("etapa2", "configs/etapa2.yaml"),
]
COLS = ["etapa", "modelo", "inicio", "fim", "tempo_total_s", "tempo_laco_treino_s",
        "tempo_acumulado_s", "returncode"]


def nome_pasta(model: str) -> str:
    """Mesma regra do src/train.py::_safe_name."""
    return Path(model).name if Path(model).is_dir() else model.replace("/", "__")


def salvar(linhas: list[dict]) -> None:
    with open(OUT / "tempos_finetuning.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(linhas)


def main() -> None:
    os.chdir(ROOT)
    if OUT.exists():
        sys.exit(f"{OUT}/ já existe; não vou sobrescrever. Apague ou renomeie antes.")

    gpu = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.get_device_name(0))"],
                         capture_output=True, text=True)
    if gpu.returncode != 0:
        sys.exit(f"GPU CUDA indisponível; o tempo não seria comparável.\n{gpu.stderr}")
    print(f"GPU: {gpu.stdout.strip()} | Python: {sys.executable}", flush=True)

    (OUT / "configs").mkdir(parents=True)
    linhas: list[dict] = []
    tempo_etapa1: dict[str, float] = {}

    for etapa, cfg_path in ETAPAS:
        base = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        for model in base["models"]:
            if etapa != "etapa1_cardiff":  # Etapa 2 parte da Etapa 1 desta execução
                model = model.replace("modelos/etapa1_cardiff/", f"{OUT}/etapa1_cardiff/")
            nome = Path(model).name if model.startswith(str(OUT)) else nome_pasta(model)
            out_dir = OUT / "resultados" / etapa / nome
            out_dir.mkdir(parents=True)

            cfg = dict(base, models=[model], models_dir=str(OUT / etapa), output_dir=str(out_dir))
            cfg_file = OUT / "configs" / f"{etapa}__{nome}.yaml"
            with open(cfg_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

            print(f">>> [{etapa}] {nome} ... ", end="", flush=True)
            inicio = datetime.now()
            t0 = time.perf_counter()
            with open(out_dir / "run.log", "w", encoding="utf-8") as log:
                rc = subprocess.run([sys.executable, "run.py", "--config", str(cfg_file)],
                                    stdout=log, stderr=subprocess.STDOUT).returncode
            total = time.perf_counter() - t0
            fim = datetime.now()

            res_path = out_dir / "results.csv"
            res = next(csv.DictReader(open(res_path, encoding="utf-8"))) if res_path.exists() else {}
            if etapa == "etapa1_cardiff":
                tempo_etapa1[nome] = total
            linha = {
                "etapa": etapa, "modelo": nome,
                "inicio": inicio.isoformat(timespec="seconds"), "fim": fim.isoformat(timespec="seconds"),
                "tempo_total_s": round(total, 1),
                "tempo_laco_treino_s": res.get("train_time_s", ""),
                # Etapa 1: igual ao total. Etapa 2: total da Etapa 1 + total da Etapa 2 do
                # mesmo modelo (tempo para chegar ao modelo final a partir do modelo base).
                "tempo_acumulado_s": round(total if etapa == "etapa1_cardiff"
                                           else tempo_etapa1[nome] + total, 1),
                "returncode": rc,
            }
            linhas.append(linha)
            salvar(linhas)
            print(f"total {total:.1f} s | laço de treino {linha['tempo_laco_treino_s']} s | "
                  f"rc {rc}", flush=True)
            if rc != 0 or not res:
                sys.exit(f"Falhou em {etapa}/{nome}; veja {out_dir / 'run.log'}")

    print(f"\nPronto. Tempos em {OUT / 'tempos_finetuning.csv'}", flush=True)


if __name__ == "__main__":
    main()

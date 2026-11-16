"""Valida config + carregamento do dataset SEM treinar nada (roda em segundos, sem GPU).

Uso: python validate_setup.py --config configs/etapa1_cardiff.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter

from src.config import load_config
from src.data import load_raw_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[ok] config valido: {args.config}")
    print(f"     modelos: {len(cfg.models)}")
    for m in cfg.models:
        print(f"       - {m}")
    print(f"     output_dir: {cfg.output_dir}")

    raw = load_raw_data(cfg)
    lc = cfg.dataset.label_column
    names = raw.label_names

    def dist(ds):
        c = Counter(ds[lc])
        return {names[k]: v for k, v in sorted(c.items())}

    print(f"\n[ok] splits")
    print(f"     treino    : {len(raw.train):>7,}  {dist(raw.train)}")
    if raw.val is not None:
        print(f"     validacao : {len(raw.val):>7,}  {dist(raw.val)}")
    print(f"     teste     : {len(raw.test):>7,}  {dist(raw.test)}")
    total = len(raw.train) + len(raw.test) + (len(raw.val) if raw.val is not None else 0)
    print(f"     TOTAL     : {total:>7,}")
    print(f"     classes   : {raw.num_labels} -> {names}")

    # Nenhum texto pode aparecer em mais de um split (vazamento entre treino/teste).
    tc = cfg.dataset.text_column
    s_tr, s_te = set(raw.train[tc]), set(raw.test[tc])
    overlap = len(s_tr & s_te)
    print(f"\n[{'ok' if overlap == 0 else 'ATENCAO'}] sobreposicao treino/teste: {overlap} textos")
    if raw.val is not None:
        ov2 = len(set(raw.val[tc]) & s_te)
        print(f"[{'ok' if ov2 == 0 else 'ATENCAO'}] sobreposicao validacao/teste: {ov2} textos")

    print("\n--- amostras do treino ---")
    for i in range(3):
        ex = raw.train[i]
        print(f"  [{names[ex[lc]]}] {str(ex[tc])[:110]}")

    print("\n[ok] tudo pronto. Rode:  python run.py --config " + args.config)


if __name__ == "__main__":
    main()

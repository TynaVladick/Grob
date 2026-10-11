#!/usr/bin/env python3
"""Шаг 6. Сравнение результата GROBID с ручным эталоном (только gold/*.json с verified=true).

Поля и метрики
  * Точное сопоставление (доля правильных): название (после нормализации), DOI, год.
  * Обнаружение сущностей (precision / recall / F1): авторы, заголовки разделов, ссылки.
  * Аннотация: среднее сходство текстов и token-F1 (точное равенство длинного текста
    не имеет смысла), плюс доля статей со сходством >= --abstract-threshold.

Если статья не обработана GROBID (ошибка/нет TEI), её предсказание считается пустым:
все поля в ней — промахи. Так итог отражает пайплайн целиком; число таких статей выводится отдельно.
Флаг --skip-failed исключает их из оценки (тогда оценивается только качество разбора).

Для сущностей даны micro-значения (суммируем TP/FP/FN по всем статьям) и macro (среднее F1 по статьям).

    python 06_evaluate.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from datetime import datetime, timezone

from common import (EVALUATION_CSV, EVALUATION_JSON, EXTRACTED_DIR, GOLD_DIR, ensure_dirs,
                    setup_logging)
from metrics import evaluate_article, prf

log = logging.getLogger("eval")
ENTITIES = ["authors", "headings", "references"]


def mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else None


def accuracy(values: list) -> dict:
    vals = [v for v in values if v is not None]
    n_ok = sum(1 for v in vals if v)
    return {"correct": n_ok, "total": len(vals), "accuracy": (n_ok / len(vals)) if vals else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-threshold", type=float, default=0.85,
                    help="порог сходства для сопоставления ссылки с эталонной")
    ap.add_argument("--abstract-threshold", type=float, default=0.90)
    ap.add_argument("--skip-failed", action="store_true", help="не учитывать статьи без результата GROBID")
    args = ap.parse_args()
    setup_logging()
    ensure_dirs()

    gold_files = sorted(GOLD_DIR.glob("*.json"))
    golds, unverified = [], []
    for f in gold_files:
        g = json.loads(f.read_text(encoding="utf-8"))
        (golds if g.get("verified") is True else unverified).append((f.stem, g))
    if unverified:
        log.warning("Не сверено вручную (пропущено): %d — %s", len(unverified),
                    ", ".join(a for a, _ in unverified))
    if not golds:
        raise SystemExit("Нет ни одного gold/*.json с \"verified\": true. Сверьте черновики с PDF (см. README).")
    if len(golds) < 10:
        log.warning("В эталоне только %d статей (план — 10–15): оценка будет шумной.", len(golds))

    per_article, n_failed = [], 0
    for aid, gold in golds:
        pred_path = EXTRACTED_DIR / f"{aid}.json"
        pred = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists() else None
        if pred is None:
            n_failed += 1
            if args.skip_failed:
                log.warning("%s: нет результата GROBID — исключена (--skip-failed)", aid)
                continue
            log.warning("%s: нет результата GROBID — считается пустым", aid)
        r = evaluate_article(gold, pred, args.ref_threshold)
        r["article_id"] = aid
        r["has_prediction"] = pred is not None
        per_article.append(r)

    n = len(per_article)
    summary: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_gold_verified": len(golds), "n_evaluated": n, "n_without_grobid_result": n_failed,
        "skip_failed": args.skip_failed,
        "settings": {"ref_threshold": args.ref_threshold, "abstract_threshold": args.abstract_threshold},
        "exact_match": {
            "title": accuracy([r["title_exact"] for r in per_article]),
            "doi": accuracy([r["doi_exact"] for r in per_article]),
            "year": accuracy([r["year_exact"] for r in per_article]),
        },
        "title_similarity_mean": mean(r["title_sim"] for r in per_article),
        "abstract": {
            "similarity_mean": mean(r["abstract_sim"] for r in per_article),
            "token_f1_mean": mean(r["abstract_token_f1"] for r in per_article),
            "share_ge_threshold": mean(r["abstract_sim"] >= args.abstract_threshold for r in per_article),
        },
        "entities": {},
    }
    for ent in ENTITIES:
        tp = sum(r[ent][0] for r in per_article)
        fp = sum(r[ent][1] for r in per_article)
        fn = sum(r[ent][2] for r in per_article)
        micro = prf(tp, fp, fn)
        macro_f1 = mean(prf(*r[ent])["f1"] for r in per_article)
        summary["entities"][ent] = {"micro": micro, "macro_f1": macro_f1}

    EVALUATION_JSON.write_text(json.dumps({"summary": summary, "per_article": per_article},
                                          ensure_ascii=False, indent=2, default=list), encoding="utf-8")

    cols = ["article_id", "has_prediction", "title_exact", "title_sim", "doi_exact", "year_exact",
            "abstract_sim", "abstract_token_f1"]
    for ent in ENTITIES:
        cols += [f"{ent}_precision", f"{ent}_recall", f"{ent}_f1", f"{ent}_gold", f"{ent}_pred"]
    key_counts = {"authors": ("n_gold_authors", "n_pred_authors"),
                  "headings": ("n_gold_headings", "n_pred_headings"),
                  "references": ("n_gold_refs", "n_pred_refs")}
    with EVALUATION_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in per_article:
            row = {c: r.get(c) for c in cols[:8]}
            for ent in ENTITIES:
                m = prf(*r[ent])
                row.update({f"{ent}_precision": round(m["precision"], 3), f"{ent}_recall": round(m["recall"], 3),
                            f"{ent}_f1": round(m["f1"], 3), f"{ent}_gold": r[key_counts[ent][0]],
                            f"{ent}_pred": r[key_counts[ent][1]]})
            w.writerow(row)

    # --- печать ---
    def pct(x):
        return "  —  " if x is None else f"{100 * x:5.1f}%"

    print(f"\nОценено статей: {n} (эталон сверен: {len(golds)}, без результата GROBID: {n_failed})\n")
    print("Точное сопоставление            верно / всего   доля")
    for name, key in (("Название (норм.)", "title"), ("DOI", "doi"), ("Год", "year")):
        a = summary["exact_match"][key]
        print(f"  {name:<28} {a['correct']:>4} / {a['total']:<4}   {pct(a['accuracy'])}")
    print(f"  Сходство названий (среднее)                {pct(summary['title_similarity_mean'])}")
    ab = summary["abstract"]
    print(f"\nАннотация: сходство {pct(ab['similarity_mean'])} | token-F1 {pct(ab['token_f1_mean'])} | "
          f"доля со сходством ≥ {args.abstract_threshold:.2f}: {pct(ab['share_ge_threshold'])}")
    print("\nСущности              Precision  Recall     F1    macro-F1   TP/FP/FN")
    for ent, label in (("authors", "Авторы"), ("headings", "Заголовки разделов"), ("references", "Ссылки")):
        e = summary["entities"][ent]
        m = e["micro"]
        print(f"  {label:<19} {pct(m['precision'])}  {pct(m['recall'])}  {pct(m['f1'])}   {pct(e['macro_f1'])}"
              f"   {m['tp']}/{m['fp']}/{m['fn']}")
    print(f"\nПодробно: {EVALUATION_JSON}\nПо статьям: {EVALUATION_CSV}")


if __name__ == "__main__":
    main()
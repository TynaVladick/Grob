#!/usr/bin/env python3
"""Шаг 7. Базовая линия по времени и ошибкам (из results/processing_log.csv и run_meta.json).

Выводит: общее время прогона, медианное время на статью, число успешных PDF и число ошибок
(с разбивкой по типам). Это БАЗОВАЯ ЛИНИЯ для одной конфигурации «1 клиент, 1 экземпляр
GROBID, последовательно» — а не доказательство ускорения от распределения.

Первый запрос обычно медленнее (прогрев JVM/моделей), поэтому медиана и среднее дополнительно
даны без первого документа.

    python 07_timing_report.py
"""
from __future__ import annotations

import json
import statistics
from collections import Counter

from common import (DIAGNOSTICS_CSV, PROCESSING_LOG, RUN_META, TIMING_JSON, ensure_dirs, read_csv)


def percentile(sorted_vals: list[float], q: float) -> float:
    """Линейная интерполяция между соседними значениями (q в 0..1)."""
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def stats_of(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return {"n": len(s), "sum": sum(s), "mean": statistics.fmean(s), "median": statistics.median(s),
            "p90": percentile(s, 0.90), "p95": percentile(s, 0.95), "min": s[0], "max": s[-1]}


def main() -> None:
    ensure_dirs()
    rows = read_csv(PROCESSING_LOG)
    if not rows:
        raise SystemExit(f"Нет {PROCESSING_LOG}. Сначала запустите 03_run_grobid.py")
    meta = json.loads(RUN_META.read_text(encoding="utf-8")) if RUN_META.exists() else {}

    ok = [r for r in rows if r["status"] == "ok"]
    errors = [r for r in rows if r["status"] != "ok"]
    t_ok = [float(r["seconds"]) for r in ok]
    t_all = [float(r["seconds"]) for r in rows]
    first = rows[0]

    summary = {
        "run_id": meta.get("run_id"), "label": meta.get("label"),
        "grobid_version": meta.get("grobid_version"), "grobid_params": meta.get("params"),
        "mode": meta.get("mode"), "client_machine": meta.get("client_machine"),
        "n_pdfs": len(rows), "n_ok": len(ok), "n_errors": len(errors),
        "success_rate": len(ok) / len(rows),
        "errors_by_status": dict(Counter(r["status"] for r in errors)),
        "failed_articles": [{"article_id": r["article_id"], "status": r["status"], "error": r["error"]}
                            for r in errors],
        "batch_wall_seconds": meta.get("batch_wall_seconds"),
        "request_seconds_ok": stats_of(t_ok),
        "request_seconds_all": stats_of(t_all),
        "request_seconds_ok_without_first": stats_of(
            [float(r["seconds"]) for r in ok if r["article_id"] != first["article_id"]]),
        "first_document": {"article_id": first["article_id"], "seconds": float(first["seconds"])},
        "note": "Базовая линия одной конфигурации; не сравнение распределённой и обычной обработки.",
    }

    # секунд на страницу (если есть диагностика PyMuPDF)
    diag = {r["article_id"]: r for r in read_csv(DIAGNOSTICS_CSV) if r.get("status") == "ok"}
    pages = [(int(diag[r["article_id"]]["pages"]), float(r["seconds"])) for r in ok if r["article_id"] in diag]
    if pages:
        total_pages = sum(p for p, _ in pages)
        summary["pages_total_ok"] = total_pages
        summary["seconds_per_page_overall"] = sum(s for _, s in pages) / total_pages
        summary["pymupdf_text_extraction_seconds"] = stats_of(
            [float(d["pymupdf_seconds"]) for d in diag.values() if d.get("pymupdf_seconds")])

    TIMING_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- печать ---
    print(f"\nПрогон {meta.get('run_id', '?')}  |  GROBID {meta.get('grobid_version', '?')}  |  {meta.get('label', '')}")
    print(f"PDF всего: {len(rows)}   успешно: {len(ok)}   ошибок: {len(errors)}   "
          f"доля успеха: {100 * len(ok) / len(rows):.1f}%")
    if errors:
        print("Ошибки по типам:", dict(Counter(r["status"] for r in errors)))
        for r in errors:
            print(f"  - {r['article_id']}: {r['status']} {r['error'][:100]}")
    if meta.get("batch_wall_seconds") is not None:
        print(f"Общее время прогона (wall-clock): {meta['batch_wall_seconds']:.1f} с")
    s = summary["request_seconds_ok"]
    if s:
        print(f"Время на статью (успешные), с:  медиана {s['median']:.2f}   среднее {s['mean']:.2f}   "
              f"p90 {s['p90']:.2f}   p95 {s['p95']:.2f}   мин {s['min']:.2f}   макс {s['max']:.2f}")
    w = summary["request_seconds_ok_without_first"]
    if w:
        print(f"  без первого документа (прогрев): медиана {w['median']:.2f}   среднее {w['mean']:.2f}"
              f"   (первый: {float(first['seconds']):.2f})")
    if "seconds_per_page_overall" in summary:
        print(f"Секунд на страницу: {summary['seconds_per_page_overall']:.2f} "
              f"({summary['pages_total_ok']} страниц)")
    print(f"\nПодробно: {TIMING_JSON}")


if __name__ == "__main__":
    main()
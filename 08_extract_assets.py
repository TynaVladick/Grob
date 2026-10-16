#!/usr/bin/env python3
"""Шаг 8 (дополнительный). Вшивание рисунков, графиков и таблиц в текст статей для RAG.

Читает:   data/pdf/<id>.pdf, data/tei/<id>.tei.xml (если есть), data/tei_coords/<id>.tei.xml (если есть)
Пишет:    data/rag/<id>.md              статья с блоками рисунков/таблиц на своих местах
          data/rag/<id>.chunks.jsonl    чанки для индексации (текст + по чанку на рисунок/таблицу)
          data/assets/<id>/             PNG-вырезки, CSV таблиц, assets.json
          results/assets_summary.csv    сколько найдено (+ сверка с числом <fig>/<table-wrap> в JATS)

Примеры:
  # 1. Без внешних моделей: подписи + таблицы в Markdown + PNG-вырезки (лучше после --regrobid)
  python 08_extract_assets.py --regrobid
  # 2. Docker/GROBID не работает: только PyMuPDF
  python 08_extract_assets.py --layout pymupdf
  # 3. С описанием рисунков моделью зрения (Anthropic API)
  export ANTHROPIC_API_KEY=...
  python 08_extract_assets.py --regrobid --vlm anthropic --vlm-model claude-sonnet-5-5 --vlm-tables
  # 4. Локальная VLM через OpenAI-совместимый сервер (vLLM, Ollama, LM Studio)
  python 08_extract_assets.py --vlm openai --vlm-base-url http://localhost:11434/v1 --vlm-model <имя-модели>

Шаг не входит в baseline-замеры GROBID: он читает результаты шагов 1-3 и ничего в них не меняет.
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import multimodal as mm
from common import ensure_dirs, write_csv

log = logging.getLogger("assets")

FIELDS = ["article_id", "layout", "n_figures", "n_tables", "figures_with_image", "tables_with_structure",
          "vlm_calls", "n_chunks", "jats_fig", "jats_table_wrap", "status", "error", "seconds"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*", help="обработать только эти article_id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="пересчитать, даже если результат уже есть")
    ap.add_argument("--layout", choices=["auto", "grobid", "pymupdf"], default="auto",
                    help="auto: TEI с координатами -> TEI+эвристика -> только PyMuPDF; "
                         "grobid: требовать TEI; pymupdf: игнорировать GROBID (если Docker недоступен)")
    ap.add_argument("--regrobid", action="store_true",
                    help="получить TEI с координатами рисунков (teiCoordinates=figure,ref) в data/tei_coords/")
    ap.add_argument("--grobid-url", default=os.environ.get("GROBID_URL", "http://localhost:8070"))
    ap.add_argument("--no-refine", action="store_true", help="не расширять бокс GROBID до касающейся графики")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-px", type=int, default=2000, help="максимум пикселей по длинной стороне вырезки")
    ap.add_argument("--vlm", choices=["none", "anthropic", "openai"], default="none")
    ap.add_argument("--vlm-model", default=os.environ.get("VLM_MODEL", ""))
    ap.add_argument("--vlm-base-url", default=os.environ.get("VLM_BASE_URL"))
    ap.add_argument("--vlm-tables", action="store_true",
                    help="если таблицу не удалось разобрать из PDF — расшифровать вырезку моделью зрения")
    ap.add_argument("--lang", default="en", help="язык описаний и служебных подписей (en, ru, ...)")
    ap.add_argument("--max-chars", type=int, default=1500, help="целевой размер текстового чанка, символов")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    mm.setup_logging(args.verbose)
    ensure_dirs()
    mm._require_pymupdf()

    vlm = None
    if args.vlm != "none":
        if not args.vlm_model:
            raise SystemExit("Укажите --vlm-model (или переменную VLM_MODEL)")
        vlm = mm.VLM(args.vlm, args.vlm_model, base_url=args.vlm_base_url, lang=args.lang,
                     cache_path=mm.ASSETS_DIR / "vlm_cache.jsonl")
    elif args.vlm_tables:
        raise SystemExit("--vlm-tables требует --vlm")

    ids = args.ids or mm.pdf_article_ids()
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        raise SystemExit(f"В {mm.PDF_DIR} нет PDF. Сначала запустите 01_select_and_download.py")

    rows = []
    for i, aid in enumerate(ids, 1):
        t0 = time.perf_counter()
        if not args.force and (mm.RAG_DIR / f"{aid}.chunks.jsonl").exists():
            log.info("[%2d/%d] %-16s уже обработана (--force, чтобы пересчитать)", i, len(ids), aid)
            continue
        try:
            row = mm.process_article(aid, layout=args.layout, regrobid=args.regrobid, grobid_url=args.grobid_url,
                                     refine=not args.no_refine, dpi=args.dpi, max_px=args.max_px, vlm=vlm,
                                     vlm_tables=args.vlm_tables, lang=args.lang, max_chars=args.max_chars)
        except Exception as exc:  # noqa: BLE001  одна плохая статья не должна остановить прогон
            log.exception("%s: ошибка", aid)
            row = {"article_id": aid, "status": "error", "error": str(exc)[:300]}
        row["seconds"] = round(time.perf_counter() - t0, 2)
        rows.append(row)
        log.info("[%2d/%d] %-16s %-26s рис=%s (JATS %s) табл=%s (JATS %s) VLM=%s", i, len(ids), aid,
                 row.get("layout", row["status"]), row.get("n_figures", "-"), row.get("jats_fig", "-"),
                 row.get("n_tables", "-"), row.get("jats_table_wrap", "-"), row.get("vlm_calls", 0))

    if rows:
        write_csv(mm.SUMMARY_CSV, rows, FIELDS)
        log.info("Сводка: %s", mm.SUMMARY_CSV)


if __name__ == "__main__":
    main()

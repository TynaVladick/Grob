#!/usr/bin/env python3
"""Шаг 2 (диагностика). PyMuPDF: что внутри каждого PDF и сколько занимает простое извлечение текста.

Для каждого PDF считает:
  * число страниц, размер файла;
  * сколько символов текстового слоя на странице;
  * сколько страниц «только картинка» (текста почти нет, а изображения есть) — признак скана;
  * эвристику «две колонки» (по расположению текстовых блоков) — где порядок чтения может ломаться;
  * время извлечения текста PyMuPDF (простая базовая линия для сравнения с GROBID).

Базовый текст сохраняется в data/pymupdf_text/<id>.txt (страницы разделены символом \\f).

OCR здесь НЕ выполняется. GROBID сам OCR тоже не делает: если для каких-то PDF стоит флаг
scanned_suspect=True, их нужно либо исключить из теста, либо отдельно прогнать через OCR
(в PyMuPDF для этого есть page.get_textpage_ocr(), нужен установленный Tesseract).

    python 02_pymupdf_diagnostics.py
"""
from __future__ import annotations

import argparse
import logging
import statistics
import time

try:
    import pymupdf  # PyMuPDF >= 1.24.3
except ImportError:  # старые версии
    import fitz as pymupdf

from common import (DIAGNOSTICS_CSV, PDF_DIR, TEXT_DIR, ensure_dirs, pdf_article_ids,
                    setup_logging, write_csv)

log = logging.getLogger("pymupdf")

MIN_TEXT_CHARS = 100      # меньше — считаем, что на странице нет текстового слоя
MIN_BLOCK_CHARS = 40      # блоки короче не учитываем при оценке колонок

FIELDS = ["article_id", "status", "error", "file_kb", "pages", "text_chars_total",
          "chars_per_page_median", "image_only_pages", "pages_with_images",
          "two_column_pages", "scanned_suspect", "encrypted", "pdf_title_meta",
          "pymupdf_seconds"]


def two_column_page(page) -> bool:
    """Эвристика: заметная доля длинных блоков целиком в левой половине И в правой половине."""
    w = page.rect.width
    left = right = total = 0
    for b in page.get_text("blocks"):
        x0, _y0, x1, _y1, text, _no, btype = b[:7]
        if btype != 0 or len(text.strip()) < MIN_BLOCK_CHARS:
            continue
        total += 1
        if x1 <= 0.55 * w:
            left += 1
        elif x0 >= 0.45 * w:
            right += 1
    return total >= 4 and left / total > 0.25 and right / total > 0.25


def diagnose(path) -> tuple[dict, str]:
    row = {"article_id": path.stem, "file_kb": round(path.stat().st_size / 1024, 1)}
    t0 = time.perf_counter()
    doc = pymupdf.open(path)
    try:
        if doc.needs_pass:
            row.update(status="error", error="encrypted", encrypted=True)
            return row, ""
        pages_text, chars, image_only, with_images, two_col = [], [], 0, 0, 0
        for page in doc:
            text = page.get_text("text", sort=True)   # sort=True: чтение сверху вниз, слева направо
            n = len(text.strip())
            imgs = len(page.get_images(full=True))
            pages_text.append(text)
            chars.append(n)
            with_images += imgs > 0
            image_only += (n < MIN_TEXT_CHARS and imgs > 0)
            two_col += two_column_page(page)
        n_pages = len(chars)
        row.update(
            status="ok", error="", encrypted=False, pages=n_pages,
            text_chars_total=sum(chars),
            chars_per_page_median=int(statistics.median(chars)) if chars else 0,
            image_only_pages=image_only, pages_with_images=with_images,
            two_column_pages=two_col,
            scanned_suspect=bool(n_pages and image_only / n_pages >= 0.5),
            pdf_title_meta=(doc.metadata or {}).get("title", ""),
        )
        return row, "\f".join(pages_text)
    finally:
        doc.close()
        row["pymupdf_seconds"] = round(time.perf_counter() - t0, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    ensure_dirs()

    ids = pdf_article_ids()
    if not ids:
        log.error("В %s нет PDF. Сначала запустите 01_select_and_download.py", PDF_DIR)
        raise SystemExit(1)

    rows = []
    for aid in ids:
        path = PDF_DIR / f"{aid}.pdf"
        try:
            row, text = diagnose(path)
            if text:
                (TEXT_DIR / f"{aid}.txt").write_text(text, encoding="utf-8")
        except Exception as exc:  # повреждённый PDF и т.п.
            row = {"article_id": aid, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        log.info("%-16s %-5s стр=%-3s симв/стр=%-5s image_only=%-2s 2col=%-2s scan?=%s",
                 aid, row.get("status"), row.get("pages", "-"), row.get("chars_per_page_median", "-"),
                 row.get("image_only_pages", "-"), row.get("two_column_pages", "-"),
                 row.get("scanned_suspect", "-"))

    write_csv(DIAGNOSTICS_CSV, rows, FIELDS)
    ok = [r for r in rows if r.get("status") == "ok"]
    scans = [r["article_id"] for r in ok if r.get("scanned_suspect")]
    log.info("Готово: %d PDF, ошибок чтения: %d, подозрение на скан: %d %s",
             len(rows), len(rows) - len(ok), len(scans), scans or "")
    log.info("Отчёт: %s", DIAGNOSTICS_CSV)


if __name__ == "__main__":
    main()
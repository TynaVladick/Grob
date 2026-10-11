"""Общие пути и вспомогательные функции для всего пайплайна.

Все данные и результаты лежат внутри папки res/ (рядом с этим файлом).
Другую корневую папку можно задать переменной окружения RES_DIR.
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("RES_DIR", Path(__file__).resolve().parent))

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdf"              # исходные PDF
JATS_DIR = DATA_DIR / "jats"            # JATS XML из PMC (только для черновиков эталона!)
META_DIR = DATA_DIR / "meta"            # JSON-метаданные PMC (лицензия, DOI, md5 ...)
TEI_DIR = DATA_DIR / "tei"              # результат GROBID
EXTRACTED_DIR = DATA_DIR / "extracted"  # поля, разобранные из TEI
TEXT_DIR = DATA_DIR / "pymupdf_text"    # базовый текст PyMuPDF

GOLD_DIR = BASE_DIR / "gold"            # ручной эталон
RESULTS_DIR = BASE_DIR / "results"

MANIFEST_CSV = DATA_DIR / "manifest.csv"
REJECTED_CSV = DATA_DIR / "rejected.csv"
PROCESSING_LOG = RESULTS_DIR / "processing_log.csv"
RUN_META = RESULTS_DIR / "run_meta.json"
DIAGNOSTICS_CSV = RESULTS_DIR / "pdf_diagnostics.csv"
EVALUATION_JSON = RESULTS_DIR / "evaluation.json"
EVALUATION_CSV = RESULTS_DIR / "evaluation_per_article.csv"
TIMING_JSON = RESULTS_DIR / "timing_summary.json"

TEI_SUFFIX = ".tei.xml"


def ensure_dirs() -> None:
    for d in (PDF_DIR, JATS_DIR, META_DIR, TEI_DIR, EXTRACTED_DIR, TEXT_DIR,
              GOLD_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def pdf_article_ids() -> list[str]:
    """Идентификаторы статей (например 'PMC10009416.1') по файлам в data/pdf, по алфавиту."""
    return sorted(p.stem for p in PDF_DIR.glob("*.pdf"))


def tei_path(article_id: str) -> Path:
    return TEI_DIR / f"{article_id}{TEI_SUFFIX}"
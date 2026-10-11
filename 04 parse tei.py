#!/usr/bin/env python3
"""Шаг 4. TEI -> JSON с полями: title, authors, year, doi, abstract, section_headings, references.

    python 04_parse_tei.py
Результат: data/extracted/<id>.json
"""
from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET

from common import EXTRACTED_DIR, TEI_DIR, TEI_SUFFIX, ensure_dirs, setup_logging
from tei_parse import parse_tei

log = logging.getLogger("parse")


def main() -> None:
    setup_logging()
    ensure_dirs()
    files = sorted(TEI_DIR.glob(f"*{TEI_SUFFIX}"))
    if not files:
        raise SystemExit(f"В {TEI_DIR} нет TEI. Сначала запустите 03_run_grobid.py")
    ok = 0
    for f in files:
        aid = f.name[: -len(TEI_SUFFIX)]
        try:
            data = parse_tei(f)
        except ET.ParseError as exc:
            log.warning("%s: некорректный XML (%s) — пропущено", aid, exc)
            continue
        (EXTRACTED_DIR / f"{aid}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        ok += 1
        log.info("%-16s авторов=%-2d год=%s DOI=%s разделов=%-2d ссылок=%-3d аннотация=%d симв.",
                 aid, len(data["authors"]), data["year"] or "-", "да" if data["doi"] else "нет",
                 len(data["section_headings"]), data["n_references"], len(data["abstract"]))
    log.info("Разобрано %d из %d TEI -> %s", ok, len(files), EXTRACTED_DIR)


if __name__ == "__main__":
    main()
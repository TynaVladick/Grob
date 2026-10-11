#!/usr/bin/env python3
"""Шаг 5. Черновики эталона для ручной проверки.

Составлять эталон с нуля по PDF долго, поэтому скрипт делает ЧЕРНОВИК из JATS XML той же
статьи (PMC отдаёт его рядом с PDF). Это только заготовка:

  1. Откройте gold/<id>.json и сам PDF (data/pdf/<id>.pdf).
  2. Сверьте КАЖДОЕ поле с тем, что напечатано в PDF, и исправьте расхождения
     (особенно: год, состав авторов, заголовки разделов, ссылки без названий).
  3. Только после сверки поставьте "verified": true. Оценка (шаг 6) использует
     ТОЛЬКО файлы с verified=true.

Существующие файлы эталона НИКОГДА не перезаписываются — ваша ручная правка не потеряется.

Формат gold/<id>.json:
  title: str
  authors: [{given, surname}]
  year: int                 — год публикации, как он указан в PDF
  year_alt: [int]           — другие допустимые годы (например, год онлайн-публикации и год выпуска)
  doi: str
  abstract: str
  section_headings: [str]   — заголовки разделов тела статьи в порядке следования
  references: [{title, doi, year, first_author_surname}]

    python 05_make_gold_drafts.py --n 12
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import xml.etree.ElementTree as ET

from common import GOLD_DIR, JATS_DIR, MANIFEST_CSV, ensure_dirs, read_csv, setup_logging

log = logging.getLogger("gold")

PUBDATE_PREFERENCE = ["ppub", "collection", "epub", "epub-ppub", "pmc-release", "ecorrected"]


BLOCK_TAGS = {"p", "title", "sec", "label", "list-item", "caption", "abstract", "td", "th"}


def itxt(el) -> str:
    """Текст элемента: пробел ставится на границах блоков (title/p/sec...), но не внутри
    строчной разметки (<italic>, <sup> ...), чтобы 'pH' не превратилось в 'p H'."""
    if el is None:
        return ""
    parts: list[str] = []

    def walk(e) -> None:
        block = e.tag in BLOCK_TAGS
        if block:
            parts.append(" ")
        if e.text:
            parts.append(e.text)
        for ch in e:
            walk(ch)
            if ch.tail:
                parts.append(ch.tail)
        if block:
            parts.append(" ")

    walk(el)
    return " ".join("".join(parts).split())


def first(el, *paths):
    for p in paths:
        r = el.find(p)
        if r is not None:
            return r
    return None


def parse_jats(path) -> dict:
    root = ET.parse(path).getroot()
    meta = root if root.tag == "article-meta" else next(root.iter("article-meta"), None)
    if meta is None:
        raise ValueError("нет <article-meta>")

    title = itxt(meta.find("title-group/article-title"))

    authors = []
    for c in meta.findall("contrib-group/contrib"):
        if c.get("contrib-type", "author") != "author":
            continue
        n = first(c, "name", "name-alternatives/name")
        if n is None:
            continue
        authors.append({"given": itxt(n.find("given-names")), "surname": itxt(n.find("surname"))})

    # годы: основной — по приоритету типа даты, остальные — как допустимые альтернативы
    years = []
    for d in meta.findall("pub-date"):
        y = d.findtext("year")
        kind = d.get("pub-type") or d.get("date-type") or ""
        if y and y.strip().isdigit():
            years.append((PUBDATE_PREFERENCE.index(kind) if kind in PUBDATE_PREFERENCE else 99, int(y)))
    years.sort(key=lambda x: x[0])
    ordered = [y for _, y in years]
    year = ordered[0] if ordered else None
    year_alt = sorted({y for y in ordered if y != year})

    doi = ""
    for a in meta.findall("article-id"):
        if a.get("pub-id-type") == "doi":
            doi = itxt(a)

    abstract = ""
    abstracts = meta.findall("abstract")
    chosen = next((a for a in abstracts if not a.get("abstract-type")), abstracts[0] if abstracts else None)
    if chosen is not None:
        parts = []
        for ch in chosen:
            if ch.tag == "title" and itxt(ch).lower() == "abstract":
                continue
            parts.append(itxt(ch))
        abstract = " ".join(p for p in parts if p)

    headings = []
    body = root.find(".//body")
    if body is not None:
        for sec in body.iter("sec"):
            t = itxt(sec.find("title"))
            if t:
                headings.append(t)

    refs = []
    for ref in root.iter("ref"):
        cit = first(ref, "element-citation", "mixed-citation", "citation")
        if cit is None:
            continue
        ref_title = ""
        for tag in ("article-title", "chapter-title", "data-title", "source"):
            ref_title = itxt(cit.find(tag))
            if ref_title:
                break
        doi_r = ""
        for pid in cit.iter("pub-id"):
            if pid.get("pub-id-type") == "doi":
                doi_r = itxt(pid)
        y = cit.findtext(".//year")
        year_r = int(y.strip()[:4]) if y and y.strip()[:4].isdigit() else None
        sn = cit.find(".//person-group//surname")
        if sn is None:
            sn = cit.find(".//surname")
        refs.append({"title": ref_title, "doi": doi_r, "year": year_r,
                     "first_author_surname": itxt(sn)})

    return {"title": title, "authors": authors, "year": year, "year_alt": year_alt, "doi": doi,
            "abstract": abstract, "section_headings": headings, "references": refs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12, help="сколько статей взять в эталон (10–15)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    setup_logging()
    ensure_dirs()

    ids = [r["article_id"] for r in read_csv(MANIFEST_CSV)]
    ids = [a for a in ids if (JATS_DIR / f"{a}.xml").exists()]
    if not ids:
        raise SystemExit("Нет статей с JATS XML. Сначала запустите 01_select_and_download.py")

    existing = {p.stem for p in GOLD_DIR.glob("*.json")}
    chosen = sorted(existing)                      # уже начатые — сохраняем в выборке
    pool = [a for a in ids if a not in existing]
    random.Random(args.seed).shuffle(pool)
    chosen += pool[: max(0, args.n - len(chosen))]

    created = 0
    for aid in chosen:
        out = GOLD_DIR / f"{aid}.json"
        if out.exists():
            log.info("%-16s уже есть — не трогаю", aid)
            continue
        try:
            draft = parse_jats(JATS_DIR / f"{aid}.xml")
        except (ET.ParseError, ValueError) as exc:
            log.warning("%s: JATS не разобран (%s)", aid, exc)
            continue
        draft = {"article_id": aid, "verified": False, "source": "draft_from_JATS", "notes": "", **draft}
        out.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
        created += 1
        log.info("%-16s черновик создан: авторов=%d, разделов=%d, ссылок=%d",
                 aid, len(draft["authors"]), len(draft["section_headings"]), len(draft["references"]))

    log.info("Создано черновиков: %d. Теперь вручную сверьте каждый gold/*.json с PDF и "
             "поставьте \"verified\": true.", created)


if __name__ == "__main__":
    main()
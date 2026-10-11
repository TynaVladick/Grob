#!/usr/bin/env python3
"""Шаг 1. Отбор и загрузка статей PMC Open Access.

Способы автоматической загрузки, которые PMC разрешает (на октябрь 2026):
  * E-utilities (eSearch) — только чтобы найти PMCID;
  * PMC Cloud Service (S3-бакет pmc-oa-opendata, HTTPS без авторизации) —
    сами файлы: PDF, JATS XML и JSON-метаданные с лицензией.
Старые OA Web Service (oa.fcgi) и FTP отключены в августе 2026, здесь они не используются.

Лицензия проверяется ОТДЕЛЬНО для каждой статьи по её JSON-метаданным
(поле license_code), а не только по фильтру поиска. Скрипт отбрасывает рукописи
авторов, ретрагированные статьи, исторические OCR-сканы и статьи без PDF.
Причины отказа пишутся в data/rejected.csv.

Скрипт можно запускать повторно: уже скачанные статьи пропускаются,
добираются недостающие до --n.

Пример:
    export NCBI_EMAIL="you@example.org"     # желательно
    export NCBI_API_KEY="..."               # необязательно, ускоряет eutils
    python 01_select_and_download.py --n 30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import requests

from common import (JATS_DIR, MANIFEST_CSV, META_DIR, PDF_DIR, REJECTED_CSV,
                    ensure_dirs, read_csv, setup_logging, write_csv)

log = logging.getLogger("download")

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/eutils/esearch.fcgi"
S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
S3_BUCKET = "pmc-oa-opendata"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# Разные клинические области, чтобы в выборке были разные журналы и вёрстки.
DEFAULT_TOPICS = [
    "cardiology", "diabetes", "oncology", "infectious diseases", "neurology",
    "pediatrics", "psychiatry", "surgery", "public health", "pharmacology",
]

MANIFEST_FIELDS = [
    "article_id", "pmcid", "version", "pmid", "doi", "title", "citation", "journal",
    "license_code", "pdf_url", "pdf_bytes", "pdf_md5", "md5_verified",
    "source_prefix", "downloaded_utc",
]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    s = requests.Session()
    email = os.environ.get("NCBI_EMAIL", "")
    s.headers["User-Agent"] = f"grobid-baseline-benchmark/1.0 ({email or 'no-email'})"
    return s


def http_get(session: requests.Session, url: str, params=None, stream=False,
             tries: int = 4, timeout: int = 60) -> requests.Response:
    last: Exception | None = None
    for i in range(tries):
        try:
            r = session.get(url, params=params, stream=stream, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            return r
        except requests.RequestException as exc:
            last = exc
            wait = 2 ** i
            log.warning("GET %s не удался (%s); повтор через %s с", url, exc, wait)
            time.sleep(wait)
    assert last is not None
    raise last


def ncbi_params() -> dict:
    p = {"tool": "grobid-baseline-benchmark"}
    if os.environ.get("NCBI_EMAIL"):
        p["email"] = os.environ["NCBI_EMAIL"]
    if os.environ.get("NCBI_API_KEY"):
        p["api_key"] = os.environ["NCBI_API_KEY"]
    return p


def eutils_pause() -> None:
    # без ключа лимит 3 запроса/с, с ключом — 10 запросов/с
    time.sleep(0.12 if os.environ.get("NCBI_API_KEY") else 0.4)


# --------------------------------------------------------------------------- #
# Лицензии и проверки
# --------------------------------------------------------------------------- #
def norm_license(code: str | None) -> str:
    """'cc_by' / 'CC-BY' / 'CC BY' -> 'CC BY';  'CC BY-NC' остаётся 'CC BY-NC'."""
    s = re.sub(r"[_\s]+", " ", (code or "").strip().upper())
    return re.sub(r"^CC[- ]", "CC ", s)


def license_filter_name(code: str) -> str:
    """'CC BY' -> 'cc_by_license', 'CC0' -> 'cc0_license' (формат фильтров PMC)."""
    return code.strip().lower().replace(" ", "_") + "_license"


def truthy(v) -> bool:
    return v is True or str(v).strip().lower() in {"yes", "y", "true", "1"}


def check_meta(meta: dict, allowed: set[str]) -> list[str]:
    """Возвращает список причин отказа (пустой список = статья подходит)."""
    reasons = []
    if not truthy(meta.get("is_pmc_openaccess")):
        reasons.append("not_in_oa_subset")
    if truthy(meta.get("is_manuscript")):
        reasons.append("author_manuscript")
    if truthy(meta.get("is_historical_ocr")):
        reasons.append("historical_ocr")
    if truthy(meta.get("is_retracted")):
        reasons.append("retracted")
    lic = norm_license(meta.get("license_code"))
    if lic not in allowed:
        reasons.append(f"license:{lic or 'none'}")
    if not meta.get("pdf_url"):
        reasons.append("no_pdf")
    return reasons


def journal_of(citation: str | None) -> str:
    """Грубо: название журнала = текст до первой точки/точки с запятой в citation."""
    if not citation:
        return ""
    return re.split(r"\.\s|;\s", citation.strip(), maxsplit=1)[0].strip().lower()


# --------------------------------------------------------------------------- #
# Поиск кандидатов (E-utilities)
# --------------------------------------------------------------------------- #
def build_query(topic: str, years: str, licenses: set[str]) -> str:
    lic = " OR ".join(f"{license_filter_name(x)}[filter]" for x in sorted(licenses))
    return (f"({topic}) AND {years}[pdat] AND open_access[filter] AND has_pdf[filter] "
            f"AND ({lic}) NOT pmc embargo[filter]")


def esearch(session, term: str, retmax: int, retstart: int) -> dict:
    params = {"db": "pmc", "term": term, "retmode": "json",
              "retmax": retmax, "retstart": retstart, **ncbi_params()}
    r = http_get(session, ESEARCH, params=params)
    r.raise_for_status()
    eutils_pause()
    return r.json()["esearchresult"]


def collect_candidates(session, topics, years, licenses, per_topic, rng,
                       custom_query: str | None) -> list[str]:
    pmcids: list[str] = []
    queries = [custom_query] if custom_query else [build_query(t, years, licenses) for t in topics]
    for q in queries:
        head = esearch(session, q, retmax=1, retstart=0)
        count = int(head.get("count", 0))
        if count == 0:
            log.warning("Запрос ничего не нашёл: %s", q)
            continue
        # eSearch отдаёт максимум 10 000 записей; берём случайное окно, чтобы выборка не
        # состояла только из самых «релевантных»/новых статей
        upper = max(1, min(count, 9000) - per_topic)
        start = rng.randrange(0, upper)
        res = esearch(session, q, retmax=per_topic, retstart=start)
        ids = res.get("idlist", [])
        log.info("Запрос: %-60.60s | найдено %d, взято %d (с позиции %d)", q, count, len(ids), start)
        pmcids += [f"PMC{i}" for i in ids]
    seen, uniq = set(), []
    for p in pmcids:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    rng.shuffle(uniq)
    return uniq


# --------------------------------------------------------------------------- #
# PMC Cloud Service (S3, HTTPS)
# --------------------------------------------------------------------------- #
def list_versions(session, pmcid: str) -> list[str]:
    """Версии статьи в бакете: ['PMC123.1', 'PMC123.2'] (префикс 'PMC123.' + delimiter '/')."""
    r = http_get(session, S3_BASE + "/",
                 params={"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    out = [e.text.rstrip("/") for e in root.findall(".//s3:CommonPrefixes/s3:Prefix", S3_NS) if e.text]
    return sorted(out)


def fetch_meta(session, article_id: str) -> dict | None:
    r = http_get(session, f"{S3_BASE}/{article_id}/{article_id}.json")
    if r.status_code != 200:
        log.warning("Нет JSON-метаданных для %s (HTTP %s)", article_id, r.status_code)
        return None
    return r.json()


def s3_key_and_md5(url: str) -> tuple[str, str | None]:
    """Из s3://bucket/key?md5=... или https://...amazonaws.com/key?md5=... берём key и md5."""
    u = urlparse(url)
    key = u.path.lstrip("/")
    if key.startswith(S3_BUCKET + "/"):          # path-style URL
        key = key[len(S3_BUCKET) + 1:]
    md5 = parse_qs(u.query).get("md5", [None])[0]
    return key, md5


def download(session, key: str, dest, expected_md5: str | None, must_be_pdf: bool = False):
    """Скачивает объект, проверяет md5 (если он известен). Возвращает (bytes, md5, md5_ok)."""
    r = http_get(session, f"{S3_BASE}/{key}", stream=True, timeout=120)
    r.raise_for_status()
    h = hashlib.md5()
    size = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    first = b""
    with tmp.open("wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 16):
            if not first:
                first = chunk[:8]
            fh.write(chunk)
            h.update(chunk)
            size += len(chunk)
    digest = h.hexdigest()
    md5_ok = (digest == expected_md5) if expected_md5 else None
    if md5_ok is False:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"md5 не совпал для {key}: {digest} != {expected_md5}")
    if must_be_pdf and not first.startswith(b"%PDF"):
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{key} не похож на PDF")
    tmp.replace(dest)
    return size, digest, md5_ok


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30, help="сколько статей нужно всего (20–30)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--years", default="2021:2025", help="диапазон лет публикации для поиска")
    ap.add_argument("--licenses", default="CC BY,CC0",
                    help="допустимые лицензии через запятую, например 'CC BY,CC0,CC BY-SA'")
    ap.add_argument("--topics", default=",".join(DEFAULT_TOPICS),
                    help="темы поиска через запятую (для разнообразия журналов)")
    ap.add_argument("--query", default=None, help="свой полный запрос eSearch вместо тем")
    ap.add_argument("--pool-factor", type=float, default=3.0,
                    help="сколько кандидатов брать относительно --n (часть отсеется проверками)")
    ap.add_argument("--max-per-journal", type=int, default=4,
                    help="не больше стольких статей из одного журнала (разнообразие вёрсток)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    ensure_dirs()
    allowed = {norm_license(x) for x in args.licenses.split(",") if x.strip()}
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    rng = random.Random(args.seed)
    session = make_session()

    manifest = read_csv(MANIFEST_CSV)
    have = {r["article_id"] for r in manifest}
    have_pmcids = {r["pmcid"] for r in manifest}
    per_journal = Counter(r.get("journal", "") for r in manifest)
    rejected = read_csv(REJECTED_CSV)
    rejected_ids = {r["article_id"] for r in rejected}

    if len(manifest) >= args.n:
        log.info("В манифесте уже %d статей — ничего докачивать не нужно.", len(manifest))
        return

    per_topic = max(5, math.ceil(args.n * args.pool_factor / max(1, len(topics))))
    candidates = collect_candidates(session, topics, args.years, allowed, per_topic, rng, args.query)
    log.info("Кандидатов: %d; допустимые лицензии: %s", len(candidates), sorted(allowed))

    try:
        for pmcid in candidates:
            if len(manifest) >= args.n:
                break
            if pmcid in have_pmcids:
                continue
            try:
                versions = list_versions(session, pmcid)
            except requests.RequestException as exc:
                log.warning("%s: не удалось получить список версий: %s", pmcid, exc)
                continue
            for article_id in versions:
                if article_id in have or article_id in rejected_ids:
                    continue
                meta = fetch_meta(session, article_id)
                if meta is None:
                    rejected.append({"article_id": article_id, "reason": "no_metadata"})
                    continue
                reasons = check_meta(meta, allowed)
                journal = journal_of(meta.get("citation"))
                if not reasons and per_journal[journal] >= args.max_per_journal:
                    reasons.append(f"journal_limit:{journal}")
                if reasons:
                    log.info("× %-16s %s", article_id, ", ".join(reasons))
                    rejected.append({"article_id": article_id, "reason": ";".join(reasons),
                                     "license_code": meta.get("license_code", "")})
                    continue

                try:
                    key, md5 = s3_key_and_md5(meta["pdf_url"])
                    pdf_path = PDF_DIR / f"{article_id}.pdf"
                    size, digest, md5_ok = download(session, key, pdf_path, md5, must_be_pdf=True)
                except (requests.RequestException, ValueError, KeyError) as exc:
                    log.warning("× %-16s загрузка PDF не удалась: %s", article_id, exc)
                    rejected.append({"article_id": article_id, "reason": f"pdf_download_failed:{exc}"})
                    continue

                # JATS XML нужен только для черновика эталона (шаг 5); GROBID его не видит.
                try:
                    xml_key, xml_md5 = s3_key_and_md5(meta.get("xml_url") or f"{article_id}/{article_id}.xml")
                    download(session, xml_key, JATS_DIR / f"{article_id}.xml", xml_md5)
                except (requests.RequestException, ValueError) as exc:
                    log.warning("  %s: JATS XML не скачан (%s) — черновик эталона для неё будет недоступен",
                                article_id, exc)

                (META_DIR / f"{article_id}.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

                manifest.append({
                    "article_id": article_id, "pmcid": pmcid,
                    "version": article_id.rsplit(".", 1)[-1],
                    "pmid": meta.get("pmid", ""), "doi": meta.get("doi", ""),
                    "title": meta.get("title", ""), "citation": meta.get("citation", ""),
                    "journal": journal, "license_code": meta.get("license_code", ""),
                    "pdf_url": meta.get("pdf_url", ""), "pdf_bytes": size, "pdf_md5": digest,
                    "md5_verified": md5_ok, "source_prefix": f"{S3_BASE}/{article_id}/",
                    "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                have.add(article_id)
                have_pmcids.add(pmcid)
                per_journal[journal] += 1
                write_csv(MANIFEST_CSV, manifest, MANIFEST_FIELDS)
                log.info("✓ %-16s %-8s %7.1f КБ  %s", article_id, meta.get("license_code", ""),
                         size / 1024, (meta.get("title") or "")[:70])
                break   # одной версии на PMCID достаточно
    finally:
        write_csv(REJECTED_CSV, rejected, ["article_id", "reason", "license_code"])

    log.info("Готово: %d статей в манифесте (цель %d). Отклонено: %d.", len(manifest), args.n, len(rejected))
    if len(manifest) < args.n:
        log.warning("Статей меньше цели. Запустите ещё раз с другим --seed или расширьте --years/--topics.")
    log.info("Не забудьте проверить лицензии в %s и сохранить атрибуцию при любом распространении.", MANIFEST_CSV)


if __name__ == "__main__":
    main()
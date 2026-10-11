#!/usr/bin/env python3
"""Шаг 3. Последовательная обработка всех PDF одним локальным экземпляром GROBID.

Для каждой статьи сохраняется:
  * исходный PDF            -> data/pdf/<id>.pdf        (уже лежит с шага 1)
  * результат TEI           -> data/tei/<id>.tei.xml
  * время обработки, статус -> results/processing_log.csv
Общие сведения о прогоне (версия GROBID, параметры, общее время, машина) -> results/run_meta.json

PDF отправляются строго по одному, без параллелизма: это базовая линия, с которой потом
можно сравнивать распределённую обработку. По умолчанию консолидация через внешние
сервисы (Crossref и т.п.) ВЫКЛЮЧЕНА, чтобы измерять сам GROBID, а не доступность сети.

Запуск GROBID (один из вариантов, версии смотрите в документации GROBID):
    docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.0-crf
    # полный образ с DL-моделями (точнее, но тяжелее; желательно GPU):
    docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.0-full

Затем:
    python 03_run_grobid.py --label crf-0.9.0
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

from common import (PDF_DIR, PROCESSING_LOG, RUN_META, TEI_DIR, ensure_dirs, pdf_article_ids,
                    setup_logging, tei_path)

log = logging.getLogger("grobid")

LOG_FIELDS = ["run_id", "index", "article_id", "pdf_path", "pdf_kb", "tei_path", "status",
              "http_status", "error", "seconds", "seconds_incl_retries", "attempts",
              "tei_kb", "started_utc"]


def check_service(base: str) -> str:
    try:
        r = requests.get(f"{base}/api/isalive", timeout=10)
        alive = r.ok and r.text.strip().lower() == "true"
    except requests.RequestException as exc:
        raise SystemExit(f"GROBID недоступен по адресу {base}: {exc}\n"
                         "Запустите, например: docker run --rm --init --ulimit core=0 "
                         "-p 8070:8070 grobid/grobid:0.9.0-crf")
    if not alive:
        raise SystemExit(f"GROBID по адресу {base} отвечает, но /api/isalive не вернул 'true'.")
    try:
        return requests.get(f"{base}/api/version", timeout=10).text.strip()
    except requests.RequestException:
        return "unknown"


def process_one(session: requests.Session, endpoint: str, pdf_path, params: dict,
                timeout: int, max_retries: int) -> dict:
    """Одна статья. 'seconds' — время последнего (успешного) запроса, без ожиданий между повторами."""
    t_start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    res = {"status": "error", "http_status": "", "error": "", "attempts": 0,
           "seconds": 0.0, "content": b"", "started_utc": started}
    for attempt in range(1, max_retries + 2):
        res["attempts"] = attempt
        t0 = time.perf_counter()
        try:
            with pdf_path.open("rb") as fh:
                r = session.post(endpoint, files={"input": (pdf_path.name, fh, "application/pdf")},
                                 data=params, timeout=timeout)
            res["seconds"] = time.perf_counter() - t0
            res["http_status"] = r.status_code
            if r.status_code == 200:
                res["status"], res["error"], res["content"] = "ok", "", r.content
                break
            if r.status_code == 503 and attempt <= max_retries:       # все потоки GROBID заняты
                wait = float(r.headers.get("Retry-After", 5))
                log.warning("  503 (занят), повтор через %.0f с", wait)
                time.sleep(wait)
                continue
            res["status"] = "empty_response" if r.status_code == 204 else "http_error"
            res["error"] = ((r.text or "")[:300].replace("\n", " ")
                            or ("GROBID вернул 204: текст не извлечён" if r.status_code == 204 else ""))
            break
        except requests.Timeout:
            res["seconds"] = time.perf_counter() - t0
            res.update(status="timeout", error=f"timeout {timeout}s")
            break
        except requests.RequestException as exc:
            res["seconds"] = time.perf_counter() - t0
            res.update(status="connection_error", error=str(exc)[:300])
            break
    res["seconds_incl_retries"] = time.perf_counter() - t_start
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("GROBID_URL", "http://localhost:8070"))
    ap.add_argument("--label", default="", help="метка прогона для run_meta.json, например 'crf-0.9.0'")
    ap.add_argument("--timeout", type=int, default=300, help="таймаут одного запроса, с")
    ap.add_argument("--max-retries", type=int, default=3, help="повторы только при HTTP 503")
    ap.add_argument("--consolidate-header", default="0", choices=["0", "1", "2"])
    ap.add_argument("--consolidate-citations", default="0", choices=["0", "1", "2"])
    ap.add_argument("--limit", type=int, default=None, help="обработать только первые N PDF (для отладки)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    ensure_dirs()
    base = args.url.rstrip("/")
    version = check_service(base)
    log.info("GROBID %s на %s", version, base)

    ids = pdf_article_ids()
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        raise SystemExit(f"В {PDF_DIR} нет PDF. Сначала запустите 01_select_and_download.py")

    # предыдущий лог не затираем
    if PROCESSING_LOG.exists():
        stamp = datetime.fromtimestamp(PROCESSING_LOG.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
        PROCESSING_LOG.rename(PROCESSING_LOG.with_name(f"processing_log.{stamp}.csv"))

    params = {
        "consolidateHeader": args.consolidate_header,
        "consolidateCitations": args.consolidate_citations,
        "includeRawCitations": "1",   # исходная строка ссылки — удобно для разбора ошибок
        "teiCoordinates": "",         # координаты не нужны
    }
    params = {k: v for k, v in params.items() if v != ""}
    endpoint = f"{base}/api/processFulltextDocument"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    session = requests.Session()
    n_ok = n_err = 0
    t_batch0 = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with PROCESSING_LOG.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for i, aid in enumerate(ids, 1):
            pdf = PDF_DIR / f"{aid}.pdf"
            out = tei_path(aid)
            res = process_one(session, endpoint, pdf, params, args.timeout, args.max_retries)

            tei_kb = ""
            if res["status"] == "ok":
                out.write_bytes(res["content"])
                tei_kb = round(len(res["content"]) / 1024, 1)
                try:                                   # TEI должен быть корректным XML
                    ET.fromstring(res["content"])
                except ET.ParseError as exc:
                    res.update(status="bad_tei", error=f"XML parse error: {exc}")
            if res["status"] == "ok":
                n_ok += 1
            else:
                n_err += 1

            writer.writerow({
                "run_id": run_id, "index": i, "article_id": aid, "pdf_path": str(pdf),
                "pdf_kb": round(pdf.stat().st_size / 1024, 1),
                "tei_path": str(out) if tei_kb != "" else "", "status": res["status"],
                "http_status": res["http_status"], "error": res["error"],
                "seconds": round(res["seconds"], 3),
                "seconds_incl_retries": round(res["seconds_incl_retries"], 3),
                "attempts": res["attempts"], "tei_kb": tei_kb, "started_utc": res["started_utc"],
            })
            fh.flush()
            log.info("[%2d/%d] %-16s %-15s %6.1f с  %s", i, len(ids), aid, res["status"],
                     res["seconds"], res["error"][:80])
    batch_seconds = time.perf_counter() - t_batch0

    meta = {
        "run_id": run_id, "label": args.label, "grobid_url": base, "grobid_version": version,
        "endpoint": "/api/processFulltextDocument", "params": params,
        "mode": "sequential, 1 GROBID instance, 1 client thread",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch_wall_seconds": round(batch_seconds, 3),
        "n_pdfs": len(ids), "n_ok": n_ok, "n_errors": n_err,
        "client_machine": {"platform": platform.platform(), "python": platform.python_version(),
                           "cpu_count": os.cpu_count()},
        "note": ("Параметры сервера GROBID (число потоков, образ crf/full, GPU, ресурсы Docker) "
                 "в API не видны — укажите их в --label и в отчёте."),
    }
    RUN_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Готово: %d ok, %d ошибок, общее время %.1f с. Лог: %s", n_ok, n_err, batch_seconds, PROCESSING_LOG)


if __name__ == "__main__":
    main()
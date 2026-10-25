"""Автоматическая загрузка статей в открытом доступе (только законные OA-источники, без обхода paywall).

Для каждой статьи по очереди пробуются источники (первый успешный PDF останавливает перебор):
  1. PMC Cloud Service (S3 pmc-oa-opendata) — по PMCID; проверяется md5 и лицензия;
  2. Europe PMC (рендер PDF) — по PMCID;
  3. ссылки на OA-PDF, которые отдали сами базы поиска (OpenAlex, Semantic Scholar, Europe PMC);
  4. Unpaywall — по DOI (нужен email: CONTACT_EMAIL или NCBI_EMAIL).
Каждый скачанный файл проверяется: это действительно PDF (сигнатура %PDF), а не страница издателя.
"""
from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import requests

from .http import Http, contact_email
from .models import Article, SearchPlan

log = logging.getLogger("litsearch.download")

S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
S3_BUCKET = "pmc-oa-opendata"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
EPMC_PDF = "https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
UNPAYWALL = "https://api.unpaywall.org/v2/"
MIN_PDF_BYTES = 5_000
MAX_PDF_BYTES = 200 * 1024 * 1024


class DownloadError(RuntimeError):
    pass


@dataclass
class Candidate:
    source: str
    url: str
    license: str = ""
    md5: str | None = None


@dataclass
class DownloadResult:
    status: str                  # ok | exists | failed
    path: str = ""
    source: str = ""
    url: str = ""
    license: str = ""
    bytes: int = 0
    error: str = ""

    @property
    def success(self) -> bool:
        return self.status in ("ok", "exists")


def norm_license(code: str | None) -> str:
    s = re.sub(r"[_\s]+", " ", (code or "").strip().upper())
    return re.sub(r"^CC[- ]", "CC ", s)


class Downloader:
    def __init__(self, http: Http, pdf_dir: Path, email: str | None = None,
                 allowed_licenses: set[str] | None = None, use_unpaywall: bool = True):
        self.http = http
        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.email = contact_email() if email is None else email
        self.allowed = {norm_license(x) for x in allowed_licenses} if allowed_licenses else None
        self.use_unpaywall = use_unpaywall

    # ---- имена файлов --------------------------------------------------- #
    @staticmethod
    def file_stem(a: Article) -> str:
        if a.pmcid:
            return a.pmcid
        if a.pmid:
            return f"PMID{a.pmid}"
        if a.doi:
            return "doi_" + re.sub(r"[^\w.-]+", "_", a.doi)[:100]
        return "title_" + hashlib.md5((a.title or "").encode()).hexdigest()[:12]

    # ---- источники ------------------------------------------------------ #
    def _pmc_s3(self, pmcid: str) -> Candidate | None:
        r = self.http.get(S3_BASE + "/", params={"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"})
        if r.status_code != 200:
            return None
        root = ET.fromstring(r.content)
        versions = [e.text.rstrip("/") for e in root.findall(".//s3:CommonPrefixes/s3:Prefix", S3_NS) if e.text]
        if not versions:                                  # статьи нет в OA-наборе PMC
            return None
        art_id = max(versions, key=lambda v: int(v.rsplit(".", 1)[-1]) if v.rsplit(".", 1)[-1].isdigit() else 0)
        r = self.http.get(f"{S3_BASE}/{art_id}/{art_id}.json")
        if r.status_code != 200:
            return None
        meta = r.json()
        if str(meta.get("is_retracted", "")).lower() in ("yes", "true", "1"):
            raise DownloadError("статья отозвана (retracted)")
        lic = norm_license(meta.get("license_code"))
        if self.allowed is not None and lic not in self.allowed:
            raise DownloadError(f"лицензия {lic or 'не указана'} не входит в разрешённые")
        if not meta.get("pdf_url"):
            return None
        u = urlparse(meta["pdf_url"])
        key = u.path.lstrip("/")
        if key.startswith(S3_BUCKET + "/"):
            key = key[len(S3_BUCKET) + 1:]
        return Candidate("pmc_s3", f"{S3_BASE}/{key}", meta.get("license_code", ""), parse_qs(u.query).get("md5", [None])[0])

    def _unpaywall(self, doi: str) -> list[Candidate]:
        d = self.http.get_json(UNPAYWALL + quote(doi, safe="/"), params={"email": self.email})
        locs = [d.get("best_oa_location"), *(d.get("oa_locations") or [])]
        out, seen = [], set()
        for loc in locs:
            if loc and loc.get("url_for_pdf") and loc["url_for_pdf"] not in seen:
                seen.add(loc["url_for_pdf"])
                out.append(Candidate("unpaywall", loc["url_for_pdf"], loc.get("license") or ""))
        return out

    def candidates(self, a: Article, errors: list[str]):
        """Ленивый перебор: следующий источник (и его API-запрос) вызывается, только если предыдущие не помогли."""
        seen = set()

        def fresh(c: Candidate | None):
            if c and c.url not in seen:
                seen.add(c.url)
                return c

        if a.pmcid:
            try:
                c = fresh(self._pmc_s3(a.pmcid))
                if c:
                    yield c
            except (requests.RequestException, ET.ParseError, ValueError, DownloadError) as exc:
                errors.append(f"pmc_s3: {exc}")
            c = fresh(Candidate("europepmc", EPMC_PDF.format(pmcid=a.pmcid), a.license))
            if c:
                yield c
        for u in a.oa_urls:
            c = fresh(Candidate("search_result", u, a.license))
            if c:
                yield c
        if a.doi and self.email and self.use_unpaywall:
            try:
                for cand in self._unpaywall(a.doi):
                    c = fresh(cand)
                    if c:
                        yield c
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"unpaywall: {exc}")

    # ---- загрузка ------------------------------------------------------- #
    def _fetch_pdf(self, c: Candidate, dest: Path) -> int:
        r = self.http.get(c.url, stream=True, timeout=120, tries=2)
        try:
            if r.status_code != 200:
                raise DownloadError(f"HTTP {r.status_code}")
            chunks = r.iter_content(chunk_size=1 << 16)
            first = next(chunks, b"")
            if b"%PDF" not in first[:1024]:
                raise DownloadError(f"не PDF ({r.headers.get('Content-Type', '?')[:40]})")
            tmp = dest.with_suffix(".part")
            h, size = hashlib.md5(), 0
            try:
                with tmp.open("wb") as fh:
                    for chunk in _chain(first, chunks):
                        size += len(chunk)
                        if size > MAX_PDF_BYTES:
                            raise DownloadError("файл больше 200 МБ")
                        fh.write(chunk)
                        h.update(chunk)
                if size < MIN_PDF_BYTES:
                    raise DownloadError(f"слишком маленький файл ({size} Б)")
                if c.md5 and h.hexdigest() != c.md5:
                    raise DownloadError("md5 не совпал")
                tmp.replace(dest)
            finally:
                tmp.unlink(missing_ok=True)
            return size
        finally:
            r.close()

    def download(self, a: Article) -> DownloadResult:
        dest = self.pdf_dir / f"{self.file_stem(a)}.pdf"
        if dest.exists() and dest.stat().st_size >= MIN_PDF_BYTES:
            return DownloadResult("exists", str(dest), bytes=dest.stat().st_size)
        errors: list[str] = []
        for c in self.candidates(a, errors):
            try:
                size = self._fetch_pdf(c, dest)
            except (DownloadError, requests.RequestException, OSError) as exc:
                errors.append(f"{c.source}: {exc}")
                continue
            return DownloadResult("ok", str(dest), c.source, c.url, c.license or a.license, size)
        return DownloadResult("failed", error="; ".join(errors) or "нет ни одного источника открытого PDF")

    def download_many(self, articles: list[Article], target: int, workers: int = 4,
                      on_result=None) -> dict[int, DownloadResult]:
        """Скачивает по порядку, пока не наберётся `target` успешных (неудачные заменяются следующими).
        Возвращает {индекс статьи: результат} только для тех, что пробовали."""
        results: dict[int, DownloadResult] = {}
        done = 0
        pending = iter(range(len(articles)))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            while done < target:
                batch = [i for _, i in zip(range(max(1, min(workers, target - done))), pending)]
                if not batch:
                    break
                futs = {ex.submit(self.download, articles[i]): i for i in batch}
                for f in as_completed(futs):
                    i = futs[f]
                    try:
                        res = f.result()
                    except Exception as exc:            # noqa: BLE001
                        res = DownloadResult("failed", error=f"{type(exc).__name__}: {exc}")
                    results[i] = res
                    done += res.success
                    if on_result:
                        on_result(articles[i], res, done, target)
        return results


def _chain(first: bytes, rest):
    yield first
    yield from rest

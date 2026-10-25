"""HTTP-слой: общая сессия, повторы, уважение к лимитам (пауза между запросами к одному хосту)."""
from __future__ import annotations

import logging
import os
import threading
import time
from urllib.parse import urlparse

import requests

log = logging.getLogger("litsearch.http")


def contact_email() -> str:
    return os.environ.get("CONTACT_EMAIL") or os.environ.get("NCBI_EMAIL") or ""


def default_intervals() -> dict[str, float]:
    """Минимальный интервал между запросами к хосту, секунды."""
    return {
        "eutils.ncbi.nlm.nih.gov": 0.11 if os.environ.get("NCBI_API_KEY") else 0.36,  # 10 / 3 запроса в секунду
        "www.ebi.ac.uk": 0.1,
        "europepmc.org": 0.2,
        "api.openalex.org": 0.1,
        "api.semanticscholar.org": 1.05,       # лимит Semantic Scholar — 1 запрос в секунду
        "api.unpaywall.org": 0.1,
    }


class Http:
    def __init__(self, intervals: dict[str, float] | None = None):
        self.session = requests.Session()
        email = contact_email()
        self.session.headers["User-Agent"] = f"grob-litsearch/1.0 ({email or 'no-email'})"
        self.intervals = default_intervals() if intervals is None else intervals
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def _wait(self, url: str) -> None:
        host = urlparse(url).netloc
        gap = self.intervals.get(host, 0)
        if not gap:
            return
        with self._lock:                       # резервируем слот, затем спим вне блокировки
            now = time.monotonic()
            slot = max(self._next.get(host, 0.0), now)
            self._next[host] = slot + gap
        if slot > now:
            time.sleep(slot - now)

    def get(self, url, params=None, headers=None, stream=False, timeout=60, tries=4) -> requests.Response:
        last: Exception | None = None
        for i in range(tries):
            self._wait(url)
            try:
                r = self.session.get(url, params=params, headers=headers, stream=stream, timeout=timeout)
            except requests.RequestException as exc:
                last = exc
                log.warning("GET %s не удался: %s", urlparse(url).netloc, type(exc).__name__)
            else:
                if r.status_code not in (429, 500, 502, 503, 504):
                    return r
                last = requests.HTTPError(f"HTTP {r.status_code}", response=r)
                ra = r.headers.get("Retry-After", "")
                r.close()
                if i < tries - 1:
                    time.sleep(min(float(ra) if ra.isdigit() else 2 ** (i + 1), 30))
                    continue
                break
            if i < tries - 1:
                time.sleep(min(2 ** i, 10))
        assert last is not None
        raise last

    def get_json(self, url, params=None, headers=None, timeout=60) -> dict:
        r = self.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

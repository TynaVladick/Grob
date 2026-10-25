"""PubMed (NCBI E-utilities). Основной провайдер диалога: именно по нему подбирается число статей.

Запрос строится по плану: ("MeSH"[MeSH Terms] OR "слово"[tiab] ...) AND (...) AND фильтры.
"""
from __future__ import annotations

import logging
import os
import re

import requests

from ..models import Article, SearchPlan, LANGUAGES, norm_doi, norm_pmcid
from .base import Provider, ProviderError

log = logging.getLogger("litsearch.pubmed")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

PUBMED_PT = {
    "rct": "randomized controlled trial[pt]",
    "clinical_trial": "clinical trial[pt]",
    "meta_analysis": "meta-analysis[pt]",
    "systematic_review": "systematic review[pt]",
    "review": "review[pt]",
    "guideline": "(guideline[pt] OR practice guideline[pt])",
    "case_report": "case reports[pt]",
}


def tiab(term: str) -> str:
    """Свободное слово в заголовке/аннотации. Усечение (*) в PubMed работает только вне кавычек."""
    if "*" in term:
        if " " in term:
            term = term.replace("*", "")
        elif len(term.rstrip("*")) >= 4:
            return f"{term}[tiab]"
        else:
            term = term.replace("*", "")
    return f'"{term}"[tiab]'


class PubMed(Provider):
    name = "pubmed"
    label = "PubMed"

    def available(self):
        return True, "" if os.environ.get("NCBI_API_KEY") else "без NCBI_API_KEY лимит 3 запроса/с (работает, но медленнее)"

    # ---- запрос ------------------------------------------------------------
    def render(self, plan: SearchPlan) -> str:
        blocks = []
        for c in plan.concepts:
            alts = [f'"{m}"[MeSH Terms]' for m in c.mesh] + [tiab(k) for k in c.keywords]
            if alts:
                blocks.append("(" + " OR ".join(alts) + ")")
        q = " AND ".join(blocks)
        if plan.exclude:
            q = f"({q}) NOT (" + " OR ".join(tiab(x) for x in plan.exclude) + ")"
        f = plan.filters
        extra = []
        if f.year_from or f.year_to:
            extra.append(f"{f.year_from or 1800}:{f.year_to or 3000}[dp]")
        if f.study_types:
            extra.append("(" + " OR ".join(PUBMED_PT[t] for t in f.study_types) + ")")
        if f.humans_only:
            extra.append("humans[mh]")
        if f.languages:
            extra.append("(" + " OR ".join(f"{LANGUAGES[x][0]}[la]" for x in f.languages) + ")")
        if f.open_access_only:
            extra.append("free full text[sb]")
        if f.has_abstract:
            extra.append("hasabstract[text]")
        return " AND ".join([q, *extra]) if q else " AND ".join(extra)

    # ---- E-utilities -------------------------------------------------------
    def _params(self, **kw) -> dict:
        p = {"tool": "grob-litsearch"}
        if os.environ.get("NCBI_EMAIL"):
            p["email"] = os.environ["NCBI_EMAIL"]
        if os.environ.get("NCBI_API_KEY"):
            p["api_key"] = os.environ["NCBI_API_KEY"]
        p.update(kw)
        return p

    def _esearch(self, term: str, retmax: int, retstart: int = 0, sort: str | None = None) -> dict:
        p = self._params(db="pubmed", term=term, retmode="json", retmax=retmax, retstart=retstart)
        if sort:
            p["sort"] = sort
        data = self.http.get_json(EUTILS + "esearch.fcgi", params=p)
        res = data.get("esearchresult")
        if not res:
            raise ProviderError(f"PubMed: {data.get('error') or data}")
        if res.get("ERROR"):
            raise ProviderError(f"PubMed: {res['ERROR']}")
        return res

    def count(self, plan: SearchPlan) -> int:
        return int(self._esearch(self.render(plan), 0)["count"])

    def search(self, plan: SearchPlan, limit: int) -> list[Article]:
        res = self._esearch(self.render(plan), min(limit, 9999), sort="relevance")
        ids = res.get("idlist", [])
        arts = []
        for i in range(0, len(ids), 150):
            data = self.http.get_json(EUTILS + "esummary.fcgi", params=self._params(
                db="pubmed", id=",".join(ids[i:i + 150]), retmode="json"))
            result = data.get("result", {})
            for uid in ids[i:i + 150]:
                if uid in result:
                    arts.append(self.parse_summary(uid, result[uid]))
        return arts

    @staticmethod
    def parse_summary(uid: str, rec: dict) -> Article:
        ids = {x.get("idtype"): x.get("value", "") for x in rec.get("articleids", [])}
        m = re.search(r"\d{4}", rec.get("pubdate") or rec.get("epubdate") or "")
        return Article(
            title=(rec.get("title") or "").strip().rstrip("."),
            year=int(m.group()) if m else None,
            journal=rec.get("fulljournalname") or rec.get("source") or "",
            authors=[a["name"] for a in rec.get("authors", []) if a.get("name")][:50],
            doi=norm_doi(ids.get("doi")), pmid=uid,
            pmcid=norm_pmcid(ids.get("pmc") or ids.get("pmcid")),
        )

    # ---- проверка MeSH -----------------------------------------------------
    def validate_mesh(self, plan: SearchPlan) -> list[str]:
        """Удаляет из плана заголовки MeSH, которых нет в PubMed (LLM мог ошибиться);
        удалённое сохраняется как обычное слово. Сетевые сбои не приводят к удалению."""
        removed, offline = [], False
        for c in plan.concepts:
            keep = []
            for m in c.mesh:
                ok = True
                if not offline:
                    try:
                        ok = int(self._esearch(f'"{m}"[MeSH Terms]', 0)["count"]) > 0
                    except requests.RequestException as exc:       # сеть/сервер недоступны — дальше не пытаемся
                        log.warning("PubMed недоступен, проверка MeSH пропущена: %s", exc)
                        offline = True
                    except (ProviderError, ValueError, KeyError) as exc:
                        log.debug("MeSH %r не проверен: %s", m, exc)
                if ok:
                    keep.append(m)
                else:
                    removed.append(m)
                    if m.casefold() not in {k.casefold() for k in c.keywords}:
                        c.keywords.append(m)
            c.mesh = keep
        return removed

"""Europe PMC: PubMed + PMC + патенты/препринты; удобен тем, что сразу знает, какие статьи в открытом доступе.

Запрос: (MESH:"..." OR TITLE:"..." OR ABSTRACT:"...") AND PUB_YEAR:[..] AND ... AND NOT SRC:PPR
"""
from __future__ import annotations

from ..models import Article, SearchPlan, LANGUAGES
from .base import Provider

URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

EPMC_PT = {
    "rct": ['"Randomized Controlled Trial"'],
    "clinical_trial": ['"Clinical Trial"'],
    "meta_analysis": ['"Meta-Analysis"'],
    "systematic_review": ['"Systematic Review"'],
    "review": ['"Review"'],
    "guideline": ['"Guideline"', '"Practice Guideline"'],
    "case_report": ['"Case Reports"'],
}
PDF_RENDER = "https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"


def fld(name: str, term: str) -> str:
    if "*" in term:
        if " " in term or len(term.rstrip("*")) < 4:
            term = term.replace("*", "")
        else:
            return f"{name}:{term}"
    return f'{name}:"{term}"'


class EuropePMC(Provider):
    name = "europepmc"
    label = "Europe PMC"

    def render(self, plan: SearchPlan) -> str:
        blocks = []
        for c in plan.concepts:
            alts = [f'MESH:"{m}"' for m in c.mesh]
            for k in c.keywords:
                alts += [fld("TITLE", k), fld("ABSTRACT", k)]
            if alts:
                blocks.append("(" + " OR ".join(alts) + ")")
        q = " AND ".join(blocks)
        if plan.exclude:
            q = f"({q}) NOT (" + " OR ".join(fld(n, x) for x in plan.exclude for n in ("TITLE", "ABSTRACT")) + ")"
        f = plan.filters
        extra = []
        if f.year_from or f.year_to:
            extra.append(f"PUB_YEAR:[{f.year_from or 1800} TO {f.year_to or 2100}]")
        if f.study_types:
            extra.append("(" + " OR ".join(f"PUB_TYPE:{t}" for s in f.study_types for t in EPMC_PT[s]) + ")")
        if f.humans_only:
            extra.append('MESH:"Humans"')
        if f.languages:
            extra.append("(" + " OR ".join(f"LANG:{LANGUAGES[x][1]}" for x in f.languages) + ")")
        if f.open_access_only:
            extra.append("OPEN_ACCESS:y")
        if f.has_abstract:
            extra.append("HAS_ABSTRACT:y")
        extra.append("NOT SRC:PPR")                       # без препринтов
        return " AND ".join([q, *extra])

    def count(self, plan: SearchPlan) -> int:
        d = self.http.get_json(URL, params={"query": self.render(plan), "format": "json",
                                            "pageSize": 1, "resultType": "lite"})
        return int(d.get("hitCount", 0))

    def search(self, plan: SearchPlan, limit: int) -> list[Article]:
        q, cursor, arts = self.render(plan), "*", []
        while len(arts) < limit:
            d = self.http.get_json(URL, params={"query": q, "format": "json", "resultType": "lite",
                                                "pageSize": min(100, limit - len(arts)), "cursorMark": cursor})
            batch = d.get("resultList", {}).get("result", [])
            arts += [self.parse(r) for r in batch]
            nxt = d.get("nextCursorMark")
            if not batch or not nxt or nxt == cursor:
                break
            cursor = nxt
        return arts[:limit]

    @staticmethod
    def parse(r: dict) -> Article:
        pmcid = r.get("pmcid") or ""
        oa = r.get("isOpenAccess") == "Y"
        year = r.get("pubYear")
        return Article(
            title=(r.get("title") or "").strip().rstrip("."),
            year=int(year) if str(year or "").isdigit() else None,
            journal=r.get("journalTitle") or "",
            authors=[a.strip() for a in (r.get("authorString") or "").rstrip(".").split(",") if a.strip()][:50],
            doi=r.get("doi") or "", pmcid=pmcid,
            pmid=r.get("pmid") or (r.get("id") if r.get("source") == "MED" else "") or "",
            is_oa=oa if r.get("isOpenAccess") else None,
            oa_urls=[PDF_RENDER.format(pmcid=pmcid)] if pmcid and oa else [],
            cited_by=r.get("citedByCount"),
        )

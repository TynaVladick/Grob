"""Semantic Scholar (Academic Graph API, endpoint /paper/search/bulk с булевым синтаксисом).

Особенности: выдача bulk не ранжируется по релевантности, поэтому берём самые цитируемые (sort=citationCount:desc);
фильтры «только люди», «язык», «есть аннотация» не поддерживаются. Ключ S2_API_KEY необязателен, но без него общий
лимит часто даёт HTTP 429 (запросы автоматически повторяются).
"""
from __future__ import annotations

import os

from ..models import Article, SearchPlan, norm_doi, norm_pmcid, norm_pmid
from .base import Provider, concept_terms, study_keywords

URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "title,year,venue,externalIds,openAccessPdf,isOpenAccess,citationCount,authors"
NATIVE_TYPES = {"rct": "ClinicalTrial", "clinical_trial": "ClinicalTrial", "meta_analysis": "MetaAnalysis",
                "review": "Review", "systematic_review": "Review", "case_report": "CaseReport"}
KEYWORD_ONLY = {"guideline", "systematic_review"}       # у S2 для них нет типа — добавляем как слова


def s2_term(t: str) -> str:
    if "*" in t:
        if " " in t or len(t.rstrip("*")) < 4:
            t = t.replace("*", "")
        else:
            return t
    return f'"{t}"' if " " in t or "-" in t else t


class SemanticScholar(Provider):
    name = "semanticscholar"
    label = "Semantic Scholar"

    def available(self):
        return True, "" if os.environ.get("S2_API_KEY") else "без S2_API_KEY действует общий лимит (возможны паузы и 429)"

    @staticmethod
    def type_plan(plan: SearchPlan) -> tuple[list[str], list[str]]:
        """(типы публикаций S2, типы, которые придётся искать словами). Типы внутри фильтра — OR,
        поэтому смесь «родных» и «словесных» типов целиком переводится на слова."""
        st = plan.filters.study_types
        if len(st) == 1 and st[0] in NATIVE_TYPES:
            return [NATIVE_TYPES[st[0]]], ([st[0]] if st[0] in KEYWORD_ONLY else [])
        if st and all(t in NATIVE_TYPES and t not in KEYWORD_ONLY for t in st):
            return list(dict.fromkeys(NATIVE_TYPES[t] for t in st)), []
        return [], list(st)

    def boolean_query(self, plan: SearchPlan) -> str:
        blocks = ["(" + " | ".join(s2_term(t) for t in concept_terms(c)) + ")" for c in plan.concepts if concept_terms(c)]
        kw = study_keywords(self.type_plan(plan)[1])
        if kw:
            blocks.append("(" + " | ".join(s2_term(t) for t in kw) + ")")
        q = " + ".join(blocks)
        for x in plan.exclude:
            q += f" -{s2_term(x)}"
        return q

    def _params(self, plan: SearchPlan, **kw) -> dict:
        f = plan.filters
        p = {"query": self.boolean_query(plan), "fields": kw.pop("fields", FIELDS),
             "fieldsOfStudy": os.environ.get("S2_FIELDS_OF_STUDY", "Medicine"),
             "sort": "citationCount:desc"}
        if f.year_from or f.year_to:
            p["year"] = f"{f.year_from or ''}-{f.year_to or ''}"
        native = self.type_plan(plan)[0]
        if native:
            p["publicationTypes"] = ",".join(native)
        if f.open_access_only:
            p["openAccessPdf"] = ""
        p.update(kw)
        return p

    def _headers(self) -> dict:
        return {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}

    def render(self, plan: SearchPlan) -> str:
        p = self._params(plan)
        skipped = [n for n, on in (("только люди", plan.filters.humans_only), ("язык", bool(plan.filters.languages)),
                                   ("есть аннотация", plan.filters.has_abstract)) if on]
        note = f"  [не поддерживается, пропущено: {', '.join(skipped)}]" if skipped else ""
        return f"query={p['query']} | " + ", ".join(f"{k}={v}" for k, v in p.items()
                                                    if k not in ("query", "fields")) + note

    def count(self, plan: SearchPlan) -> int:
        d = self.http.get_json(URL, params=self._params(plan, fields="paperId"), headers=self._headers())
        return int(d.get("total", 0))

    def search(self, plan: SearchPlan, limit: int) -> list[Article]:
        arts, token = [], None
        while len(arts) < limit:
            kw = {"token": token} if token else {}
            d = self.http.get_json(URL, params=self._params(plan, **kw), headers=self._headers())
            arts += [self.parse(x) for x in d.get("data", [])]
            token = d.get("token")
            if not token or not d.get("data"):
                break
        return arts[:limit]

    @staticmethod
    def parse(x: dict) -> Article:
        ext = x.get("externalIds") or {}
        pdf = (x.get("openAccessPdf") or {}).get("url")
        return Article(
            title=(x.get("title") or "").strip().rstrip("."), year=x.get("year"), journal=x.get("venue") or "",
            authors=[a.get("name", "") for a in (x.get("authors") or [])][:50],
            doi=norm_doi(ext.get("DOI")), pmid=norm_pmid(ext.get("PubMed")),
            pmcid=norm_pmcid(("PMC" + str(ext["PubMedCentral"])) if ext.get("PubMedCentral") else ""),
            is_oa=x.get("isOpenAccess"), oa_urls=[pdf] if pdf else [], cited_by=x.get("citationCount"),
        )

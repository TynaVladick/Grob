"""OpenAlex: открытый каталог ~250 млн работ, умеет булев поиск по названию и аннотации и знает ссылки на OA-PDF.

С 2026 г. работает по дневному бюджету: без ключа около $0.10/день, с бесплатным ключом $1/день
(ключ: openalex.org/settings/api, переменная OPENALEX_API_KEY).
"""
from __future__ import annotations

import logging
import os

import requests

from ..models import Article, SearchPlan, norm_doi, norm_pmcid, norm_pmid
from .base import Provider, concept_terms, study_keywords

log = logging.getLogger("litsearch.openalex")
URL = "https://api.openalex.org/works"
SELECT = ("id,doi,title,publication_year,primary_location,authorships,ids,open_access,"
          "best_oa_location,locations,cited_by_count")


def bool_term(t: str) -> str:
    if "*" in t:
        if " " in t or len(t.rstrip("*")) < 4:
            t = t.replace("*", "")
        else:
            return t
    if " " in t or t.upper() in ("AND", "OR", "NOT"):
        return f'"{t}"'
    return t


class OpenAlex(Provider):
    name = "openalex"
    label = "OpenAlex"

    def __init__(self, http):
        super().__init__(http)
        self._use_search_param = False        # запасной вариант, если фильтр title_and_abstract.search недоступен

    def available(self):
        if os.environ.get("OPENALEX_API_KEY"):
            return True, ""
        return True, "без OPENALEX_API_KEY дневной бюджет мал (~$0.10): получите бесплатный ключ на openalex.org/settings/api"

    # ---- запрос ------------------------------------------------------------
    def boolean_query(self, plan: SearchPlan) -> str:
        blocks = ["(" + " OR ".join(bool_term(t) for t in concept_terms(c, strip_commas=True)) + ")"
                  for c in plan.concepts if concept_terms(c)]
        sk = study_keywords(plan.filters.study_types)
        if sk:
            blocks.append("(" + " OR ".join(bool_term(t) for t in sk) + ")")
        q = " AND ".join(blocks)
        if plan.exclude:
            q = f"({q}) NOT (" + " OR ".join(bool_term(t.replace(",", " ")) for t in plan.exclude) + ")"
        return q

    def filters(self, plan: SearchPlan) -> list[str]:
        f, out = plan.filters, []
        if f.year_from:
            out.append(f"from_publication_date:{f.year_from}-01-01")
        if f.year_to:
            out.append(f"to_publication_date:{f.year_to}-12-31")
        out.append("type:article|review")
        if f.languages:
            out.append("language:" + "|".join(f.languages))
        if f.open_access_only:
            out.append("open_access.is_oa:true")
        if f.has_abstract:
            out.append("has_abstract:true")
        return out

    def render(self, plan: SearchPlan) -> str:
        note = "  [«только люди» в OpenAlex не поддерживается — фильтр пропущен]" if plan.filters.humans_only else ""
        return f"title_and_abstract.search:{self.boolean_query(plan)} | " + ", ".join(self.filters(plan)) + note

    def _params(self, plan: SearchPlan, **kw) -> dict:
        filt = self.filters(plan)
        q = self.boolean_query(plan)
        if self._use_search_param:
            p = {"search": q, "filter": ",".join(filt)}
        else:
            p = {"filter": ",".join([f"title_and_abstract.search:{q}", *filt])}
        if os.environ.get("OPENALEX_API_KEY"):
            p["api_key"] = os.environ["OPENALEX_API_KEY"]
        if os.environ.get("CONTACT_EMAIL") or os.environ.get("NCBI_EMAIL"):
            p["mailto"] = os.environ.get("CONTACT_EMAIL") or os.environ["NCBI_EMAIL"]
        p.update(kw)
        return p

    def _get(self, plan: SearchPlan, **kw) -> dict:
        try:
            return self.http.get_json(URL, params=self._params(plan, **kw))
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 400 and not self._use_search_param:
                log.warning("OpenAlex: фильтр title_and_abstract.search отклонён, пробую параметр search")
                self._use_search_param = True
                return self.http.get_json(URL, params=self._params(plan, **kw))
            raise

    def count(self, plan: SearchPlan) -> int:
        return int(self._get(plan, **{"per-page": 1, "select": "id"}).get("meta", {}).get("count", 0))

    def search(self, plan: SearchPlan, limit: int) -> list[Article]:
        arts, cursor = [], "*"
        while len(arts) < limit and cursor:
            d = self._get(plan, **{"per-page": min(100, limit - len(arts)), "select": SELECT, "cursor": cursor})
            arts += [self.parse(w) for w in d.get("results", [])]
            cursor = d.get("meta", {}).get("next_cursor")
            if not d.get("results"):
                break
        return arts[:limit]

    @staticmethod
    def parse(w: dict) -> Article:
        ids = w.get("ids") or {}
        oa = w.get("open_access") or {}
        best = w.get("best_oa_location") or {}
        urls = []
        for loc in [best, *(w.get("locations") or [])]:
            if loc and loc.get("is_oa") and loc.get("pdf_url") and loc["pdf_url"] not in urls:
                urls.append(loc["pdf_url"])
        if oa.get("oa_url") and oa["oa_url"] not in urls:
            urls.append(oa["oa_url"])                      # может быть страницей, а не PDF — проверяется при загрузке
        src = ((w.get("primary_location") or {}).get("source") or {})
        return Article(
            title=(w.get("title") or "").strip().rstrip("."), year=w.get("publication_year"),
            journal=src.get("display_name") or "",
            authors=[(a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])][:50],
            doi=norm_doi(w.get("doi") or ids.get("doi")), pmid=norm_pmid(ids.get("pmid")),
            pmcid=norm_pmcid(ids.get("pmcid")), is_oa=oa.get("is_oa"), oa_urls=urls,
            license=best.get("license") or "", cited_by=w.get("cited_by_count"),
        )

"""Тесты пакета litsearch: без сети и без ключей ИИ (HTTP и LLM подменяются).
Запуск: python -m pytest test_litsearch.py -q     или просто:  python test_litsearch.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import requests

from litsearch import dialog
from litsearch.downloader import Downloader
from litsearch.llm import LLM, build_plan, refine_plan
from litsearch.models import Article, Concept, Filters, SearchPlan, merge_articles
from litsearch.providers.base import Provider
from litsearch.providers.europepmc import EuropePMC
from litsearch.providers.openalex import OpenAlex
from litsearch.providers.pubmed import PubMed
from litsearch.providers.semanticscholar import SemanticScholar

PDF = b"%PDF-1.4\n" + b"x" * 6000


def make_plan(**filters) -> SearchPlan:
    return SearchPlan(
        original="метформин и деменция",
        concepts=[Concept("metformin", ["Metformin"], ["metformin", "glucophage"]),
                  Concept("dementia", ["Dementia"], ["dementia", "alzheimer*", "cognitive decline"])],
        filters=Filters(**filters))


class Resp:
    def __init__(self, status=200, data=None, content=b"", headers=None):
        self.status_code, self._data, self.content = status, data, content
        self.headers = headers or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def close(self):
        pass


class FakeHttp:
    """routes: [(подстрока URL, Resp | функция(url, params) -> Resp)]; первое совпадение выигрывает."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, headers=None, stream=False, timeout=60, tries=4):
        self.calls.append(url)
        for sub, h in self.routes:
            if sub in url:
                return h(url, params) if callable(h) else h
        return Resp(404)

    def get_json(self, url, params=None, headers=None, timeout=60):
        r = self.get(url, params, headers)
        r.raise_for_status()
        return r.json()


# --------------------------------------------------------------------------- #
# план и запросы
# --------------------------------------------------------------------------- #
def test_plan_sanitizing():
    plan = SearchPlan.from_dict({
        "interpretation_ru": "Метформин и деменция",
        "concepts": [{"label": "A", "mesh": ['Metformin', 'metformin', '  '], "keywords": ['met"formin', "x[tiab]"]},
                     {"label": "empty", "mesh": [], "keywords": []}],
        "filters": {"year_from": 2026, "year_to": 2020, "study_types": ["rct", "nonsense"],
                    "languages": ["EN", "xx"], "open_access_only": 1}}, original="q")
    assert len(plan.concepts) == 1 and plan.concepts[0].mesh == ["Metformin"]
    assert plan.concepts[0].keywords == ["met formin", "x tiab"]
    f = plan.filters
    assert (f.year_from, f.year_to) == (2020, 2026)            # перепутанные годы поменялись местами
    assert f.study_types == ["rct"] and f.languages == ["en"] and f.open_access_only is True


def test_pubmed_query():
    q = PubMed(None).render(make_plan(year_from=2021, year_to=2026, study_types=["rct", "meta_analysis"],
                                      humans_only=True, languages=["en"], open_access_only=True, has_abstract=True))
    assert q == (
        '("Metformin"[MeSH Terms] OR "metformin"[tiab] OR "glucophage"[tiab]) AND '
        '("Dementia"[MeSH Terms] OR "dementia"[tiab] OR alzheimer*[tiab] OR "cognitive decline"[tiab]) '
        'AND 2021:2026[dp] AND (randomized controlled trial[pt] OR meta-analysis[pt]) AND humans[mh] '
        'AND (english[la]) AND free full text[sb] AND hasabstract[text]')


def test_pubmed_wildcard_rules():
    from litsearch.providers.pubmed import tiab
    assert tiab("alzheimer*") == "alzheimer*[tiab]"
    assert tiab("ab*") == '"ab"[tiab]'                         # слишком короткая основа — без усечения
    assert tiab("heart fail*") == '"heart fail"[tiab]'         # в фразе усечение не работает


def test_other_renderers():
    plan = make_plan(year_from=2021, year_to=2026, study_types=["rct"], open_access_only=True, humans_only=True)
    e = EuropePMC(None).render(plan)
    assert 'MESH:"Metformin"' in e and 'TITLE:"metformin"' in e and "ABSTRACT:alzheimer*" in e
    assert "PUB_YEAR:[2021 TO 2026]" in e and 'PUB_TYPE:"Randomized Controlled Trial"' in e
    assert "OPEN_ACCESS:y" in e and e.endswith("NOT SRC:PPR")

    oa = OpenAlex(None)
    bq = oa.boolean_query(plan)
    assert '(Metformin OR glucophage)' in bq and "alzheimer*" in bq and '"cognitive decline"' in bq
    assert '"randomized controlled trial"' in bq                 # тип исследования добавлен словами
    flt = oa.filters(plan)
    assert "from_publication_date:2021-01-01" in flt and "to_publication_date:2026-12-31" in flt
    assert "open_access.is_oa:true" in flt and "type:article|review" in flt
    assert "не поддерживается" in oa.render(plan)               # «только люди» честно помечено как пропущенное

    s2 = SemanticScholar(None)
    p = s2._params(plan)
    assert p["publicationTypes"] == "ClinicalTrial" and p["year"] == "2021-2026" and p["openAccessPdf"] == ""
    assert '"cognitive decline"' in p["query"] and " + " in p["query"]
    # смесь «родных» и «словесных» типов целиком переводится на слова
    mixed = make_plan(study_types=["rct", "guideline"])
    assert s2.type_plan(mixed) == ([], ["rct", "guideline"])
    assert s2.type_plan(make_plan(study_types=["systematic_review"])) == (["Review"], ["systematic_review"])


def test_comma_in_mesh_for_openalex():
    plan = SearchPlan(concepts=[Concept("t2d", ["Diabetes Mellitus, Type 2"], ["T2DM"])])
    assert "," not in OpenAlex(None).boolean_query(plan)       # запятая разделяет фильтры в OpenAlex
    assert '"Diabetes Mellitus, Type 2"[MeSH Terms]' in PubMed(None).render(plan)


# --------------------------------------------------------------------------- #
# разбор ответов баз
# --------------------------------------------------------------------------- #
def test_parsers():
    a = PubMed.parse_summary("111", {"title": "Title one.", "pubdate": "2023 Jan 5", "source": "J Test",
                                     "authors": [{"name": "Doe J"}],
                                     "articleids": [{"idtype": "doi", "value": "10.1/ABC"},
                                                    {"idtype": "pmc", "value": "PMC999"}]})
    assert (a.title, a.year, a.doi, a.pmid, a.pmcid) == ("Title one", 2023, "10.1/abc", "111", "PMC999")

    e = EuropePMC.parse({"id": "222", "source": "MED", "pmid": "222", "pmcid": "PMC5", "doi": "10.2/x",
                         "title": "T", "pubYear": "2022", "isOpenAccess": "Y", "authorString": "A B, C D.",
                         "citedByCount": 7})
    assert e.is_oa is True and "PMC5" in e.oa_urls[0] and e.authors == ["A B", "C D"] and e.cited_by == 7

    w = OpenAlex.parse({"title": "T", "publication_year": 2021, "doi": "https://doi.org/10.3/Y",
                        "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/333", "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7"},
                        "open_access": {"is_oa": True, "oa_url": "https://x.org/landing"},
                        "best_oa_location": {"is_oa": True, "pdf_url": "https://x.org/a.pdf", "license": "cc-by"},
                        "locations": [], "authorships": [{"author": {"display_name": "Z"}}]})
    assert (w.doi, w.pmid, w.pmcid) == ("10.3/y", "333", "PMC7") and w.oa_urls[0] == "https://x.org/a.pdf"

    s = SemanticScholar.parse({"title": "T", "year": 2020, "externalIds": {"DOI": "10.4/z", "PubMedCentral": "88"},
                               "openAccessPdf": {"url": "https://s.org/p.pdf"}, "isOpenAccess": True})
    assert s.pmcid == "PMC88" and s.oa_urls == ["https://s.org/p.pdf"]


def test_merge_dedupe():
    pm = [Article(title="Metformin and dementia risk in older adults: a cohort", doi="10.1/A", pmid="1"),
          Article(title="Only in pubmed, something else entirely different here", pmid="2")]
    oa = [Article(title="Only in openalex, another unrelated study title", doi="10.9/Z"),
          Article(title="Metformin and dementia risk in older adults: a cohort", doi="https://doi.org/10.1/a",
                  is_oa=True, oa_urls=["https://x/a.pdf"])]
    ep = [Article(title="metformin and dementia risk in older adults - a cohort!", pmid="1", pmcid="PMC5")]
    merged = merge_articles({"pubmed": pm, "openalex": oa, "europepmc": ep})
    assert len(merged) == 3
    top = merged[0]                                            # найдена в трёх базах -> выше всех (RRF)
    assert sorted(top.sources) == ["europepmc", "openalex", "pubmed"]
    assert top.pmcid == "PMC5" and top.is_oa is True and top.oa_urls == ["https://x/a.pdf"]


# --------------------------------------------------------------------------- #
# MeSH-проверка
# --------------------------------------------------------------------------- #
def test_validate_mesh_drops_unknown():
    def esearch(url, params):
        n = 0 if "Bogus" in params["term"] else 5
        return Resp(data={"esearchresult": {"count": str(n), "idlist": []}})

    pm = PubMed(FakeHttp([("esearch", esearch)]))
    plan = SearchPlan(concepts=[Concept("x", ["Metformin", "Bogus Heading"], ["metformin"])])
    assert pm.validate_mesh(plan) == ["Bogus Heading"]
    assert plan.concepts[0].mesh == ["Metformin"] and "Bogus Heading" in plan.concepts[0].keywords


def test_validate_mesh_keeps_on_network_error():
    http = FakeHttp([("esearch", Resp(500))])
    plan = SearchPlan(concepts=[Concept("x", ["Metformin", "Dementia"], [])])
    assert PubMed(http).validate_mesh(plan) == [] and plan.concepts[0].mesh == ["Metformin", "Dementia"]
    assert len(http.calls) == 1                                  # после первой сетевой ошибки больше не стучимся


def test_openalex_falls_back_to_search_param_on_400():
    calls = []

    def handler(url, params):
        calls.append(params)
        return Resp(400) if "filter" in params and "title_and_abstract" in params["filter"] else \
            Resp(data={"meta": {"count": 42}})

    oa = OpenAlex(FakeHttp([("openalex", handler)]))
    assert oa.count(make_plan()) == 42 and oa._use_search_param and "search" in calls[-1]


# --------------------------------------------------------------------------- #
# ИИ (HTTP подменён)
# --------------------------------------------------------------------------- #
RAW_PLAN = {"interpretation_ru": "Метформин и деменция", "comment_ru": "добавил РКИ",
            "concepts": [{"label": "metformin", "mesh": ["Metformin"], "keywords": ["metformin"]}],
            "filters": {"study_types": ["rct"]}}


def test_llm_anthropic_tool_call():
    resp = Resp(data={"content": [{"type": "text", "text": "..."}, {"type": "tool_use", "input": RAW_PLAN}]})
    with mock.patch("litsearch.llm.requests.post", return_value=resp) as post:
        plan = build_plan(LLM(provider="anthropic", api_key="k"), "метформин")
    body = post.call_args.kwargs["json"]
    assert body["tool_choice"] == {"type": "tool", "name": "submit_plan"} and post.call_args.kwargs["headers"]["x-api-key"] == "k"
    assert plan.original == "метформин" and plan.filters.study_types == ["rct"]


def test_llm_openai_tool_call_and_refine():
    resp = Resp(data={"choices": [{"message": {"tool_calls": [{"function": {"arguments": json.dumps(RAW_PLAN)}}]}}]})
    llm = LLM(provider="openai", api_key="k", model="m")
    with mock.patch("litsearch.llm.requests.post", return_value=resp):
        plan, comment = refine_plan(llm, make_plan(), "только РКИ")
    assert comment == "добавил РКИ" and plan.original == "метформин и деменция"   # исходный запрос сохраняется
    assert plan.filters.study_types == ["rct"]


# --------------------------------------------------------------------------- #
# диалог
# --------------------------------------------------------------------------- #
class FakeProvider(Provider):
    name, label = "fake", "Fake"

    def render(self, plan):
        return "q"

    def count(self, plan):
        f, n = plan.filters, 10000
        if f.year_from:
            n //= 4
        if f.open_access_only:
            n //= 2
        if f.humans_only:
            n = n * 2 // 3
        if f.study_types:
            n //= 20
        return n // max(1, len(plan.concepts) - 1) if len(plan.concepts) > 1 else n * 3


def test_dialog_auto_converges_into_range():
    out = []
    plan = dialog.run_dialog(make_plan(), [FakeProvider(None)], 20, 300, auto=True, say=out.append)
    assert FakeProvider(None).count(plan) in range(20, 301)
    f = plan.filters
    assert f.year_from and f.open_access_only and f.humans_only          # сначала мягкие ограничения...
    assert f.study_types == ["meta_analysis", "systematic_review"]       # ...тип исследований — только когда иначе не хватило


def test_dialog_auto_relaxes_when_too_few():
    plan = make_plan(year_from=2025, study_types=["rct"], humans_only=True, open_access_only=True)
    out = []
    plan = dialog.run_dialog(plan, [FakeProvider(None)], 200, 3000, auto=True, say=out.append)
    assert 200 <= FakeProvider(None).count(plan) <= 3000


def test_dialog_interactive_numbers_and_free_text():
    answers = iter(["1", "только РКИ", "ok"])
    prompts = []

    def ask(p):
        prompts.append(p)
        return next(answers)

    class Stub:
        def call_tool(self, *a, **k):
            return RAW_PLAN

    plan = dialog.run_dialog(make_plan(), [FakeProvider(None)], 20, 300, llm=Stub(), ask=ask, say=lambda s: None)
    assert len(prompts) == 3 and plan.filters.study_types == ["rct"]     # правка ИИ применена, концепция метформина осталась
    assert dialog.run_dialog(make_plan(), [FakeProvider(None)], ask=lambda p: "q", say=lambda s: None) is None


# --------------------------------------------------------------------------- #
# скачивание
# --------------------------------------------------------------------------- #
S3_LIST = (b'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
           b'<CommonPrefixes><Prefix>PMC5.1/</Prefix></CommonPrefixes><CommonPrefixes><Prefix>PMC5.2/</Prefix></CommonPrefixes>'
           b'</ListBucketResult>')


def s3_routes(license_code="cc_by", md5=None):
    import hashlib
    md5 = md5 or hashlib.md5(PDF).hexdigest()
    meta = {"is_retracted": "no", "license_code": license_code,
            "pdf_url": f"s3://pmc-oa-opendata/PMC5.2/PMC5.2.pdf?md5={md5}"}
    return [("PMC5.2/PMC5.2.json", Resp(data=meta)), ("PMC5.2/PMC5.2.pdf", Resp(content=PDF)),
            ("pmc-oa-opendata.s3.amazonaws.com/", lambda u, p: Resp(content=S3_LIST) if p else Resp(404))]


def run_dl(routes, article, **kw):
    with tempfile.TemporaryDirectory() as d:
        http = FakeHttp(routes)
        dl = Downloader(http, Path(d), email=kw.pop("email", "me@example.org"), **kw)
        res = dl.download(article)
        leftovers = [p.name for p in Path(d).iterdir() if p.suffix != ".pdf"]
        return res, http, leftovers


def test_download_pmc_s3_latest_version_with_md5():
    res, http, left = run_dl(s3_routes(), Article(title="t", pmcid="PMC5", doi="10.1/a"))
    assert res.status == "ok" and res.source == "pmc_s3" and res.bytes == len(PDF) and res.license == "cc_by" and not left
    assert not any("unpaywall" in u for u in http.calls)                 # дальше по каскаду не идём


def test_download_md5_mismatch_falls_through():
    routes = s3_routes(md5="0" * 32) + [("ptpmcrender", Resp(content=PDF))]
    res, _, left = run_dl(routes, Article(title="t", pmcid="PMC5"))
    assert res.status == "ok" and res.source == "europepmc" and not left


def test_download_license_filter():
    routes = s3_routes(license_code="cc_by_nc") + [("ptpmcrender", Resp(404))]
    res, _, _ = run_dl(routes, Article(title="t", pmcid="PMC5"), allowed_licenses={"CC BY"})
    assert res.status == "failed" and "лицензия" in res.error


def test_download_skips_html_and_uses_next_source():
    html = Resp(content=b"<html>" + b"x" * 9000, headers={"Content-Type": "text/html"})
    routes = [("publisher.org/landing", html), ("good.org/a.pdf", Resp(content=PDF))]
    a = Article(title="t", oa_urls=["https://publisher.org/landing", "https://good.org/a.pdf"])
    res, _, left = run_dl(routes, a)
    assert res.status == "ok" and res.source == "search_result" and res.url.endswith("a.pdf") and not left


def test_download_unpaywall_by_doi_and_reuse_existing():
    routes = [("api.unpaywall.org", Resp(data={"best_oa_location": {"url_for_pdf": "https://oa.org/p.pdf", "license": "cc-by"},
                                                "oa_locations": []})), ("oa.org/p.pdf", Resp(content=PDF))]
    with tempfile.TemporaryDirectory() as d:
        dl = Downloader(FakeHttp(routes), Path(d), email="me@example.org")
        a = Article(title="t", doi="10.1/a")
        assert dl.download(a).source == "unpaywall"
        again = dl.download(a)
        assert again.status == "exists" and (Path(d) / "doi_10.1_a.pdf").exists()
    res, _, _ = run_dl(routes, Article(title="t", doi="10.1/a"), email="")      # без email Unpaywall не вызывается
    assert res.status == "failed" and "нет ни одного источника" in res.error


def test_download_many_replaces_failures():
    good = Resp(content=PDF)
    routes = [("good", good)]
    arts = [Article(title=f"t{i}", pmid=str(i), oa_urls=["https://good.org/x" if i % 2 else "https://bad.org/x"])
            for i in range(10)]
    with tempfile.TemporaryDirectory() as d:
        dl = Downloader(FakeHttp(routes), Path(d), email="")
        results = dl.download_many(arts, target=3, workers=2)
        assert sum(r.success for r in results.values()) == 3
        assert len(list(Path(d).glob("*.pdf"))) == 3


if __name__ == "__main__":                       # запуск без pytest
    import sys
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception:                    # noqa: BLE001
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if failed else 0)

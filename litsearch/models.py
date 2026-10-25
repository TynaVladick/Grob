"""Модели данных: фильтры, структурный план поиска, статья.

План поиска (SearchPlan) — единое «нейтральное» описание запроса: понятия (MeSH + слова),
исключения и фильтры. Из него детерминированным кодом собираются запросы для каждой базы
(PubMed, Europe PMC, OpenAlex, Semantic Scholar), поэтому синтаксические ошибки LLM
в итоговый запрос не попадают.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import asdict, dataclass, field

STUDY_TYPES = {
    "rct": "рандомизированные контролируемые исследования",
    "clinical_trial": "клинические исследования",
    "meta_analysis": "метаанализы",
    "systematic_review": "систематические обзоры",
    "review": "обзоры",
    "guideline": "клинические рекомендации",
    "case_report": "клинические случаи",
}

# ISO 639-1 -> (название для PubMed [la], код для Europe PMC LANG:)
LANGUAGES = {
    "en": ("english", "eng"), "ru": ("russian", "rus"), "de": ("german", "ger"),
    "fr": ("french", "fre"), "es": ("spanish", "spa"), "zh": ("chinese", "chi"),
}

_BAD_CHARS = re.compile(r'[\[\]"“”(){}:;|\\]')


def clean_term(s) -> str:
    """Убирает символы, ломающие синтаксис запросов (скобки, кавычки, двоеточия). Запятые и * остаются."""
    s = _BAD_CHARS.sub(" ", str(s or ""))
    return re.sub(r"\s+", " ", s).strip()


def uniq(items) -> list[str]:
    seen, out = set(), []
    for x in items:
        x = clean_term(x)
        if x and x.casefold() not in seen:
            seen.add(x.casefold())
            out.append(x)
    return out


def norm_doi(s) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", s)
    return s


def norm_pmcid(s) -> str:
    m = re.search(r"PMC\s*(\d+)", str(s or ""), re.I)
    return f"PMC{m.group(1)}" if m else ""


def norm_pmid(s) -> str:
    m = re.search(r"\d+", str(s or ""))
    return m.group(0) if m else ""


# --------------------------------------------------------------------------- #
@dataclass
class Filters:
    year_from: int | None = None
    year_to: int | None = None
    study_types: list[str] = field(default_factory=list)   # между собой через OR
    humans_only: bool = False
    languages: list[str] = field(default_factory=list)     # коды ISO 639-1, OR
    open_access_only: bool = False
    has_abstract: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> "Filters":
        d = d or {}

        def year(v):
            try:
                v = int(v)
            except (TypeError, ValueError):
                return None
            return v if 1800 <= v <= 2100 else None

        yf, yt = year(d.get("year_from")), year(d.get("year_to"))
        if yf and yt and yf > yt:
            yf, yt = yt, yf
        return cls(
            year_from=yf, year_to=yt,
            study_types=[t for t in uniq(d.get("study_types") or []) if t in STUDY_TYPES],
            humans_only=bool(d.get("humans_only")),
            languages=[x for x in (s.lower() for s in uniq(d.get("languages") or [])) if x in LANGUAGES],
            open_access_only=bool(d.get("open_access_only")),
            has_abstract=bool(d.get("has_abstract")),
        )

    def describe(self) -> str:
        parts = []
        if self.year_from or self.year_to:
            parts.append(f"годы {self.year_from or '…'}–{self.year_to or '…'}")
        if self.study_types:
            parts.append("типы: " + ", ".join(STUDY_TYPES[t] for t in self.study_types))
        if self.humans_only:
            parts.append("только люди")
        if self.languages:
            parts.append("языки: " + ", ".join(self.languages))
        if self.open_access_only:
            parts.append("открытый доступ")
        if self.has_abstract:
            parts.append("есть аннотация")
        return "; ".join(parts) or "нет"


@dataclass
class Concept:
    label: str
    mesh: list[str] = field(default_factory=list)       # точные заголовки MeSH
    keywords: list[str] = field(default_factory=list)   # свободный текст (заголовок/аннотация)


@dataclass
class SearchPlan:
    original: str = ""
    interpretation: str = ""
    concepts: list[Concept] = field(default_factory=list)   # между понятиями AND
    exclude: list[str] = field(default_factory=list)        # NOT
    filters: Filters = field(default_factory=Filters)

    @classmethod
    def from_dict(cls, d: dict, original: str = "") -> "SearchPlan":
        concepts = []
        for c in (d.get("concepts") or [])[:5]:
            mesh = uniq(c.get("mesh") or [])[:6]
            kws = uniq(c.get("keywords") or [])[:10]
            if mesh or kws:
                concepts.append(Concept(label=clean_term(c.get("label")) or (kws or mesh)[0],
                                        mesh=mesh, keywords=kws))
        return cls(original=original or d.get("original", ""),
                   interpretation=clean_term(d.get("interpretation_ru") or d.get("interpretation")),
                   concepts=concepts, exclude=uniq(d.get("exclude_keywords") or d.get("exclude") or [])[:8],
                   filters=Filters.from_dict(d.get("filters")))

    def to_dict(self) -> dict:
        return asdict(self)

    def clone(self) -> "SearchPlan":
        return copy.deepcopy(self)

    def signature(self) -> str:
        import json
        d = self.to_dict()
        d.pop("interpretation", None)
        return json.dumps(d, sort_keys=True, ensure_ascii=False)


# --------------------------------------------------------------------------- #
@dataclass
class Article:
    title: str = ""
    year: int | None = None
    journal: str = ""
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    sources: list[str] = field(default_factory=list)
    ranks: dict[str, int] = field(default_factory=dict)     # источник -> позиция в его выдаче
    oa_urls: list[str] = field(default_factory=list)        # ссылки на открытый PDF от баз поиска
    is_oa: bool | None = None
    license: str = ""
    cited_by: int | None = None

    def keys(self) -> list[str]:
        ks = []
        if self.doi:
            ks.append("doi:" + norm_doi(self.doi))
        if self.pmid:
            ks.append("pmid:" + self.pmid)
        if self.pmcid:
            ks.append("pmcid:" + self.pmcid)
        t = title_key(self.title)
        if len(t) >= 25:
            ks.append("title:" + t)
        return ks

    def absorb(self, other: "Article") -> None:
        for f in ("title", "year", "journal", "doi", "pmid", "pmcid", "license"):
            if not getattr(self, f) and getattr(other, f):
                setattr(self, f, getattr(other, f))
        if not self.authors:
            self.authors = other.authors
        for s in other.sources:
            if s not in self.sources:
                self.sources.append(s)
        self.ranks.update({k: v for k, v in other.ranks.items() if k not in self.ranks})
        for u in other.oa_urls:
            if u not in self.oa_urls:
                self.oa_urls.append(u)
        if other.is_oa is True or self.is_oa is None:
            self.is_oa = other.is_oa if other.is_oa is not None else self.is_oa
        if other.cited_by is not None:
            self.cited_by = max(self.cited_by or 0, other.cited_by)

    def rrf_score(self) -> float:
        """Reciprocal rank fusion: статья, высоко стоящая в нескольких базах, оказывается выше."""
        return sum(1.0 / (60 + r) for r in self.ranks.values())


def title_key(title: str) -> str:
    t = unicodedata.normalize("NFKD", (title or "").casefold())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9а-яё]+", "", t)[:120]


def merge_articles(per_source: dict[str, list[Article]]) -> list[Article]:
    """Склеивает дубликаты между базами (по DOI → PMID → PMCID → названию) и сортирует по RRF."""
    index: dict[str, Article] = {}
    out: list[Article] = []
    for src, arts in per_source.items():
        for rank, a in enumerate(arts, 1):
            a.doi, a.pmid, a.pmcid = norm_doi(a.doi), norm_pmid(a.pmid), norm_pmcid(a.pmcid)
            a.sources, a.ranks = [src], {src: rank}
            hit = next((index[k] for k in a.keys() if k in index), None)
            if hit is None:
                out.append(a)
                hit = a
            else:
                hit.absorb(a)
            for k in hit.keys():
                index[k] = hit
    out.sort(key=lambda a: a.rrf_score(), reverse=True)
    return out

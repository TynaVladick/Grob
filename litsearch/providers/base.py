"""Базовый класс провайдера поиска и общие помощники для сборки запросов."""
from __future__ import annotations

from ..models import Article, Concept, SearchPlan, uniq


class ProviderError(RuntimeError):
    pass


class Provider:
    name = ""
    label = ""

    def __init__(self, http):
        self.http = http

    def available(self) -> tuple[bool, str]:
        """(можно ли использовать, пояснение для пользователя)."""
        return True, ""

    def render(self, plan: SearchPlan) -> str:
        """Человекочитаемый запрос (то, что реально уходит в базу)."""
        raise NotImplementedError

    def count(self, plan: SearchPlan) -> int | None:
        raise NotImplementedError

    def search(self, plan: SearchPlan, limit: int) -> list[Article]:
        raise NotImplementedError


def concept_terms(c: Concept, strip_commas: bool = False) -> list[str]:
    """Названия MeSH + свободные слова одним списком (для баз, где MeSH не поддерживается)."""
    terms = uniq([*c.mesh, *c.keywords])
    if strip_commas:                                   # «Diabetes Mellitus, Type 2» -> «Diabetes Mellitus Type 2»
        terms = uniq(t.replace(",", " ") for t in terms)
    return terms


# Слова для баз без типа публикации «РКИ/метаанализ» (OpenAlex): подмешиваются в запрос как ещё одно понятие.
STUDY_KEYWORDS = {
    "rct": ["randomized controlled trial", "randomised controlled trial", "randomized trial", "randomised trial", "RCT"],
    "clinical_trial": ["clinical trial"],
    "meta_analysis": ["meta-analysis", "meta analysis", "metaanalysis"],
    "systematic_review": ["systematic review"],
    "review": ["review"],
    "guideline": ["guideline*", "consensus statement", "practice guideline"],
    "case_report": ["case report"],
}


def study_keywords(types: list[str]) -> list[str]:
    return uniq(k for t in types for k in STUDY_KEYWORDS.get(t, []))

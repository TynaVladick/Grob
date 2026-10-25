from __future__ import annotations

from .base import Provider, ProviderError
from .europepmc import EuropePMC
from .openalex import OpenAlex
from .pubmed import PubMed
from .semanticscholar import SemanticScholar

REGISTRY = {c.name: c for c in (PubMed, EuropePMC, OpenAlex, SemanticScholar)}


def make_providers(names: list[str], http) -> list[Provider]:
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Неизвестные источники: {', '.join(unknown)}. Доступны: {', '.join(REGISTRY)}")
    return [REGISTRY[n](http) for n in names]


__all__ = ["Provider", "ProviderError", "REGISTRY", "make_providers"]

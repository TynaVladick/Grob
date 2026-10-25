"""ИИ-часть: из простого запроса на русском строит структурный план поиска (SearchPlan),
а по реплике пользователя в диалоге исправляет этот план.

LLM возвращает не готовую строку запроса, а JSON через tool calling (структура гарантируется схемой).
Строка для каждой базы потом собирается кодом — это исключает синтаксические ошибки.

Настройка через переменные окружения:
    ANTHROPIC_API_KEY   ключ Anthropic (провайдер по умолчанию)      LLM_MODEL (по умолчанию claude-sonnet-5-5)
    OPENAI_API_KEY      ключ OpenAI-совместимого API (+ LLM_MODEL, LLM_BASE_URL при необходимости)
    LLM_PROVIDER        anthropic | openai  (если не задан — по тому, какой ключ найден)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date

import requests

from .models import STUDY_TYPES, LANGUAGES, SearchPlan

log = logging.getLogger("litsearch.llm")


class LLMError(RuntimeError):
    pass


FILTERS_SCHEMA = {
    "type": "object",
    "description": "Фильтры. Заполняй только то, что пользователь явно просил или подразумевал.",
    "properties": {
        "year_from": {"type": "integer"},
        "year_to": {"type": "integer"},
        "study_types": {"type": "array", "items": {"type": "string", "enum": list(STUDY_TYPES)}},
        "humans_only": {"type": "boolean"},
        "languages": {"type": "array", "items": {"type": "string", "enum": list(LANGUAGES)}},
        "open_access_only": {"type": "boolean"},
        "has_abstract": {"type": "boolean"},
    },
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretation_ru": {"type": "string", "description": "Одно предложение по-русски: как понят запрос."},
        "comment_ru": {"type": "string", "description": "Только при правке: что именно изменено (по-русски)."},
        "concepts": {
            "type": "array", "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Короткое название понятия (по-английски)."},
                    "mesh": {"type": "array", "items": {"type": "string"},
                             "description": "Точные заголовки MeSH, в существовании которых ты уверен; иначе []."},
                    "keywords": {"type": "array", "items": {"type": "string"},
                                 "description": "3–8 синонимов/сокращений для поиска в названии и аннотации."},
                },
                "required": ["label", "keywords"],
            },
        },
        "exclude_keywords": {"type": "array", "items": {"type": "string"}},
        "filters": FILTERS_SCHEMA,
    },
    "required": ["interpretation_ru", "concepts"],
}

SYSTEM = """You are an expert medical librarian who designs PubMed/MEDLINE search strategies (PICO, MeSH, Boolean logic).
The user writes a short, informal question in Russian. Produce a structured search plan by calling the tool.

Rules:
1. Think in English: translate medical terms; use standard terminology plus common synonyms and abbreviations
   (e.g. "type 2 diabetes", "T2DM"); add US/UK spelling variants when they differ.
2. Split the question into 2-4 concepts (population/condition, intervention/exposure, outcome...). Concepts are
   combined with AND, so each one must be essential. Do not invent a concept the user did not ask about.
   Prefer fewer concepts: the user will narrow the search later in a dialogue.
3. For each concept: "mesh" = official MeSH descriptors you are CERTAIN exist (else empty list);
   "keywords" = free-text synonyms. Use * truncation only on single words with >= 4 letters before the star
   (cardiovascular*), never inside multi-word phrases.
4. Study design (RCT, meta-analysis, ...), years, language, humans, open access are FILTERS, never concepts.
   Fill filters ONLY if the user clearly asked or implied it ("за последние 5 лет" -> year_from = current year - 4;
   "рандомизированные" -> study_types ["rct"]). Otherwise leave filters empty.
5. interpretation_ru: one short Russian sentence on how you understood the question.
Today is {today}."""

REFINE_SUFFIX = """
You are now EDITING an existing plan. Apply the user's instruction to the current plan and return the FULL updated plan.
Change only what the user asked for; keep everything else as is. To drop a filter, omit it. Put a short Russian
description of what you changed in comment_ru."""


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, base_url: str | None = None):
        a_key, o_key = os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("OPENAI_API_KEY")
        self.provider = (provider or os.environ.get("LLM_PROVIDER") or
                         ("anthropic" if a_key or not o_key else "openai")).lower()
        if self.provider not in ("anthropic", "openai"):
            raise LLMError(f"Неизвестный LLM_PROVIDER: {self.provider} (нужен anthropic или openai)")
        self.api_key = api_key or (a_key if self.provider == "anthropic" else o_key)
        self.model = model or os.environ.get("LLM_MODEL") or (
            "claude-sonnet-5-5" if self.provider == "anthropic" else "")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or (
            "https://api.anthropic.com" if self.provider == "anthropic" else "https://api.openai.com/v1")).rstrip("/")
        if not self.api_key:
            raise LLMError("Не задан ключ ИИ: установите ANTHROPIC_API_KEY (или OPENAI_API_KEY + LLM_MODEL), "
                           "либо передайте готовый план через --plan-file.")
        if not self.model:
            raise LLMError("Для провайдера openai задайте модель в переменной LLM_MODEL.")

    # ------------------------------------------------------------------ #
    def call_tool(self, system: str, user: str, name: str, schema: dict, max_tokens: int = 2000) -> dict:
        if self.provider == "anthropic":
            url = f"{self.base_url}/v1/messages"
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
            body = {"model": self.model, "max_tokens": max_tokens, "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "tools": [{"name": name, "description": "Submit the structured result.", "input_schema": schema}],
                    "tool_choice": {"type": "tool", "name": name}}
        else:
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            body = {"model": self.model, "max_tokens": max_tokens,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "tools": [{"type": "function", "function": {"name": name, "description": "Submit the structured result.",
                                                                "parameters": schema}}],
                    "tool_choice": {"type": "function", "function": {"name": name}}}
        data = self._post(url, headers, body)
        try:
            if self.provider == "anthropic":
                return next(b["input"] for b in data["content"] if b.get("type") == "tool_use")
            args = data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            return json.loads(args) if isinstance(args, str) else args
        except (KeyError, IndexError, StopIteration, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"Неожиданный ответ модели: {str(data)[:300]}") from exc

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        last = ""
        for i in range(3):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=120)
            except requests.RequestException as exc:
                last = str(exc)
            else:
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (408, 429, 500, 502, 503, 529):
                    break
            time.sleep(2 ** (i + 1))
        raise LLMError(f"Запрос к ИИ не удался: {last}")


# --------------------------------------------------------------------------- #
def build_plan(llm: LLM, user_query: str) -> SearchPlan:
    raw = llm.call_tool(SYSTEM.format(today=date.today().isoformat()),
                        f"Запрос пользователя: {user_query}", "submit_plan", PLAN_SCHEMA)
    plan = SearchPlan.from_dict(raw, original=user_query)
    if not plan.concepts:
        raise LLMError("Модель не выделила ни одного понятия для поиска. Переформулируйте запрос.")
    return plan


def refine_plan(llm: LLM, plan: SearchPlan, instruction: str) -> tuple[SearchPlan, str]:
    current = json.dumps(plan.to_dict(), ensure_ascii=False, indent=1)
    raw = llm.call_tool(SYSTEM.format(today=date.today().isoformat()) + REFINE_SUFFIX,
                        f"Исходный вопрос: {plan.original}\n\nТекущий план:\n{current}\n\nПравка пользователя: {instruction}",
                        "submit_plan", PLAN_SCHEMA)
    new = SearchPlan.from_dict(raw, original=plan.original)
    if not new.concepts:
        raise LLMError("После правки не осталось ни одного понятия — правка отклонена.")
    return new, str(raw.get("comment_ru") or "").strip()

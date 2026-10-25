"""Диалог настройки фильтров: подбираем запрос так, чтобы статей было «в самый раз».

Каждый шаг: считаем число результатов (дешёвый запрос count), если вне целевого диапазона —
предлагаем варианты сузить/расширить запрос и сразу показываем, сколько статей получится.
Пользователь выбирает номер(а) или пишет по-русски, что поменять (это обрабатывает ИИ).
Режим auto делает то же без вопросов (для скриптов и непустого терминала).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable

from .llm import LLM, LLMError, refine_plan
from .models import SearchPlan
from .providers.base import Provider

log = logging.getLogger("litsearch.dialog")


@dataclass
class Option:
    label: str
    apply: Callable[[SearchPlan], None]
    count: int | None = None
    hard: bool = False          # ограничение типа исследований: в автоподборе применяется в последнюю очередь


def fmt(n: int | None) -> str:
    return "н/д" if n is None else f"{n:,}".replace(",", " ")


def safe_count(p: Provider, plan: SearchPlan) -> int | None:
    try:
        return p.count(plan)
    except Exception as exc:           # noqa: BLE001 — один недоступный источник не должен ронять диалог
        log.warning("%s: не удалось получить число результатов (%s)", p.label, exc)
        return None


# --------------------------------------------------------------------------- #
def narrowing_options(plan: SearchPlan) -> list[Option]:
    f, y = plan.filters, date.today().year
    opts: list[Option] = []

    def add(label, fn, hard=False):
        opts.append(Option(label, fn, hard=hard))

    if not f.year_from or f.year_from < y - 4:
        add(f"только последние 5 лет ({y - 4}–{y})", lambda p: setattr(p.filters, "year_from", y - 4))
    if not f.year_from or f.year_from < y - 9:
        add(f"только последние 10 лет ({y - 9}–{y})", lambda p: setattr(p.filters, "year_from", y - 9))
    if not f.open_access_only:
        add("только открытый доступ (их можно скачать)", lambda p: setattr(p.filters, "open_access_only", True))
    if not f.humans_only:
        add("только исследования на людях", lambda p: setattr(p.filters, "humans_only", True))
    if not f.languages:
        add("только английский язык", lambda p: setattr(p.filters, "languages", ["en"]))
    if sorted(f.study_types) != ["meta_analysis", "systematic_review"]:
        add("только метаанализы и систематические обзоры",
            lambda p: setattr(p.filters, "study_types", ["meta_analysis", "systematic_review"]), hard=True)
    if f.study_types != ["rct"]:
        add("только рандомизированные исследования (РКИ)", lambda p: setattr(p.filters, "study_types", ["rct"]), hard=True)
    if not f.has_abstract:
        add("только статьи с аннотацией", lambda p: setattr(p.filters, "has_abstract", True))
    return opts


def relaxing_options(plan: SearchPlan) -> list[Option]:
    f, opts = plan.filters, []

    def add(label, fn):
        opts.append(Option(label, fn))

    if f.year_from or f.year_to:
        add("убрать ограничение по годам", lambda p: (setattr(p.filters, "year_from", None), setattr(p.filters, "year_to", None)))
    if f.study_types:
        add("убрать ограничение по типу исследований", lambda p: setattr(p.filters, "study_types", []))
    if f.humans_only:
        add("убрать «только люди»", lambda p: setattr(p.filters, "humans_only", False))
    if f.languages:
        add("убрать ограничение по языку", lambda p: setattr(p.filters, "languages", []))
    if f.open_access_only:
        add("убрать «только открытый доступ»", lambda p: setattr(p.filters, "open_access_only", False))
    if f.has_abstract:
        add("убрать «только с аннотацией»", lambda p: setattr(p.filters, "has_abstract", False))
    if plan.exclude:
        add("убрать исключающие слова", lambda p: setattr(p, "exclude", []))
    if len(plan.concepts) > 1:
        for c in plan.concepts:
            add(f"убрать понятие «{c.label}»",
                lambda p, lbl=c.label: setattr(p, "concepts", [x for x in p.concepts if x.label != lbl] or p.concepts))
    return opts


def with_counts(plan: SearchPlan, opts: list[Option], primary: Provider) -> list[Option]:
    for o in opts:
        trial = plan.clone()
        o.apply(trial)
        o.count = safe_count(primary, trial)
    return opts


def rank_options(opts: list[Option], count: int | None, lo: int, hi: int) -> list[Option]:
    """Сначала варианты, попадающие в диапазон (самые мягкие первыми), затем ближайшие к нему."""
    opts = [o for o in opts if o.count is not None and o.count != count]
    fits = sorted((o for o in opts if lo <= o.count <= hi), key=lambda o: -o.count)
    rest = sorted((o for o in opts if not lo <= o.count <= hi),
                  key=lambda o: min(abs(o.count - lo), abs(o.count - hi)))
    return fits + rest


# --------------------------------------------------------------------------- #
def show_state(plan: SearchPlan, providers: list[Provider], counts: dict, say) -> None:
    say("\n" + "─" * 70)
    if plan.interpretation:
        say(f"Как я понял запрос: {plan.interpretation}")
    for c in plan.concepts:
        say(f"  • {c.label}: MeSH {c.mesh or '—'}; слова {', '.join(c.keywords[:6])}{'…' if len(c.keywords) > 6 else ''}")
    if plan.exclude:
        say(f"  • исключить: {', '.join(plan.exclude)}")
    say(f"Фильтры: {plan.filters.describe()}")
    say(f"Запрос {providers[0].label}:\n  {providers[0].render(plan)}")
    say("Найдено: " + " · ".join(f"{p.label} {fmt(counts.get(p.name))}" for p in providers))


def run_dialog(plan: SearchPlan, providers: list[Provider], lo: int = 20, hi: int = 300,
               llm: LLM | None = None, auto: bool = False, max_steps: int = 10,
               ask: Callable[[str], str] = input, say: Callable[[str], None] = print) -> SearchPlan | None:
    """Возвращает утверждённый план или None, если пользователь вышел.
    Диапазон lo–hi относится к первому провайдеру (основной, обычно PubMed)."""
    primary, seen = providers[0], {plan.signature()}
    for step in range(max_steps):
        counts = {p.name: safe_count(p, plan) for p in providers}
        n = counts[primary.name]
        show_state(plan, providers, counts, say)
        if n is None:
            say(f"{primary.label} недоступен — настройка по нему невозможна, использую план как есть.")
            return plan
        if lo <= n <= hi:
            verdict, opts = "ok", []
            say(f"✓ {fmt(n)} — в целевом диапазоне {lo}–{hi}.")
            if auto:
                return plan
        else:
            few = n < lo
            say(f"{'Мало' if few else 'Много'}: {fmt(n)} (цель {lo}–{hi}). "
                f"{'Предлагаю расширить запрос:' if few else 'Предлагаю сузить запрос:'}")
            raw = relaxing_options(plan) if few else narrowing_options(plan)
            opts = rank_options(with_counts(plan, raw, primary), n, lo, hi)
            if few:                          # при расширении «хуже» — это ещё меньше результатов
                opts = [o for o in opts if o.count > n]
            else:
                opts = [o for o in opts if o.count < n]
            for i, o in enumerate(opts, 1):
                mark = "✓" if lo <= o.count <= hi else " "
                say(f"  {i}) {o.label} → {fmt(o.count)} {mark}")
            if not opts:
                say("  (автоматических вариантов нет — опишите правку словами)")

        if auto:
            if not opts:
                say("Автоподбор: вариантов больше нет, использую текущий запрос.")
                return plan
            choice = next((o for o in opts if not o.hard), opts[0])   # сначала мягкие ограничения
            say(f"Автоподбор: {choice.label}")
            choice.apply(plan)
            if plan.signature() in seen:
                return plan
            seen.add(plan.signature())
            continue

        hint = "номер(а) через запятую, «ok» — принять, «q» — выйти, или напишите по-русски, что изменить"
        answer = ask(f"\n[{hint}]\n> ").strip()
        low = answer.lower()
        if low in ("ok", "ок", "да", "готово", ""):
            return plan
        if low in ("q", "quit", "exit", "выход", "й"):
            return None
        if all(t.strip().isdigit() for t in answer.split(",")):
            chosen = [int(t) for t in answer.split(",")]
            if all(1 <= k <= len(opts) for k in chosen):
                for k in chosen:
                    opts[k - 1].apply(plan)
                continue
            say("Нет такого номера.")
            continue
        if llm is None:
            say("ИИ не подключён (нет ключа), поэтому свободный текст обработать не могу — выберите номер.")
            continue
        try:
            plan, comment = refine_plan(llm, plan, answer)
            say(f"Изменено: {comment or 'план обновлён'}")
        except LLMError as exc:
            say(f"Не получилось применить правку: {exc}")
    say("Достигнут лимит шагов настройки — использую текущий запрос.")
    return plan

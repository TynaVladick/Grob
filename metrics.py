"""Нормализация и метрики сравнения «результат GROBID vs ручной эталон».

Точное сопоставление (доля совпадений): название, DOI, год.
Обнаружение сущностей (precision / recall / F1): авторы, заголовки разделов, ссылки.
Для длинного текста (аннотация) — меры сходства, а не точное равенство.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher


# --------------------------------------------------------------------------- #
# Нормализация
# --------------------------------------------------------------------------- #
def norm_text(s) -> str:
    """Без регистра, диакритики, знаков препинания; пробелы схлопнуты."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[\W_]+", " ", s.casefold())
    return s.strip()


def norm_doi(s) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", s)
    return s.strip().rstrip(".,;)")


_HEADING_NUM = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s*|[ivxlc]+\.\s+)", re.IGNORECASE)


def norm_heading(s) -> str:
    """Заголовок раздела без нумерации ('2.1. Methods' -> 'methods')."""
    return norm_text(_HEADING_NUM.sub("", s or ""))


def author_key(a: dict) -> tuple[str, str]:
    """(фамилия без пробелов, первая буква имени) — устойчиво к 'John A.' vs 'J.' и 'van der' vs 'vander'."""
    surname = norm_text(a.get("surname", "")).replace(" ", "")
    given = norm_text(a.get("given", ""))
    return surname, given[:1]


# --------------------------------------------------------------------------- #
# Базовые метрики
# --------------------------------------------------------------------------- #
def similarity(a: str, b: str) -> float:
    """Сходство нормализованных строк, 0..1 (SequenceMatcher). Пустая строка -> 0."""
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def token_f1(a: str, b: str) -> float:
    ca, cb = Counter(norm_text(a).split()), Counter(norm_text(b).split())
    if not ca or not cb:
        return 0.0
    tp = sum((ca & cb).values())
    if tp == 0:
        return 0.0
    p, r = tp / sum(cb.values()), tp / sum(ca.values())
    return 2 * p * r / (p + r)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def multiset_counts(gold_items: list, pred_items: list) -> tuple[int, int, int]:
    cg, cp = Counter(gold_items), Counter(pred_items)
    tp = sum((cg & cp).values())
    return tp, sum(cp.values()) - tp, sum(cg.values()) - tp


# --------------------------------------------------------------------------- #
# Ссылки на литературу
# --------------------------------------------------------------------------- #
def ref_score(g: dict, p: dict) -> float:
    """Насколько предсказанная ссылка p соответствует эталонной g (0..1).

    1.0   — совпал DOI;
    sim   — сходство нормализованных названий (если оно есть у обеих);
    0.9   — названия нет (или не сошлось), но совпали фамилия первого автора и год.
    """
    gd, pd = norm_doi(g.get("doi")), norm_doi(p.get("doi"))
    if gd and pd and gd == pd:
        return 1.0
    score = 0.0
    if g.get("title") and p.get("title"):
        score = similarity(g["title"], p["title"])
    gs = norm_text(g.get("first_author_surname")).replace(" ", "")
    ps = norm_text(p.get("first_author_surname")).replace(" ", "")
    if gs and gs == ps and g.get("year") and g.get("year") == p.get("year"):
        score = max(score, 0.9)
    return score


def match_references(gold: list[dict], pred: list[dict], threshold: float = 0.85) -> tuple[int, int, int]:
    """Жадное сопоставление один-к-одному. Возвращает (tp, fp, fn).

    «Ссылка найдена» = нашлась предсказанная ссылка с ref_score >= threshold.
    Это оценка ОБНАРУЖЕНИЯ ссылок, а не полной правильности всех их полей.
    """
    used: set[int] = set()
    tp = 0
    for g in gold:
        best_j, best = -1, 0.0
        for j, p in enumerate(pred):
            if j in used:
                continue
            s = ref_score(g, p)
            if s > best:
                best, best_j = s, j
        if best >= threshold:
            used.add(best_j)
            tp += 1
    return tp, len(pred) - tp, len(gold) - tp


# --------------------------------------------------------------------------- #
# Оценка одной статьи
# --------------------------------------------------------------------------- #
EMPTY_PRED = {"title": "", "authors": [], "year": None, "doi": "", "abstract": "",
              "section_headings": [], "references": []}


def evaluate_article(gold: dict, pred: dict | None, ref_threshold: float = 0.85) -> dict:
    pred = pred or EMPTY_PRED
    res: dict = {}

    # --- точное сопоставление ---
    res["title_exact"] = bool(norm_text(gold.get("title"))) and norm_text(gold["title"]) == norm_text(pred.get("title"))
    res["title_sim"] = similarity(gold.get("title", ""), pred.get("title", ""))

    g_doi = norm_doi(gold.get("doi"))
    res["doi_exact"] = (g_doi == norm_doi(pred.get("doi"))) if g_doi else None   # None: в эталоне DOI нет

    g_year = gold.get("year")
    if g_year:
        accepted = {int(g_year)} | {int(y) for y in gold.get("year_alt", [])}
        res["year_exact"] = pred.get("year") in accepted
    else:
        res["year_exact"] = None

    # --- аннотация: сходство ---
    res["abstract_sim"] = similarity(gold.get("abstract", ""), pred.get("abstract", ""))
    res["abstract_token_f1"] = token_f1(gold.get("abstract", ""), pred.get("abstract", ""))

    # --- сущности: precision / recall / F1 ---
    g_auth = [author_key(a) for a in gold.get("authors", [])]
    p_auth = [author_key(a) for a in pred.get("authors", [])]
    res["authors"] = multiset_counts(g_auth, p_auth)

    g_head = [h for h in (norm_heading(x) for x in gold.get("section_headings", [])) if h]
    p_head = [h for h in (norm_heading(x) for x in pred.get("section_headings", [])) if h]
    res["headings"] = multiset_counts(g_head, p_head)

    res["references"] = match_references(gold.get("references", []), pred.get("references", []), ref_threshold)

    res["n_gold_authors"], res["n_pred_authors"] = len(g_auth), len(p_auth)
    res["n_gold_headings"], res["n_pred_headings"] = len(g_head), len(p_head)
    res["n_gold_refs"], res["n_pred_refs"] = len(gold.get("references", [])), len(pred.get("references", []))
    return res
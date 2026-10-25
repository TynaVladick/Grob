#!/usr/bin/env python3
"""Поиск и автоскачивание научных статей по простому запросу на русском.

Этапы:
  1. ИИ превращает запрос в структурный план (понятия, MeSH, синонимы, фильтры);
     по плану собираются «профессиональные» запросы для PubMed, Europe PMC, OpenAlex, Semantic Scholar.
  2. Диалог: подбираем фильтры, чтобы статей было приемлемое количество (по PubMed, целевой диапазон --min/--max).
  3. Результаты всех баз объединяются, дубликаты склеиваются (DOI → PMID → PMCID → название).
  4. Статьи в открытом доступе скачиваются автоматически (PMC S3 → Europe PMC → ссылки баз → Unpaywall).

Примеры:
    export ANTHROPIC_API_KEY=...  NCBI_EMAIL=you@example.org
    python search_articles.py "влияет ли метформин на риск деменции у пожилых"
    python search_articles.py "ингибиторы SGLT2 при сердечной недостаточности" --auto --max-download 20
    python search_articles.py "..." --dry-run                    # только показать запросы, ничего не искать
    python search_articles.py --plan-file plan.json --no-llm     # готовый план без ИИ (см. README)

Результат: data/search_runs/<время>_<тема>/ {plan.json, articles.csv, data/pdf/*.pdf}.
Скачанные PDF можно сразу пускать в GROBID-пайплайн:  RES_DIR=<эта папка> python "03 run grobid.py"
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from common import BASE_DIR, setup_logging
from litsearch.dialog import fmt, run_dialog, safe_count
from litsearch.downloader import Downloader
from litsearch.http import Http
from litsearch.llm import LLM, LLMError, build_plan
from litsearch.models import SearchPlan, merge_articles
from litsearch.providers import REGISTRY, make_providers
from litsearch.providers.pubmed import PubMed

log = logging.getLogger("search")
DEFAULT_SOURCES = "pubmed,europepmc,openalex"
CSV_FIELDS = ["rank", "title", "year", "journal", "doi", "pmid", "pmcid", "sources", "is_oa", "license",
              "cited_by", "download_status", "download_source", "file", "error"]


def say(msg: str = "") -> None:
    print(msg, flush=True)


def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^\w]+", "-", text.lower(), flags=re.UNICODE).strip("-")
    return s[:n].strip("-") or "query"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?", help="запрос на русском (в кавычках)")
    ap.add_argument("--sources", default=DEFAULT_SOURCES,
                    help=f"базы через запятую; доступны: {', '.join(REGISTRY)} (по умолчанию {DEFAULT_SOURCES}). "
                         "Первая используется для подбора числа результатов")
    ap.add_argument("--min", type=int, default=20, dest="lo", help="целевой минимум результатов (по умолчанию 20)")
    ap.add_argument("--max", type=int, default=300, dest="hi", help="целевой максимум результатов (по умолчанию 300)")
    ap.add_argument("--auto", action="store_true", help="подбирать фильтры автоматически, без вопросов")
    ap.add_argument("--no-dialog", action="store_true", help="не подбирать фильтры: взять план как есть")
    ap.add_argument("--dry-run", action="store_true", help="только показать план и запросы по каждой базе")
    ap.add_argument("--plan-file", help="готовый план поиска (JSON) вместо ИИ")
    ap.add_argument("--no-llm", action="store_true", help="не обращаться к ИИ (в диалоге останутся только номера вариантов)")
    ap.add_argument("--max-records", type=int, default=200, help="сколько записей брать из каждой базы (по умолчанию 200)")
    ap.add_argument("--max-download", type=int, default=30, help="сколько статей скачать (по умолчанию 30)")
    ap.add_argument("--no-download", action="store_true", help="только найти, не скачивать")
    ap.add_argument("--no-unpaywall", action="store_true", help="не использовать Unpaywall")
    ap.add_argument("--licenses", default="", help="скачивать из PMC только эти лицензии, напр. 'CC BY,CC0' (по умолчанию любые)")
    ap.add_argument("--workers", type=int, default=4, help="параллельных загрузок (по умолчанию 4)")
    ap.add_argument("--out", help="папка результатов (по умолчанию data/search_runs/<время>_<тема>)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    if not args.query and not args.plan_file:
        ap.error("нужен запрос на русском или --plan-file")
    http = Http()
    try:
        providers = make_providers([s.strip() for s in args.sources.split(",") if s.strip()], http)
    except ValueError as exc:
        ap.error(str(exc))
    for p in providers:
        note = p.available()[1]
        if note:
            say(f"ℹ {p.label}: {note}")

    # ---- 1. ИИ: русский запрос -> структурный план --------------------------------
    llm = None
    if not args.no_llm:
        try:
            llm = LLM()
        except LLMError as exc:
            if not args.plan_file:
                say(f"✗ {exc}")
                return 2
            say(f"ℹ ИИ недоступен ({exc}) — работаю с готовым планом.")
    if args.plan_file:
        plan = SearchPlan.from_dict(json.loads(Path(args.plan_file).read_text(encoding="utf-8")),
                                    original=args.query or "")
    else:
        say("→ Строю поисковый запрос с помощью ИИ…")
        try:
            plan = build_plan(llm, args.query)
        except LLMError as exc:
            say(f"✗ {exc}")
            return 2
    pubmed = next((p for p in providers if isinstance(p, PubMed)), PubMed(http))
    for m in pubmed.validate_mesh(plan):
        say(f"ℹ Заголовка MeSH «{m}» нет в PubMed — использую как обычное слово.")

    if args.dry_run:
        say(f"План: {plan.interpretation}\nФильтры: {plan.filters.describe()}")
        for p in providers:
            say(f"\n[{p.label}]\n  {p.render(plan)}")
        say("\n" + json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
        return 0

    # ---- 2. Диалог подбора фильтров ------------------------------------------------
    if not args.no_dialog:
        interactive = sys.stdin.isatty() and not args.auto
        if not interactive and not args.auto:
            say("ℹ Терминал не интерактивный — включаю автоподбор фильтров (--auto).")
        plan = run_dialog(plan, providers, args.lo, args.hi, llm=llm, auto=not interactive)
        if plan is None:
            say("Отменено.")
            return 1

    # ---- 3. Поиск по всем базам и объединение -------------------------------------
    per_source = {}
    for p in providers:
        try:
            per_source[p.name] = p.search(plan, args.max_records)
            say(f"✓ {p.label}: получено {len(per_source[p.name])} записей")
        except Exception as exc:                       # noqa: BLE001 — одна упавшая база не отменяет остальные
            say(f"✗ {p.label}: поиск не удался ({exc})")
    articles = merge_articles(per_source)
    if not articles:
        say("Ничего не найдено ни в одной базе.")
        return 1
    say(f"Итого после объединения и удаления дубликатов: {len(articles)} статей "
        f"(в нескольких базах сразу: {sum(len(a.sources) > 1 for a in articles)})")

    out = Path(args.out) if args.out else BASE_DIR / "data" / "search_runs" / \
        f"{datetime.now():%Y%m%d-%H%M%S}_{slugify(plan.original or plan.interpretation)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps({
        "plan": plan.to_dict(),
        "queries": {p.name: p.render(plan) for p in providers},
        "counts": {p.name: safe_count(p, plan) for p in providers},
        "created": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 4. Скачивание ---------------------------------------------------------------
    results: dict[int, object] = {}
    # сначала статьи, у которых есть признаки открытого доступа (порядок по релевантности сохраняется)
    articles.sort(key=lambda a: 0 if (a.is_oa or a.pmcid or a.oa_urls) else 1)
    n_dl = 0 if args.no_download else args.max_download
    if n_dl and sys.stdin.isatty() and not args.auto:
        try:
            ans = input(f"Сколько статей скачать (в открытом доступе)? [Enter = {n_dl}, 0 = не скачивать] ").strip()
            n_dl = int(ans) if ans else n_dl
        except (ValueError, EOFError):
            pass
    if n_dl > 0:
        dl = Downloader(http, out / "data" / "pdf", allowed_licenses={x for x in args.licenses.split(",") if x.strip()} or None,
                        use_unpaywall=not args.no_unpaywall)
        if not dl.email and not args.no_unpaywall:
            say("ℹ Unpaywall отключён: задайте CONTACT_EMAIL или NCBI_EMAIL, чтобы он находил больше открытых PDF.")

        def progress(a, res, done, target):
            mark = "✓" if res.success else "×"
            tail = res.source or res.error[:80]
            say(f"  [{done}/{target}] {mark} {dl.file_stem(a):<18} {tail}  {a.title[:60]}")

        say(f"→ Скачиваю до {n_dl} статей (параллельно: {args.workers})…")
        results = dl.download_many(articles, n_dl, args.workers, progress)

    # ---- 5. Отчёт ---------------------------------------------------------------------
    with (out / "articles.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for i, a in enumerate(articles):
            r = results.get(i)
            w.writerow({
                "rank": i + 1, "title": a.title, "year": a.year or "", "journal": a.journal, "doi": a.doi,
                "pmid": a.pmid, "pmcid": a.pmcid, "sources": "+".join(a.sources),
                "is_oa": "" if a.is_oa is None else a.is_oa, "license": r.license if r and r.license else a.license,
                "cited_by": "" if a.cited_by is None else a.cited_by,
                "download_status": r.status if r else "not_attempted", "download_source": r.source if r else "",
                "file": str(Path(r.path).relative_to(out)) if r and r.path else "", "error": r.error if r else "",
            })
    ok = sum(1 for r in results.values() if r.success)
    say(f"\nГотово. Найдено {len(articles)}, скачано {ok} из {len(results)} попыток.")
    say(f"Папка: {out}\n  articles.csv — все найденные статьи и статус загрузки\n  plan.json — запросы и фильтры (воспроизводимость)")
    if ok:
        say(f"  data/pdf/ — PDF.  В GROBID-пайплайн:  RES_DIR='{out}' python \"03 run grobid.py\"")
    if results and ok < len(results):
        say("Часть статей не скачана: PDF в открытом доступе есть не у всех (причины — в колонке error). "
            "Обход платного доступа не выполняется.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

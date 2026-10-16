"""Вшивание рисунков, графиков и таблиц в текст статьи, чтобы их «прочитала» текстовая LLM из RAG.

Идея: для каждой статьи получить Markdown, где на месте рисунка/таблицы стоит ТЕКСТОВЫЙ блок
(подпись + таблица в Markdown или описание картинки + данные с графика), а сам файл картинки
сохранён рядом и указан в метаданных чанка (для мультимодальной модели на этапе генерации).

Откуда берётся геометрия (где на странице лежит рисунок/таблица), по убыванию надёжности:
  1. TEI от GROBID с координатами (teiCoordinates=figure[,ref]) -> layout "grobid-tei+coords";
  2. TEI без координат (как сейчас в шаге 3 baseline) + эвристика PyMuPDF; рисунки сопоставляются
     по номеру ("Figure 3") -> "grobid-tei+pymupdf-match";
  3. только PyMuPDF (GROBID/Docker недоступен) -> "pymupdf-only".

Зависимости: только requests и pymupdf (они уже в requirements.txt). Модель зрения (VLM) необязательна.
Все пути берутся из common.py, если он есть рядом; иначе используются те же значения по умолчанию.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

try:  # PyMuPDF >= 1.24.3 импортируется как pymupdf, старые версии — как fitz
    import pymupdf
except ImportError:  # pragma: no cover
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

# ----------------------------------------------------------------------------- пути
try:
    try:
        from common import (DATA_DIR, JATS_DIR, PDF_DIR, RESULTS_DIR, TEI_DIR, TEI_SUFFIX,
                            pdf_article_ids, setup_logging, tei_path)
    except ImportError:
        from res_common import (DATA_DIR, JATS_DIR, PDF_DIR, RESULTS_DIR, TEI_DIR,  # type: ignore
                                TEI_SUFFIX, pdf_article_ids, setup_logging, tei_path)
except ImportError:  # модуль можно запускать и отдельно от остального репозитория
    BASE_DIR = Path(os.environ.get("RES_DIR", Path(__file__).resolve().parent))
    DATA_DIR = BASE_DIR / "data"
    PDF_DIR, JATS_DIR, TEI_DIR = DATA_DIR / "pdf", DATA_DIR / "jats", DATA_DIR / "tei"
    RESULTS_DIR = BASE_DIR / "results"
    TEI_SUFFIX = ".tei.xml"

    def pdf_article_ids() -> list[str]:
        return sorted(p.stem for p in PDF_DIR.glob("*.pdf"))

    def tei_path(article_id: str) -> Path:
        return TEI_DIR / f"{article_id}{TEI_SUFFIX}"

    def setup_logging(verbose: bool = False) -> None:
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                            format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

TEI_COORDS_DIR = DATA_DIR / "tei_coords"   # TEI с координатами (отдельно, чтобы не трогать baseline)
ASSETS_DIR = DATA_DIR / "assets"           # PNG-вырезки, CSV таблиц, assets.json, кэш VLM
RAG_DIR = DATA_DIR / "rag"                 # <id>.md и <id>.chunks.jsonl
SUMMARY_CSV = RESULTS_DIR / "assets_summary.csv"

log = logging.getLogger("assets")

TEI = "http://www.tei-c.org/ns/1.0"
NS = {"t": TEI}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
_BLOCK = {"p", "head", "div", "note", "item", "label", "figDesc", "row"}


# ----------------------------------------------------------------------------- модель данных
@dataclass
class Asset:
    asset_id: str
    kind: str                         # "figure" | "table"
    label: str = ""                   # "Figure 1", "Table 2"
    caption: str = ""
    page: int | None = None           # 1-based
    box: tuple | None = None          # (x0, y0, x1, y1) в пунктах PyMuPDF
    section: str = ""
    mentions: list[str] = field(default_factory=list)
    layout_source: str = ""
    image_path: str = ""              # относительно папки статьи в ASSETS_DIR
    table_rows: list[list[str]] = field(default_factory=list)
    table_markdown: str = ""
    table_source: str = ""            # pymupdf-lines | pymupdf-text | tei-cells | vlm | tei-flat-text
    tei_rows: list[list[str]] = field(default_factory=list)
    tei_flat_text: str = ""
    vlm: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    _tei_boxes: list = field(default_factory=list, repr=False)   # [(page, x, y, w, h)] из TEI


# ----------------------------------------------------------------------------- геометрия (чистый Python)
def box_union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def boxes_touch(a, b, gap: float = 6.0) -> bool:
    return max(b[0] - a[2], a[0] - b[2]) <= gap and max(b[1] - a[3], a[1] - b[3]) <= gap


def merge_boxes(boxes, gap: float = 6.0) -> list[tuple]:
    """Склеивает пересекающиеся и соприкасающиеся (с зазором <= gap) прямоугольники."""
    pool = [tuple(b) for b in boxes]
    out: list[tuple] = []
    while pool:
        cur = pool.pop()
        grew = True
        while grew:
            grew = False
            for i, o in enumerate(pool):
                if boxes_touch(cur, o, gap):
                    cur = box_union(cur, o)
                    pool.pop(i)
                    grew = True
                    break
        out.append(cur)
    return out


def clamp_box(b, page_box):
    return (max(b[0], page_box[0]), max(b[1], page_box[1]), min(b[2], page_box[2]), min(b[3], page_box[3]))


def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def h_overlap(a, b) -> float:
    """Доля горизонтального перекрытия относительно более узкого прямоугольника."""
    inter = min(a[2], b[2]) - max(a[0], b[0])
    return max(0.0, inter) / max(1e-6, min(a[2] - a[0], b[2] - b[0]))


def parse_coords(s: str | None) -> list[tuple]:
    """GROBID coords: 'page,x,y,w,h;page,x,y,w,h' (начало координат слева сверху)."""
    out = []
    for part in (s or "").split(";"):
        try:
            p, x, y, w, h = part.split(",")
            out.append((int(p), float(x), float(y), float(w), float(h)))
        except ValueError:
            continue
    return out


def tei_boxes_to_page_box(tei_boxes, page_no: int, surface: tuple | None, page_size: tuple) -> tuple | None:
    """Объединяет боксы TEI одной страницы и пересчитывает в пункты PyMuPDF.

    surface — (width, height) страницы из <facsimile> TEI; page_size — (width, height) в PyMuPDF.
    Если размеры различаются (CropBox/MediaBox), координаты масштабируются.
    """
    same = [b for b in tei_boxes if b[0] == page_no]
    if not same:
        return None
    x0 = min(b[1] for b in same)
    y0 = min(b[2] for b in same)
    x1 = max(b[1] + b[3] for b in same)
    y1 = max(b[2] + b[4] for b in same)
    sx = page_size[0] / surface[0] if surface and surface[0] else 1.0
    sy = page_size[1] / surface[1] if surface and surface[1] else 1.0
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)


# ----------------------------------------------------------------------------- разбор TEI
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(el) -> str:
    """Текст элемента без вложенных <figure>; пробел на границах блочных элементов."""
    if el is None:
        return ""
    parts: list[str] = []

    def walk(e) -> None:
        tag = _local(e.tag)
        if e is not el and tag == "figure":
            return
        block = tag in _BLOCK
        if block:
            parts.append(" ")
        if e.text:
            parts.append(e.text)
        for ch in e:
            walk(ch)
            if ch.tail:
                parts.append(ch.tail)
        if block:
            parts.append(" ")

    walk(el)
    return " ".join("".join(parts).split())


_LABEL_RE = re.compile(r"^\s*(figure|fig\.?|table|tab\.?)\s*(S?\d+[A-Za-z]?)?", re.I)


def norm_key(kind: str, label: str) -> tuple[str, str] | None:
    """('figure', '3') из 'Fig. 3.' — ключ для сопоставления рисунков между TEI и PDF."""
    m = re.search(r"S?\d+", label or "", re.I)
    return (kind, m.group(0).upper()) if m else None


def strip_label(caption: str, label: str) -> str:
    """Убирает ведущее 'Figure 1.' из подписи, чтобы метка не дублировалась."""
    if not caption:
        return ""
    if label and caption.lower().startswith(label.lower()):
        return caption[len(label):].lstrip(" .:|–—-")
    m = _LABEL_RE.match(caption)
    if m and m.group(2):
        return caption[m.end():].lstrip(" .:|–—-")
    return caption


def _snippet(par: str, needle: str, width: int = 220) -> str:
    i = par.find(needle) if needle else -1
    if i < 0:
        return par[: 2 * width].strip()
    a, b = max(0, i - width), min(len(par), i + len(needle) + width)
    return ("…" if a else "") + par[a:b].strip() + ("…" if b < len(par) else "")


def _heading_level(head_el) -> int:
    n = (head_el.get("n") or "").strip().rstrip(".")
    return min(6, 2 + n.count(".")) if n else 2


def _walk_body(node, items: list, state: dict, fig_ids: dict) -> None:
    for ch in node:
        tag = _local(ch.tag)
        if tag == "div":
            _walk_body(ch, items, state, fig_ids)
        elif tag == "head":
            t = _text(ch)
            if t:
                state["section"] = t
                items.append(("heading", _heading_level(ch), t))
        elif tag in ("p", "formula"):
            t = _text(ch)
            if t:
                items.append(("para", t))
            for fig in ch.iter(f"{{{TEI}}}figure"):          # вложенные рисунки (редко)
                items.append(("asset", fig_ids[fig], state["section"]))
        elif tag == "figure":
            items.append(("asset", fig_ids[ch], state["section"]))


def parse_tei_document(path: Path | str) -> dict:
    """TEI -> {title, surfaces, assets, items, has_coords}. items: heading/para/asset в порядке чтения."""
    root = ET.parse(str(path)).getroot()
    surfaces: dict[int, tuple] = {}
    for s in root.findall("t:facsimile/t:surface", NS):
        try:
            surfaces[int(s.get("n"))] = (float(s.get("lrx")), float(s.get("lry")))
        except (TypeError, ValueError):
            pass
    title = _text(root.find("t:teiHeader/t:fileDesc/t:titleStmt/t:title", NS))

    fig_ids: dict = {}
    assets: dict[str, Asset] = {}
    n_f = n_t = 0
    for el in root.iter(f"{{{TEI}}}figure"):
        kind = "table" if el.get("type") == "table" else "figure"
        generated = f"tab_{n_t}" if kind == "table" else f"fig_{n_f}"
        n_f += kind == "figure"
        n_t += kind == "table"
        aid = el.get(XML_ID) or generated
        fig_ids[el] = aid
        head = _text(el.find("t:head", NS))
        label_n = _text(el.find("t:label", NS))
        desc = _text(el.find("t:figDesc", NS))
        word = "Table" if kind == "table" else "Figure"
        if _LABEL_RE.match(head) and re.search(r"\d", head) and len(head) < 20:
            label = head.rstrip(" .:")
        elif label_n:
            label = f"{word} {label_n}"
        else:
            label = ""
        caption = desc or head
        if head and desc and head != label and not desc.lower().startswith(head.lower()) \
                and not _LABEL_RE.match(head):
            caption = f"{head} {desc}"
        a = Asset(aid, kind, label=label, caption=caption, layout_source="grobid-tei")
        a._tei_boxes = parse_coords(el.get("coords"))
        for g in el.iter(f"{{{TEI}}}graphic"):               # графика, если GROBID её нашёл
            a._tei_boxes += parse_coords(g.get("coords"))
        tb = el.find("t:table", NS)
        if tb is not None:
            a.tei_rows = [[_text(c) for c in r.findall("t:cell", NS)] for r in tb.findall("t:row", NS)]
            a.tei_flat_text = _text(tb)
        assets[aid] = a

    # упоминания в тексте: <ref type="figure|table" target="#fig_0">
    for p in root.iterfind("t:text/t:body//t:p", NS):
        ptxt = None
        for ref in p.iter(f"{{{TEI}}}ref"):
            if ref.get("type") in ("figure", "table") and ref.get("target"):
                tid = ref.get("target").lstrip("#")
                if tid in assets and len(assets[tid].mentions) < 3:
                    ptxt = ptxt or _text(p)
                    assets[tid].mentions.append(_snippet(ptxt, _text(ref)))

    items: list = []
    state = {"section": ""}
    body = root.find("t:text/t:body", NS)
    if body is not None:
        _walk_body(body, items, state, fig_ids)
    for it in items:
        if it[0] == "asset" and it[1] in assets:
            assets[it[1]].section = it[2]
    return {"title": title, "surfaces": surfaces, "assets": list(assets.values()), "items": items,
            "has_coords": any(a._tei_boxes for a in assets.values())}


# ----------------------------------------------------------------------------- PyMuPDF: геометрия
def _require_pymupdf() -> None:
    if pymupdf is None:
        raise SystemExit("Нужен PyMuPDF: pip install -U pymupdf")


_CAP_RE = re.compile(r"^\s*(?P<kind>Figure|Fig\.?|Table)\s*(?P<num>S?\d+)\s*(?:[.:|–—\-]|\n|$)", re.I)


def graphic_candidates(page, min_w: float = 40.0, min_h: float = 30.0) -> list[tuple]:
    """Прямоугольники растровых картинок и кластеров векторной графики на странице.

    Графики из matplotlib/R/Excel в PDF — это НЕ картинки, а векторные линии, поэтому
    page.get_images() их не видит; для них нужен cluster_drawings().
    """
    out: list[tuple] = []
    try:
        for info in page.get_image_info():
            out.append(tuple(info["bbox"]))
    except Exception as exc:  # noqa: BLE001
        log.debug("get_image_info: %s", exc)
    if hasattr(page, "cluster_drawings"):
        try:
            for r in page.cluster_drawings():
                out.append((r.x0, r.y0, r.x1, r.y1))
        except Exception as exc:  # noqa: BLE001
            log.debug("cluster_drawings: %s", exc)
    r = page.rect
    page_box = (r.x0, r.y0, r.x1, r.y1)
    res = []
    for b in merge_boxes(out, gap=6.0):
        b = clamp_box(b, page_box)
        if (b[2] - b[0]) >= min_w and (b[3] - b[1]) >= min_h and box_area(b) < 0.85 * box_area(page_box):
            res.append(b)
    return res


def refine_figure_box(page, box: tuple) -> tuple:
    """Расширяет бокс GROBID до графики, которой он касается (бокс часто покрывает лишь подпись/оси)."""
    cands = graphic_candidates(page)
    cur = box
    grew = True
    while grew:
        grew = False
        for c in cands:
            if boxes_touch(cur, c, 4.0) and box_union(cur, c) != cur:
                cur = box_union(cur, c)
                grew = True
    return cur


def detect_page_assets(page, page_no: int) -> list[Asset]:
    """Эвристика без GROBID: подпись 'Figure N'/'Table N' + ближайшая графика или ruled-таблица."""
    r = page.rect
    page_box = (r.x0, r.y0, r.x1, r.y1)
    caps = []
    for b in page.get_text("blocks"):
        if b[6] != 0:
            continue
        m = _CAP_RE.match(b[4])
        if m and len(b[4].strip()) > 12:
            kind = "table" if m.group("kind").lower().startswith("t") else "figure"
            caps.append({"kind": kind, "num": m.group("num").upper(), "box": tuple(b[:4]),
                         "text": " ".join(b[4].split())})
    if not caps:
        return []

    tables = []
    try:
        for t in page.find_tables().tables:
            tables.append(tuple(t.bbox))
    except Exception as exc:  # noqa: BLE001
        log.debug("find_tables стр.%d: %s", page_no, exc)

    found: list[Asset] = []
    used_tables: set[int] = set()
    for cap in [c for c in caps if c["kind"] == "table"]:
        best, best_d = None, 90.0
        for i, tb in enumerate(tables):
            if i in used_tables or h_overlap(cap["box"], tb) < 0.3:
                continue
            d = tb[1] - cap["box"][3] if tb[1] >= cap["box"][3] - 5 else cap["box"][1] - tb[3]
            if -5 <= d < best_d:
                best, best_d = i, d
        if best is not None:
            used_tables.add(best)
            found.append(_heur_asset("table", cap, tables[best], page_no, len(found)))

    cands = [c for c in graphic_candidates(page)
             if not any(boxes_touch(c, tb, 0) and box_area(box_union(c, tb)) <= box_area(tb) * 1.05
                        for tb in tables)]
    groups: dict[int, list] = {}
    fig_caps = [c for c in caps if c["kind"] == "figure"]
    for c in cands:
        best, best_d = None, 90.0
        for i, cap in enumerate(fig_caps):
            if h_overlap(cap["box"], c) < 0.3:
                continue
            below = cap["box"][1] - c[3]          # подпись под рисунком
            above = c[1] - cap["box"][3]          # подпись над рисунком
            d = below if -5 <= below <= 90 else (above + 30 if -5 <= above <= 40 else None)
            if d is not None and d < best_d:
                best, best_d = i, d
        if best is not None:
            groups.setdefault(best, []).append(c)
    for i, cs in groups.items():
        box = cs[0]
        for c in cs[1:]:
            box = box_union(box, c)
        found.append(_heur_asset("figure", fig_caps[i], clamp_box(box, page_box), page_no, len(found)))
    return found


def _heur_asset(kind: str, cap: dict, box: tuple, page_no: int, n: int) -> Asset:
    word = "Table" if kind == "table" else "Figure"
    label = f"{word} {cap['num']}"
    pref = "tab" if kind == "table" else "fig"
    a = Asset(f"{pref}_p{page_no}_{n}", kind, label=label, caption=strip_label(cap["text"], label),
              page=page_no, box=box, layout_source="pymupdf-heuristic")
    a.notes.append("геометрия найдена эвристикой PyMuPDF — проверьте вырезку глазами")
    return a


def detect_all_assets(doc) -> list[Asset]:
    res: list[Asset] = []
    for i in range(len(doc)):
        try:
            res += detect_page_assets(doc[i], i + 1)
        except Exception as exc:  # noqa: BLE001
            log.warning("эвристика, стр.%d: %s", i + 1, exc)
    return res


# ----------------------------------------------------------------------------- вырезка и таблицы
def render_clip(page, box: tuple, out_png: Path, dpi: int = 200, max_px: int = 2000, pad: float = 2.0) -> bool:
    r = page.rect
    b = clamp_box((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), (r.x0, r.y0, r.x1, r.y1))
    w, h = b[2] - b[0], b[3] - b[1]
    if w < 20 or h < 20:
        return False
    eff = min(dpi, int(max_px / max(w, h) * 72))
    pix = page.get_pixmap(dpi=max(eff, 72), clip=pymupdf.Rect(*b), alpha=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_png))
    return True


def _clean_cell(c) -> str:
    return " ".join(str(c).split()) if c is not None else ""


def rows_valid(rows: list[list[str]]) -> bool:
    if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
        return False
    cells = [c for r in rows for c in r]
    return sum(1 for c in cells if c) / max(1, len(cells)) >= 0.3


def rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    n = max(len(r) for r in rows)
    norm = [[(c or "").replace("|", "\\|").replace("\n", " ") for c in r] + [""] * (n - len(r)) for r in rows]
    lines = ["| " + " | ".join(norm[0]) + " |", "| " + " | ".join(["---"] * n) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in norm[1:]]
    return "\n".join(lines)


def extract_table_rows(page, box: tuple) -> tuple[list[list[str]], str]:
    """PyMuPDF find_tables в области таблицы: сначала по линиям, затем по расположению текста."""
    for strategy in ("lines", "text"):
        try:
            tf = page.find_tables(clip=pymupdf.Rect(*box), strategy=strategy)
        except Exception as exc:  # noqa: BLE001
            log.debug("find_tables(%s): %s", strategy, exc)
            continue
        best: list[list[str]] = []
        for t in tf.tables:
            rows = [[_clean_cell(c) for c in row] for row in t.extract()]
            if rows_valid(rows) and len(rows) * max(len(r) for r in rows) > \
                    len(best) * max((len(r) for r in best), default=0):
                best = rows
        if best:
            return best, f"pymupdf-{strategy}"
    return [], ""


def save_table_csv(rows: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


# ----------------------------------------------------------------------------- VLM (необязательно)
_LANG_NAMES = {"en": "English", "ru": "Russian", "de": "German", "fr": "French", "es": "Spanish"}

FIGURE_PROMPT = """You are helping index a scientific article for text-only retrieval.
Look at the image (one figure from the article) and answer with ONE JSON object, no markdown fences:
{{"kind": "chart|diagram|flowchart|photo|micrograph|medical_image|map|other",
 "readable": true,
 "description": "...",
 "text_in_image": "...",
 "data_table_markdown": "..."}}
Rules:
- description: 2-5 sentences, ONLY what is visible: plot type, axis titles/units/ranges, series and legend,
  groups compared, visible ordering/trends/outliers. No clinical interpretation, no conclusions the figure
  does not show, no guessing of unreadable text.
- text_in_image: titles, axis labels, legend entries and panel letters, verbatim.
- data_table_markdown: only for charts whose values can be read from labels or ticks; a Markdown table,
  approximate values prefixed with "~"; otherwise "".
- If the image is not a real figure or is unreadable, set "readable": false.
Write "description" in {lang}. Keep "text_in_image" verbatim.
Caption from the article: {caption}
Where the figure is cited in the text: {mentions}"""

TABLE_PROMPT = """Transcribe the table in the image into a GitHub-flavored Markdown table.
Keep every row and column and the header. Repeat the text of merged header cells over the columns they span.
Do not correct, round, add or interpret values; write "?" for unreadable cells.
Output ONLY the Markdown table; footnotes, if any, as separate lines starting with "Note:".
Caption from the article: {caption}"""


def parse_vlm_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(t[i:j + 1])
            if isinstance(d, dict):
                return {"kind": str(d.get("kind", "")), "readable": bool(d.get("readable", True)),
                        "description": str(d.get("description", "")).strip(),
                        "text_in_image": str(d.get("text_in_image", "")).strip(),
                        "data_table_markdown": str(d.get("data_table_markdown", "")).strip()}
        except json.JSONDecodeError:
            pass
    return {"kind": "", "readable": True, "description": t, "text_in_image": "", "data_table_markdown": ""}


class VLM:
    """Тонкий клиент к модели зрения. backend: 'anthropic' | 'openai' (любой OpenAI-совместимый сервер:
    OpenAI, vLLM, Ollama, LM Studio ...). Ответы кэшируются по sha256(модель + промпт + картинка)."""

    def __init__(self, backend: str, model: str, base_url: str | None = None, api_key: str | None = None,
                 lang: str = "en", cache_path: Path | None = None, timeout: int = 180, max_retries: int = 3):
        if backend not in ("anthropic", "openai"):
            raise ValueError(backend)
        self.backend, self.model, self.lang = backend, model, lang
        self.timeout, self.max_retries = timeout, max_retries
        self.base_url = (base_url or ("https://api.anthropic.com" if backend == "anthropic"
                                      else "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY" if backend == "anthropic" else "VLM_API_KEY")
        if backend == "anthropic" and not self.api_key:
            raise SystemExit("Нужен ANTHROPIC_API_KEY для --vlm anthropic")
        self.session = requests.Session()
        self.calls = 0
        self.cache: dict[str, str] = {}
        self.cache_path = cache_path
        if cache_path and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    self.cache[d["key"]] = d["text"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            r = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
            if r.status_code in (429, 500, 502, 503, 504, 529) and attempt < self.max_retries:
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")

    def ask(self, image: Path, prompt: str, max_tokens: int = 1500) -> str:
        data = image.read_bytes()
        key = hashlib.sha256(self.model.encode() + b"\0" + prompt.encode() + b"\0" + data).hexdigest()
        if key in self.cache:
            return self.cache[key]
        b64 = base64.b64encode(data).decode()
        if self.backend == "anthropic":
            body = self._post(
                f"{self.base_url}/v1/messages",
                {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                {"model": self.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt}]}]})
            text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        else:
            headers = {"content-type": "application/json"}
            if self.api_key:
                headers["authorization"] = f"Bearer {self.api_key}"
            # для новых моделей OpenAI параметр называется max_completion_tokens
            tok = os.environ.get("VLM_TOKEN_PARAM", "max_tokens")
            body = self._post(f"{self.base_url}/chat/completions", headers, {
                "model": self.model, tok: max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]})
            text = body["choices"][0]["message"]["content"] or ""
        self.calls += 1
        self.cache[key] = text
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
        return text

    def describe_figure(self, image: Path, caption: str, mentions: list[str]) -> dict:
        prompt = FIGURE_PROMPT.format(
            lang=_LANG_NAMES.get(self.lang, self.lang), caption=caption or "(none)",
            mentions=" | ".join(mentions) if mentions else "(none)")
        return parse_vlm_json(self.ask(image, prompt))

    def transcribe_table(self, image: Path, caption: str) -> str:
        text = self.ask(image, TABLE_PROMPT.format(caption=caption or "(none)"), max_tokens=4000)
        return re.sub(r"^```(?:markdown)?|```$", "", text.strip(), flags=re.M).strip()


# ----------------------------------------------------------------------------- Markdown и чанки
_I18N = {
    "en": {"desc": "Image description (auto-generated from the image, may contain errors)",
           "text": "Text visible in the image", "data": "Data read from the chart (approximate, auto-generated)",
           "vlm_table": "(table transcribed from the image by a vision model; may contain errors)",
           "flat": "(table text; row/column structure was lost)", "file": "Image file",
           "mentions": "Mentioned in the text", "no_struct": "(table content could not be extracted; see image file)",
           "extra": "Figures and tables found in the PDF but not placed in the text"},
    "ru": {"desc": "Описание изображения (сгенерировано автоматически по картинке, возможны ошибки)",
           "text": "Текст на изображении", "data": "Данные с графика (приблизительно, сгенерированы автоматически)",
           "vlm_table": "(таблица распознана по изображению моделью зрения; возможны ошибки)",
           "flat": "(текст таблицы; структура строк и столбцов потеряна)", "file": "Файл изображения",
           "mentions": "Упоминается в тексте", "no_struct": "(содержимое таблицы не извлечено; см. файл изображения)",
           "extra": "Рисунки и таблицы, найденные в PDF, но не привязанные к тексту"},
}


def render_asset_md(a: Asset, lang: str = "en", with_mentions: bool = False) -> str:
    L = _I18N.get(lang, _I18N["en"])
    word = "Table" if a.kind == "table" else "Figure"
    lab = a.label or word
    cap = strip_label(a.caption, a.label)
    out = [f"<!-- asset:begin id={a.asset_id} type={a.kind} page={a.page or '?'} -->",
           f"**{lab}.** {cap}".rstrip()]
    if a.kind == "table":
        if a.table_markdown:
            out += ["", a.table_markdown]
            if a.table_source == "vlm":
                out += ["", f"*{L['vlm_table']}*"]
        elif a.tei_flat_text:
            out += ["", f"*{L['flat']}* {a.tei_flat_text}"]
        else:
            out += ["", f"*{L['no_struct']}*"]
    else:
        v = a.vlm
        if v.get("readable", True) and v.get("description"):
            out += ["", f"*{L['desc']}:* {v['description']}"]
        if v.get("text_in_image"):
            out += ["", f"*{L['text']}:* {v['text_in_image']}"]
        if v.get("data_table_markdown"):
            out += ["", f"*{L['data']}:*", "", v["data_table_markdown"]]
    if a.image_path:
        out += ["", f"*{L['file']}:* `{a.image_path}`"]
    if with_mentions and a.mentions:
        out += ["", f"*{L['mentions']}:* " + " … ".join(f"«{m}»" for m in a.mentions[:2])]
    out.append("<!-- asset:end -->")
    return "\n".join(out)


def render_article_md(title: str, items: list, assets: dict[str, Asset], lang: str = "en") -> str:
    out = [f"# {title}"] if title else []
    for it in items:
        if it[0] == "heading":
            out.append(f"{'#' * it[1]} {it[2]}")
        elif it[0] == "para":
            out.append(it[1])
        elif it[0] == "asset" and it[1] in assets:
            out.append(render_asset_md(assets[it[1]], lang))
    return "\n\n".join(out) + "\n"


def _split_long(par: str, max_chars: int) -> list[str]:
    if len(par) <= max_chars:
        return [par]
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", par):
        if cur and len(cur) + len(s) + 1 > max_chars:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    return parts + ([cur] if cur else [])


def split_asset_text(text: str, max_chars: int) -> list[str]:
    """Режет слишком длинный блок; строки Markdown-таблицы переносятся с повтором шапки."""
    if len(text) <= max_chars:
        return [text]
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("|")]
    if len(idx) < 4:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    head_part = "\n".join(lines[:idx[0]])
    header = lines[idx[0]:idx[0] + 2]
    rows = [lines[i] for i in idx[2:]]
    tail = "\n".join(lines[idx[-1] + 1:])
    parts, cur = [], []
    budget = max_chars - len(head_part) - sum(len(h) for h in header) - 50
    size = 0
    for r in rows:
        if cur and size + len(r) > budget:
            parts.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += len(r) + 1
    if cur:
        parts.append(cur)
    n = len(parts)
    return [f"{head_part}\n" + "\n".join(header + p) + (f"\n(part {k + 1}/{n})" if n > 1 else "")
            + (f"\n{tail}" if k == n - 1 and tail.strip() else "")
            for k, p in enumerate(parts)]


def build_chunks(article_id: str, title: str, items: list, assets: dict[str, Asset], lang: str = "en",
                 max_chars: int = 1500, max_asset_chars: int = 6000) -> list[dict]:
    """Текстовые чанки по абзацам (префикс — раздел) + отдельные чанки на каждый рисунок/таблицу.

    Блок рисунка/таблицы не смешивается с соседним текстом и не рвётся на полуслове."""
    chunks: list[dict] = []
    section, buf, n = "", [], [0]

    def add(kind: str, text: str, **meta) -> None:
        n[0] += 1
        chunks.append({"chunk_id": f"{article_id}#{n[0]:04d}", "article_id": article_id, "type": kind,
                       "section": section, "text": text, "article_title": title, **meta})

    def flush() -> None:
        if buf:
            add("text", (f"[Section: {section}]\n" if section else "") + "\n\n".join(buf))
            buf.clear()

    for it in items:
        if it[0] == "heading":
            flush()
            section = it[2]
        elif it[0] == "para":
            for piece in _split_long(it[1], max_chars):
                if buf and sum(map(len, buf)) + len(piece) > max_chars:
                    flush()
                buf.append(piece)
        elif it[0] == "asset" and it[1] in assets:
            flush()
            a = assets[it[1]]
            for part in split_asset_text(render_asset_md(a, lang, with_mentions=True), max_asset_chars):
                add(a.kind, part, asset_id=a.asset_id, label=a.label, page=a.page,
                    image_path=a.image_path, layout_source=a.layout_source, table_source=a.table_source)
    flush()
    return chunks


# ----------------------------------------------------------------------------- GROBID (необязательно)
def fetch_tei_with_coords(pdf: Path, out: Path, base_url: str, timeout: int = 300) -> bool:
    """Отдельный запрос в GROBID с teiCoordinates (baseline-TEI в data/tei не перезаписывается)."""
    data = [("teiCoordinates", "figure"), ("teiCoordinates", "ref"),
            ("consolidateHeader", "0"), ("consolidateCitations", "0")]
    try:
        with pdf.open("rb") as fh:
            r = requests.post(f"{base_url.rstrip('/')}/api/processFulltextDocument",
                              files={"input": (pdf.name, fh, "application/pdf")}, data=data, timeout=timeout)
    except requests.RequestException as exc:
        log.warning("GROBID недоступен (%s) — переключаюсь на эвристику PyMuPDF", exc)
        return False
    if r.status_code != 200:
        log.warning("GROBID вернул HTTP %s для %s", r.status_code, pdf.name)
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return True


# ----------------------------------------------------------------------------- JATS: только счётчики для проверки
def jats_counts(article_id: str) -> tuple[int | None, int | None]:
    """Сколько <fig> и <table-wrap> в JATS XML статьи. Только для сверки полноты, в конвейер не идёт."""
    for p in sorted(JATS_DIR.glob(f"{article_id}*.xml")):
        try:
            root = ET.parse(str(p)).getroot()
        except ET.ParseError:
            continue
        return len(root.findall(".//fig")), len(root.findall(".//table-wrap"))
    return None, None


# ----------------------------------------------------------------------------- основной процесс
def _items_from_pdf(doc, assets: list[Asset]) -> list:
    """Без TEI: текстовые блоки PyMuPDF по страницам; подпись рисунка/таблицы заменяется блоком актива."""
    items: list = []
    by_page: dict[int, list[Asset]] = {}
    for a in assets:
        by_page.setdefault(a.page or 0, []).append(a)
    for i in range(len(doc)):
        page = doc[i]
        pa = by_page.get(i + 1, [])
        placed: set[str] = set()
        for b in page.get_text("blocks", sort=True):
            if b[6] != 0:
                continue
            bb = tuple(b[:4])
            text = " ".join(b[4].split())
            hit = next((a for a in pa if a.box and boxes_touch(bb, a.box, 0)
                        and box_area(box_union(bb, a.box)) <= box_area(a.box) * 1.5), None)
            cap = _CAP_RE.match(b[4])
            if hit is not None:                      # текст внутри рисунка/таблицы уже в блоке актива
                if hit.asset_id not in placed:
                    placed.add(hit.asset_id)
                    items.append(("asset", hit.asset_id, ""))
            elif cap:
                key = ("table" if cap.group("kind").lower().startswith("t") else "figure", cap.group("num").upper())
                m = next((a for a in pa if norm_key(a.kind, a.label) == key), None)
                if m:                                # подпись уже входит в блок актива
                    if m.asset_id not in placed:
                        placed.add(m.asset_id)
                        items.append(("asset", m.asset_id, ""))
                elif text:
                    items.append(("para", text))
            elif text:
                items.append(("para", text))
        for a in pa:
            if a.asset_id not in placed:
                items.append(("asset", a.asset_id, ""))
    return items


def process_article(aid: str, *, layout: str = "auto", regrobid: bool = False, grobid_url: str = "http://localhost:8070",
                    refine: bool = True, dpi: int = 200, max_px: int = 2000, vlm: VLM | None = None,
                    vlm_tables: bool = False, lang: str = "en", max_chars: int = 1500) -> dict:
    """Полная обработка одной статьи. Возвращает строку для assets_summary.csv."""
    _require_pymupdf()
    pdf = PDF_DIR / f"{aid}.pdf"
    out_dir = ASSETS_DIR / aid
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("*.png")) + list(out_dir.glob("*.csv")) + list(out_dir.glob("assets.json")):
        old.unlink()                      # не оставляем файлы от прошлого прогона в другом режиме
    calls0 = vlm.calls if vlm else 0

    # --- 1. какой TEI использовать
    tei = None
    if layout != "pymupdf":
        cand = TEI_COORDS_DIR / f"{aid}{TEI_SUFFIX}"
        if not cand.exists() and regrobid:
            fetch_tei_with_coords(pdf, cand, grobid_url)
        for path in (cand, tei_path(aid)):
            if path.exists():
                try:
                    parsed = parse_tei_document(path)
                except ET.ParseError as exc:
                    log.warning("%s: битый TEI %s (%s)", aid, path.name, exc)
                    continue
                if tei is None or (parsed["has_coords"] and not tei["has_coords"]):
                    tei = parsed
        if layout == "grobid" and tei is None:
            raise RuntimeError("layout=grobid, но TEI нет (запустите шаг 3 или --regrobid)")

    doc = pymupdf.open(str(pdf))
    try:
        sizes = {i + 1: (doc[i].rect.width, doc[i].rect.height) for i in range(len(doc))}
        heur: list[Asset] | None = None

        # --- 2. список активов + порядок чтения
        if tei is not None:
            assets = tei["assets"]
            items, title = tei["items"], tei["title"]
            mode = "grobid-tei+coords" if tei["has_coords"] else "grobid-tei+pymupdf-match"
            for a in assets:
                if a._tei_boxes:
                    a.page = min(b[0] for b in a._tei_boxes)
                    if 1 <= a.page <= len(doc):
                        a.box = tei_boxes_to_page_box(a._tei_boxes, a.page, tei["surfaces"].get(a.page), sizes[a.page])
                        a.layout_source = "grobid-tei+coords"
                        if len({b[0] for b in a._tei_boxes}) > 1:
                            a.notes.append("актив занимает несколько страниц: обработана только первая")
                        if refine and a.kind == "figure" and a.box:
                            a.box = refine_figure_box(doc[a.page - 1], a.box)
            missing = [a for a in assets if a.box is None]
            extra: list[Asset] = []
            if missing:
                heur = detect_all_assets(doc)
                by_key = {norm_key(h.kind, h.label): h for h in heur}
                matched: set[str] = set()
                for a in missing:
                    h = by_key.get(norm_key(a.kind, a.label))
                    if h:
                        a.page, a.box, a.layout_source = h.page, h.box, "pymupdf-heuristic"
                        a.notes += h.notes
                        matched.add(h.asset_id)
                known = {norm_key(a.kind, a.label) for a in assets}
                extra = [h for h in heur if h.asset_id not in matched and norm_key(h.kind, h.label) not in known]
            by_id = {a.asset_id: a for a in assets}
            placed = {it[1] for it in items if it[0] == "asset"}
            tail_ids = [a.asset_id for a in assets if a.asset_id not in placed]
            for h in extra:
                by_id[h.asset_id] = h
                tail_ids.append(h.asset_id)
            if tail_ids:
                items = items + [("heading", 2, _I18N.get(lang, _I18N["en"])["extra"])] + \
                        [("asset", i, "") for i in tail_ids]
            assets = list(by_id.values())
        else:
            mode = "pymupdf-only"
            assets = detect_all_assets(doc)
            items = _items_from_pdf(doc, assets)
            title = (doc.metadata or {}).get("title", "") or aid

        by_id = {a.asset_id: a for a in assets}

        # --- 3. вырезки, таблицы, описания
        for a in assets:
            if a.box is None or not a.page:
                a.notes.append("нет геометрии: только подпись")
                continue
            page = doc[a.page - 1]
            png = out_dir / f"{a.asset_id}.png"
            try:
                if render_clip(page, a.box, png, dpi=dpi, max_px=max_px):
                    a.image_path = f"assets/{aid}/{png.name}"
            except Exception as exc:  # noqa: BLE001
                a.notes.append(f"вырезка не удалась: {exc}")
            if a.kind == "table":
                rows, src = extract_table_rows(page, a.box)
                if not rows and rows_valid(a.tei_rows):
                    rows, src = a.tei_rows, "tei-cells"
                if not rows and vlm and vlm_tables and a.image_path:
                    md = vlm.transcribe_table(png, a.caption)
                    if md.lstrip().startswith("|"):
                        a.table_markdown, a.table_source = md, "vlm"
                if rows:
                    a.table_rows, a.table_source = rows, src
                    a.table_markdown = rows_to_markdown(rows)
                    save_table_csv(rows, out_dir / f"{a.asset_id}.csv")
                elif not a.table_markdown and a.tei_flat_text:
                    a.table_source = "tei-flat-text"
            elif vlm and a.image_path:
                try:
                    a.vlm = vlm.describe_figure(png, a.caption, a.mentions)
                except Exception as exc:  # noqa: BLE001
                    a.notes.append(f"VLM: {exc}")
                    log.warning("%s %s: VLM ошибка %s", aid, a.asset_id, exc)

        # --- 4. выходные файлы
        RAG_DIR.mkdir(parents=True, exist_ok=True)
        (RAG_DIR / f"{aid}.md").write_text(render_article_md(title, items, by_id, lang), encoding="utf-8")
        chunks = build_chunks(aid, title, items, by_id, lang, max_chars=max_chars)
        with (RAG_DIR / f"{aid}.chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        (out_dir / "assets.json").write_text(json.dumps(
            {"article_id": aid, "layout": mode, "assets": [{k: v for k, v in asdict(a).items() if k != "_tei_boxes"}
                                                          for a in assets]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    finally:
        doc.close()

    figs = [a for a in assets if a.kind == "figure"]
    tabs = [a for a in assets if a.kind == "table"]
    jf, jt = jats_counts(aid)
    return {"article_id": aid, "layout": mode, "n_figures": len(figs), "n_tables": len(tabs),
            "figures_with_image": sum(bool(a.image_path) for a in figs),
            "tables_with_structure": sum(bool(a.table_markdown) for a in tabs),
            "vlm_calls": (vlm.calls - calls0) if vlm else 0, "n_chunks": len(chunks),
            "jats_fig": jf if jf is not None else "", "jats_table_wrap": jt if jt is not None else "",
            "status": "ok", "error": ""}

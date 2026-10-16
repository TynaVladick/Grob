"""Тесты логики модуля multimodal (без PyMuPDF, GROBID и сети). Запуск: python -m pytest test_multimodal.py -q"""
from __future__ import annotations

import json
from pathlib import Path

import multimodal as mm

TEI_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0" xmlns:xml="http://www.w3.org/XML/1998/namespace">
 <teiHeader><fileDesc><titleStmt><title level="a" type="main">Sample study</title></titleStmt></fileDesc></teiHeader>
 <facsimile>
  <surface n="1" ulx="0.0" uly="0.0" lrx="612.0" lry="792.0"/>
  <surface n="2" ulx="0.0" uly="0.0" lrx="612.0" lry="792.0"/>
 </facsimile>
 <text xml:lang="en"><body>
  <div><head n="1">Introduction</head>
   <p>Prior work exists. <ref type="figure" target="#fig_0">Figure 1</ref> shows the design.</p>
   <figure xml:id="fig_0" coords="1,100.0,200.0,300.0,150.0">
     <head>Figure 1</head><label>1</label>
     <figDesc>Study design with two arms.</figDesc>
     <graphic coords="1,100.0,180.0,300.0,120.0" type="vector"/>
   </figure>
  </div>
  <div><head n="2.1">Results</head>
   <p>See <ref type="table" target="#tab_0">Table 1</ref> for the outcomes.</p>
   <figure type="table" xml:id="tab_0" coords="2,50.0,60.0,500.0,100.0">
     <head>Table 1</head><label>1</label>
     <figDesc>Table 1. Outcomes by arm.</figDesc>
     <table><row><cell>Arm</cell><cell>N</cell></row><row><cell>A</cell><cell>10</cell></row></table>
   </figure>
   <p>Final paragraph.</p>
  </div>
 </body></text>
</TEI>"""


def _parse(tmp_path: Path) -> dict:
    p = tmp_path / "a.tei.xml"
    p.write_text(TEI_SAMPLE, encoding="utf-8")
    return mm.parse_tei_document(p)


def test_parse_tei_assets_order_and_fields(tmp_path):
    d = _parse(tmp_path)
    assert d["title"] == "Sample study" and d["has_coords"]
    assert d["surfaces"][1] == (612.0, 792.0)
    kinds = [(it[0], it[1] if it[0] == "asset" else None) for it in d["items"]]
    assert ("asset", "fig_0") in kinds and ("asset", "tab_0") in kinds
    assert kinds.index(("asset", "fig_0")) < kinds.index(("asset", "tab_0"))
    fig = next(a for a in d["assets"] if a.asset_id == "fig_0")
    assert fig.label == "Figure 1" and fig.caption == "Study design with two arms."
    assert fig.section == "Introduction" and fig.mentions and "design" in fig.mentions[0]
    assert len(fig._tei_boxes) == 2                      # figure + вложенный graphic
    tab = next(a for a in d["assets"] if a.asset_id == "tab_0")
    assert tab.kind == "table" and tab.label == "Table 1" and tab.tei_rows[1] == ["A", "10"]


def test_heading_level_from_numbering(tmp_path):
    d = _parse(tmp_path)
    heads = [it for it in d["items"] if it[0] == "heading"]
    assert heads[0][1] == 2 and heads[1][1] == 3         # n="1" -> ##, n="2.1" -> ###


def test_coordinate_scaling_and_union():
    boxes = [(1, 100.0, 180.0, 300.0, 120.0), (1, 100.0, 200.0, 300.0, 150.0), (2, 0, 0, 10, 10)]
    b = mm.tei_boxes_to_page_box(boxes, 1, (612.0, 792.0), (612.0, 792.0))
    assert b == (100.0, 180.0, 400.0, 350.0)
    half = mm.tei_boxes_to_page_box(boxes, 1, (612.0, 792.0), (306.0, 396.0))
    assert half == (50.0, 90.0, 200.0, 175.0)             # CropBox вдвое меньше -> масштаб 0.5
    assert mm.tei_boxes_to_page_box(boxes, 3, None, (1, 1)) is None


def test_merge_boxes_touching_and_separate():
    out = mm.merge_boxes([(0, 0, 10, 10), (12, 0, 20, 10), (100, 100, 110, 110)], gap=6)
    assert sorted(out) == [(0, 0, 20, 10), (100, 100, 110, 110)]


def test_norm_key_and_strip_label():
    assert mm.norm_key("figure", "Fig. 3.") == ("figure", "3")
    assert mm.norm_key("table", "Table S2") == ("table", "S2")
    assert mm.strip_label("Figure 1. Study design", "Figure 1") == "Study design"
    assert mm.strip_label("Fig. 2: Flow", "") == "Flow"


def test_rows_to_markdown_escapes_and_pads():
    md = mm.rows_to_markdown([["a|b", "c"], ["1"]])
    assert md.splitlines()[0] == "| a\\|b | c |" and md.splitlines()[2] == "| 1 |  |"
    assert mm.rows_valid([["a", "b"], ["1", "2"]]) and not mm.rows_valid([["a"], ["b"]])


def test_article_md_places_assets_inline(tmp_path):
    d = _parse(tmp_path)
    assets = {a.asset_id: a for a in d["assets"]}
    assets["tab_0"].table_markdown = mm.rows_to_markdown(assets["tab_0"].tei_rows)
    assets["fig_0"].vlm = {"readable": True, "description": "Flow diagram with two arms.",
                           "text_in_image": "Arm A / Arm B", "data_table_markdown": ""}
    md = mm.render_article_md(d["title"], d["items"], assets)
    assert md.index("Prior work") < md.index("asset:begin id=fig_0") < md.index("### Results")
    assert md.index("asset:begin id=tab_0") < md.index("Final paragraph")
    assert md.count("<!-- asset:begin") == 2 and md.count("<!-- asset:end -->") == 2
    assert "**Table 1.** Outcomes by arm." in md and "| Arm | N |" in md
    assert "Flow diagram with two arms." in md
    assert "Figure 1. Figure 1" not in md


def test_chunks_keep_assets_whole_and_carry_metadata(tmp_path):
    d = _parse(tmp_path)
    assets = {a.asset_id: a for a in d["assets"]}
    assets["tab_0"].table_markdown = mm.rows_to_markdown(assets["tab_0"].tei_rows)
    assets["tab_0"].image_path = "assets/x/tab_0.png"
    chunks = mm.build_chunks("PMC1", d["title"], d["items"], assets, max_chars=200)
    typed = [c["type"] for c in chunks]
    assert "figure" in typed and "table" in typed and "text" in typed
    tchunk = next(c for c in chunks if c["type"] == "table")
    assert tchunk["image_path"] == "assets/x/tab_0.png" and "| Arm | N |" in tchunk["text"]
    assert "Mentioned in the text" in tchunk["text"]       # упоминание добавлено только в чанк, не в статью
    assert all(c["text"].startswith("[Section:") or c["type"] != "text" for c in chunks)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_big_table_split_repeats_header():
    rows = [["col1", "col2"]] + [[f"r{i}", "x" * 30] for i in range(200)]
    a = mm.Asset("tab_0", "table", label="Table 1", caption="Big.", table_markdown=mm.rows_to_markdown(rows))
    parts = mm.split_asset_text(mm.render_asset_md(a), 1500)
    assert len(parts) > 1
    for p in parts:
        assert "| col1 | col2 |" in p and "| --- | --- |" in p and len(p) < 2000
    body = "\n".join(parts)
    assert all(f"| r{i} |" in body for i in range(200))    # ни одна строка не потеряна


def test_parse_vlm_json_variants():
    ok = mm.parse_vlm_json('```json\n{"kind":"chart","description":"d","data_table_markdown":"| a |"}\n```')
    assert ok["kind"] == "chart" and ok["description"] == "d" and ok["readable"] is True
    assert mm.parse_vlm_json("just prose")["description"] == "just prose"
    assert mm.parse_vlm_json('{"readable": false}')["readable"] is False


class _Resp:
    def __init__(self, payload): self.payload, self.status_code = payload, 200
    def raise_for_status(self): pass
    def json(self): return self.payload


def test_vlm_payloads_and_cache(tmp_path, monkeypatch):
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG fake")
    seen = []

    def fake_post(self, url, headers=None, json=None, timeout=None):
        seen.append((url, headers, json))
        if "anthropic" in url:
            return _Resp({"content": [{"type": "text", "text": '{"description": "A chart"}'}]})
        return _Resp({"choices": [{"message": {"content": '{"description": "B chart"}'}}]})

    monkeypatch.setattr(mm.requests.Session, "post", fake_post)
    cache = tmp_path / "cache.jsonl"
    v = mm.VLM("anthropic", "m1", api_key="k", cache_path=cache)
    assert v.describe_figure(img, "cap", ["ctx"])["description"] == "A chart"
    url, headers, body = seen[0]
    assert url.endswith("/v1/messages") and headers["x-api-key"] == "k"
    blocks = body["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/png"
    v.describe_figure(img, "cap", ["ctx"])                  # из кэша — без второго запроса
    assert len(seen) == 1 and v.calls == 1
    v2 = mm.VLM("anthropic", "m1", api_key="k", cache_path=cache)   # кэш переживает перезапуск
    v2.describe_figure(img, "cap", ["ctx"])
    assert len(seen) == 1

    o = mm.VLM("openai", "local", base_url="http://localhost:11434/v1")
    assert o.describe_figure(img, "cap", [])["description"] == "B chart"
    url, headers, body = seen[1]
    assert url == "http://localhost:11434/v1/chat/completions" and "authorization" not in headers
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_fetch_tei_with_coords_sends_repeated_params(tmp_path, monkeypatch):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    got = {}

    def fake_post(url, files=None, data=None, timeout=None):
        got.update(url=url, data=data)
        r = _Resp({}); r.content = b"<TEI/>"; return r

    monkeypatch.setattr(mm.requests, "post", fake_post)
    out = tmp_path / "o" / "a.tei.xml"
    assert mm.fetch_tei_with_coords(pdf, out, "http://h:8070/")
    assert got["url"] == "http://h:8070/api/processFulltextDocument"
    assert ("teiCoordinates", "figure") in got["data"] and out.read_bytes() == b"<TEI/>"

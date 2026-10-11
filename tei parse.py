"""Разбор TEI-XML от GROBID в простой словарь с нужными полями.

Извлекаются: название, авторы, год, DOI, аннотация, заголовки разделов, список литературы.
Таблицы, проверка цитат и любые «медицинские выводы» сознательно не затрагиваются.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"t": "http://www.tei-c.org/ns/1.0"}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


_BLOCK = {"p", "head", "div", "note", "item", "label"}


def _txt(el) -> str:
    """Текст элемента; пробел на границах блоков (p/head/div), без разрыва слов внутри строчной разметки."""
    if el is None:
        return ""
    parts: list[str] = []

    def walk(e) -> None:
        block = e.tag.rsplit("}", 1)[-1] in _BLOCK
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


def _year(s: str | None) -> int | None:
    m = re.match(r"\s*(\d{4})", s or "")
    return int(m.group(1)) if m else None


def _person(pers) -> dict:
    forenames = [_txt(f) for f in pers.findall("t:forename", NS) if _txt(f)]
    return {"given": " ".join(forenames), "surname": _txt(pers.find("t:surname", NS))}


def _authors(parent) -> list[dict]:
    out = []
    for a in parent.findall("t:author", NS):
        p = a.find("t:persName", NS)
        if p is not None:
            person = _person(p)
            if person["surname"] or person["given"]:
                out.append(person)
    return out


def _doi(bibl) -> str:
    for idno in bibl.iter("{%s}idno" % NS["t"]):
        if (idno.get("type") or "").upper() == "DOI" and _txt(idno):
            return _txt(idno)
    return ""


def parse_header(root) -> dict:
    src = root.find("t:teiHeader/t:fileDesc/t:sourceDesc/t:biblStruct", NS)
    title_el = root.find("t:teiHeader/t:fileDesc/t:titleStmt/t:title[@type='main']", NS)
    if title_el is None:
        title_el = root.find("t:teiHeader/t:fileDesc/t:titleStmt/t:title", NS)
    title = _txt(title_el)
    authors, doi, year = [], "", None
    if src is not None:
        analytic = src.find("t:analytic", NS)
        if analytic is not None:
            authors = _authors(analytic)
            if not title:
                title = _txt(analytic.find("t:title", NS))
        doi = _doi(src)
        for d in src.findall(".//t:imprint/t:date", NS):
            year = year or _year(d.get("when") or _txt(d))
    if year is None:
        for d in root.findall("t:teiHeader/t:fileDesc/t:publicationStmt/t:date", NS):
            year = year or _year(d.get("when") or _txt(d))
    abstract_el = root.find("t:teiHeader/t:profileDesc/t:abstract", NS)
    return {"title": title, "authors": authors, "year": year, "doi": doi,
            "abstract": _txt(abstract_el)}


def parse_headings(root) -> list[str]:
    """Заголовки разделов: <head> прямо внутри <div> тела статьи (подписи рисунков не берём)."""
    heads = []
    for h in root.findall("t:text/t:body//t:div/t:head", NS):
        s = _txt(h)
        if s:
            heads.append(s)
    return heads


def parse_references(root) -> list[dict]:
    refs = []
    for b in root.findall("t:text/t:back//t:listBibl/t:biblStruct", NS):
        analytic = b.find("t:analytic", NS)
        monogr = b.find("t:monogr", NS)
        title, authors = "", []
        if analytic is not None:
            title = _txt(analytic.find("t:title", NS))
            authors = _authors(analytic)
        if monogr is not None:
            if not title:
                title = _txt(monogr.find("t:title", NS))
            if not authors:
                authors = _authors(monogr)
        year = None
        for d in b.findall(".//t:imprint/t:date", NS):
            year = year or _year(d.get("when") or _txt(d))
        raw = ""
        for n in b.findall("t:note", NS):
            if n.get("type") == "raw_reference":
                raw = _txt(n)
        refs.append({
            "id": b.get(XML_ID, ""), "title": title, "authors": authors,
            "first_author_surname": authors[0]["surname"] if authors else "",
            "year": year, "doi": _doi(b), "raw": raw,
        })
    return refs


def parse_tei(path: Path | str) -> dict:
    root = ET.parse(str(path)).getroot()
    out = parse_header(root)
    out["section_headings"] = parse_headings(root)
    out["references"] = parse_references(root)
    out["n_references"] = len(out["references"])
    return out
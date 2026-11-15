from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin

import httpx

from ..config import Config
from ..store import Store


@dataclass
class Ctx:
    client: httpx.AsyncClient
    cfg: Config
    store: Store
    # Snapshots are committed only after the digest was delivered (see main.run),
    # so a failed run doesn't swallow "new since last time" items.
    pending_kv: dict = field(default_factory=dict)

    def stage_kv(self, k: str, v) -> None:
        self.pending_kv[k] = v

    def commit_kv(self) -> None:
        for k, v in self.pending_kv.items():
            self.store.set_kv(k, v)
        self.pending_kv.clear()

    def conf(self, section: str) -> dict:
        return self.cfg.sources_conf.get(section, {}) or {}

    def since(self, days: int | None = None) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=days if days is not None else self.cfg.lookback_days)


TAG_RE = re.compile(r"<[^>]+>")


def strip_html(s: str | None, limit: int = 3000) -> str:
    return " ".join(html.unescape(TAG_RE.sub(" ", s or "")).split())[:limit]


def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(s)  # RFC 822 (RSS)
        except (TypeError, ValueError):
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


ATOM = "{http://www.w3.org/2005/Atom}"


def parse_feed(xml_text: str) -> list[dict]:
    """Parse RSS 2.0 or Atom into dicts: id, title, link, published, summary."""
    root = ET.fromstring(xml_text.lstrip("﻿").strip())
    out: list[dict] = []
    if root.tag == f"{ATOM}feed":
        for e in root.findall(f"{ATOM}entry"):
            link = ""
            for l in e.findall(f"{ATOM}link"):
                if l.get("rel", "alternate") == "alternate":
                    link = l.get("href", "")
                    break
            out.append({
                "id": (e.findtext(f"{ATOM}id") or link).strip(),
                "title": " ".join((e.findtext(f"{ATOM}title") or "").split()),
                "link": link,
                "published": parse_dt(e.findtext(f"{ATOM}published") or e.findtext(f"{ATOM}updated")),
                "summary": strip_html(e.findtext(f"{ATOM}content") or e.findtext(f"{ATOM}summary")),
            })
    else:
        for e in root.iter("item"):
            link = (e.findtext("link") or "").strip()
            out.append({
                "id": (e.findtext("guid") or link).strip(),
                "title": " ".join((e.findtext("title") or "").split()),
                "link": link,
                "published": parse_dt(e.findtext("pubDate")),
                "summary": strip_html(e.findtext("description")),
            })
    return out


class _Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []
        self.meta: dict[str, str] = {}
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self._href, self._text = a["href"], []
        elif tag == "meta":
            name = (a.get("name") or a.get("property") or "").lower()
            if name in ("description", "og:description", "og:title") and a.get("content"):
                self.meta.setdefault(name, a["content"])
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join("".join(self._text).split())))
            self._href = None
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)
        if self._in_title:
            self.title += data


def extract_links(page_html: str, base_url: str) -> list[tuple[str, str]]:
    p = _Anchors()
    p.feed(page_html)
    seen: dict[str, str] = {}
    for href, text in p.links:
        url = urljoin(base_url, href).split("#")[0]
        if len(text) > len(seen.get(url, "")):
            seen[url] = text
        else:
            seen.setdefault(url, text)
    return list(seen.items())


def page_meta(page_html: str) -> tuple[str, str]:
    """(title, description) of an HTML page."""
    p = _Anchors()
    p.feed(page_html)
    title = p.meta.get("og:title") or " ".join(p.title.split())
    desc = p.meta.get("og:description") or p.meta.get("description") or ""
    return title, desc

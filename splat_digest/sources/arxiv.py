"""arXiv API: https://info.arxiv.org/help/api/user-manual.html"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx

from ..models import PAPER, Item, arxiv_id_from, is_on_topic
from .base import Ctx, parse_dt

log = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
QUERY = 'all:"gaussian splatting" OR all:"gaussian splats" OR abs:3DGS OR ti:3DGS'


def parse_feed(xml_text: str, since: datetime) -> list[Item]:
    root = ET.fromstring(xml_text)
    items: list[Item] = []
    for e in root.findall("a:entry", NS):
        raw_id = (e.findtext("a:id", "", NS) or "").strip()
        aid = arxiv_id_from(raw_id)
        if not aid:
            continue
        published = datetime.fromisoformat(e.findtext("a:published", "", NS).replace("Z", "+00:00"))
        if published < since:
            continue
        title = " ".join((e.findtext("a:title", "", NS) or "").split())
        abstract = " ".join((e.findtext("a:summary", "", NS) or "").split())
        authors = [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)]
        comment = " ".join((e.findtext("arxiv:comment", "", NS) or "").split())
        cat_el = e.find("arxiv:primary_category", NS)
        items.append(
            Item(
                key=f"arxiv:{aid}",
                title=title,
                url=f"https://arxiv.org/abs/{aid}",
                abstract=abstract,
                published=published,
                authors=authors,
                sources=["arxiv"],
                extra={
                    "arxiv_comment": comment,
                    "arxiv_category": cat_el.get("term") if cat_el is not None else "",
                },
            )
        )
    return items


async def fetch(ctx: Ctx) -> list[Item]:
    # `published` is the v1 submission time, but papers are announced 1–4 days later (weekends!).
    # A short window by `published` silently drops them, so look back a week; the sent-set dedupes.
    conf = ctx.conf("arxiv")
    max_results = conf.get("max_results", 500)
    client = ctx.client
    since = ctx.since(conf.get("lookback_days", 7))
    params = {
        "search_query": QUERY,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": max_results,
    }
    r = await client.get(API_URL, params=params, timeout=60)
    r.raise_for_status()
    items = parse_feed(r.text, since)
    # Diagnostics: total matches for the query and the date range actually returned.
    root = ET.fromstring(r.text)
    total = root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    dates = sorted(e.findtext("a:published", "", NS)[:10] for e in root.findall("a:entry", NS))
    log.info("arXiv: %d items since %s (query total=%s, returned=%d, dates %s..%s)", len(items), since.date(),
             total, len(dates), dates[0] if dates else "-", dates[-1] if dates else "-")
    if len(items) >= max_results:
        log.warning("arXiv: hit max_results=%d, some papers may be missing", max_results)
    return items


async def fetch_by_ids(client: httpx.AsyncClient, ids: list[str]) -> list[Item]:
    """Fetch metadata for specific arXiv IDs (used by the awesome-list watcher)."""
    out: list[Item] = []
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        r = await client.get(API_URL, params={"id_list": ",".join(chunk), "max_results": len(chunk)}, timeout=60)
        r.raise_for_status()
        wanted = {f"arxiv:{i}" for i in chunk}
        out += [it for it in parse_feed(r.text, datetime(1990, 1, 1, tzinfo=timezone.utc)) if it.key in wanted]
    return out


# --- Daily announcement RSS (rss.arxiv.org): what arXiv announced today, no search-index lag ----------
RSS_URL = "https://rss.arxiv.org/rss/{cats}"
ANNOUNCE_RE = re.compile(r"Announce Type:\s*([\w-]+)", re.I)
ABSTRACT_RE = re.compile(r"Abstract:\s*(.*)", re.S | re.I)
DC_CREATOR = "{http://purl.org/dc/elements/1.1/}creator"
ARXIV_ANNOUNCE = "{http://arxiv.org/schemas/atom}announce_type"


def parse_rss(xml_text: str, announce_types: set[str]) -> list[Item]:
    root = ET.fromstring(xml_text)
    items: list[Item] = []
    for e in root.iter("item"):
        link = (e.findtext("link") or "").strip()
        aid = arxiv_id_from(link)
        if not aid:
            continue
        desc = e.findtext("description") or ""
        atype = (e.findtext(ARXIV_ANNOUNCE) or "").strip()
        if not atype:
            m = ANNOUNCE_RE.search(desc)
            atype = m.group(1) if m else ""
        if announce_types and atype.lower() not in announce_types:
            continue  # skip "replace" (new versions of old papers)
        title = " ".join((e.findtext("title") or "").split())
        m = ABSTRACT_RE.search(desc)
        abstract = " ".join((m.group(1) if m else desc).split())
        if not is_on_topic(title, abstract):
            continue
        creators = e.findtext(DC_CREATOR) or ""
        cats = [c.text for c in e.findall("category") if c.text]
        items.append(Item(
            key=f"arxiv:{aid}",
            title=title,
            url=f"https://arxiv.org/abs/{aid}",
            abstract=abstract,
            published=parse_dt(e.findtext("pubDate")),
            authors=[a.strip() for a in creators.split(",") if a.strip()],
            sources=["arxiv"],
            extra={"arxiv_category": cats[0] if cats else ""},
        ))
    return items


async def fetch_rss(ctx: Ctx) -> list[Item]:
    conf = ctx.conf("arxiv_rss")
    cats = "+".join(conf.get("categories", ["cs.CV", "cs.GR", "cs.RO"]))
    types = {t.lower() for t in conf.get("announce_types", ["new", "cross"])}
    r = await ctx.client.get(RSS_URL.format(cats=cats), timeout=60)
    r.raise_for_status()
    items = parse_rss(r.text, types)
    log.info("arXiv RSS (%s): %d on-topic of %d announced", cats, len(items), r.text.count("<item>"))
    return items

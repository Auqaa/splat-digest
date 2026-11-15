"""arXiv API: https://info.arxiv.org/help/api/user-manual.html"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx

from ..models import PAPER, Item, arxiv_id_from
from .base import Ctx

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


async def fetch(ctx: Ctx, max_results: int = 200) -> list[Item]:
    client = ctx.client
    since = ctx.since()
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
    log.info("arXiv: %d items since %s", len(items), since.date())
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

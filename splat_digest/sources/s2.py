"""Semantic Scholar bulk search: conference papers not on arXiv + venue/citations enrichment.
https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_bulk_search"""
from __future__ import annotations

import asyncio
import logging

from ..models import PAPER, Item, is_on_topic
from .base import Ctx, parse_dt

log = logging.getLogger(__name__)
URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "title,abstract,externalIds,venue,publicationDate,citationCount,url,authors"


async def fetch(ctx: Ctx) -> list[Item]:
    conf = ctx.conf("s2")
    days = conf.get("lookback_days", 21)  # S2 indexes conference papers with a delay
    params = {
        "query": conf.get("query", '"gaussian splatting" | "gaussian splats" | 3DGS'),
        "fields": FIELDS,
        "publicationDateOrYear": f"{ctx.since(days).date().isoformat()}:",
        "sort": "publicationDate:desc",
    }
    headers = {"x-api-key": ctx.cfg.s2_api_key} if ctx.cfg.s2_api_key else {}
    for attempt in range(4):
        r = await ctx.client.get(URL, params=params, headers=headers, timeout=60)
        if r.status_code != 429:
            break
        await asyncio.sleep(5 * (attempt + 1))  # unauthenticated pool is shared and often throttled
    r.raise_for_status()
    items = []
    for p in (r.json().get("data") or [])[: conf.get("max_items", 100)]:
        title, abstract = p.get("title") or "", p.get("abstract") or ""
        if not is_on_topic(title, abstract):
            continue
        aid = (p.get("externalIds") or {}).get("ArXiv")
        items.append(Item(
            key=f"arxiv:{aid}" if aid else f"s2:{p['paperId']}",
            title=title,
            url=f"https://arxiv.org/abs/{aid}" if aid else p.get("url", ""),
            abstract=abstract,
            published=parse_dt(p.get("publicationDate")),
            authors=[a.get("name", "") for a in p.get("authors") or []],
            sources=["s2"],
            kind=PAPER,
            extra={"venue": p.get("venue") or "", "citations": p.get("citationCount"), "s2_url": p.get("url")},
        ))
    log.info("Semantic Scholar: %d", len(items))
    return items

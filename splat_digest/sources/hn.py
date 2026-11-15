"""Hacker News via Algolia: https://hn.algolia.com/api"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from ..models import Item, is_on_topic, make_key

log = logging.getLogger(__name__)

API_URL = "https://hn.algolia.com/api/v1/search_by_date"
QUERIES = ["gaussian splatting", "gaussian splat", "3DGS"]


def parse(hits: list[dict]) -> list[Item]:
    items: list[Item] = []
    for h in hits:
        title = h.get("title") or ""
        url = h.get("url") or ""
        text = h.get("story_text") or ""
        # Algolia does typo-tolerant matching; re-check locally.
        if not is_on_topic(title, url, text):
            continue
        oid = h.get("objectID")
        hn_url = f"https://news.ycombinator.com/item?id={oid}"
        ts = h.get("created_at_i")
        items.append(
            Item(
                key=make_key(url, f"hn:{oid}"),
                title=title,
                url=url or hn_url,
                abstract=text[:2000],
                published=datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None,
                sources=["hn"],
                extra={"hn_url": hn_url, "hn_points": h.get("points"), "hn_comments": h.get("num_comments")},
            )
        )
    return items


async def fetch(client: httpx.AsyncClient, lookback_days: int) -> list[Item]:
    since = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
    out: dict[str, Item] = {}
    for q in QUERIES:
        params = {
            "query": q,
            "tags": "story",
            "numericFilters": f"created_at_i>{since}",
            "hitsPerPage": 100,
        }
        r = await client.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        for it in parse(r.json().get("hits", [])):
            out.setdefault(it.key, it)
    log.info("Hacker News: %d on-topic stories", len(out))
    return list(out.values())

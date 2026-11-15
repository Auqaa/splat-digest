"""OpenReview API v2 full-text search (ICLR/NeurIPS/... submissions)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..models import PAPER, Item, is_on_topic
from .base import Ctx

log = logging.getLogger(__name__)
URL = "https://api2.openreview.net/notes/search"


def _v(content: dict, k: str) -> str:
    x = content.get(k)
    return (x.get("value") if isinstance(x, dict) else x) or ""


def parse(notes: list[dict], since: datetime) -> list[Item]:
    items = []
    for n in notes:
        c = n.get("content") or {}
        title, abstract = _v(c, "title"), _v(c, "abstract")
        ts = n.get("cdate", n.get("tcdate"))
        published = datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if ts is not None else None
        if published and published < since:
            continue
        if not is_on_topic(title, abstract):
            continue
        forum = n.get("forum") or n.get("id")
        items.append(Item(
            key=f"openreview:{forum}",
            title=" ".join(str(title).split()),
            url=f"https://openreview.net/forum?id={forum}",
            abstract=" ".join(str(abstract).split()),
            published=published,
            sources=["openreview"],
            kind=PAPER,
            extra={"venue": _v(c, "venue")},
        ))
    return items


async def fetch(ctx: Ctx) -> list[Item]:
    conf = ctx.conf("openreview")
    since = ctx.since(conf.get("lookback_days", 14))
    out: dict[str, Item] = {}
    for term in conf.get("terms", ["gaussian splatting"]):
        r = await ctx.client.get(
            URL, params={"term": term, "type": "terms", "content": "all", "source": "forum", "limit": 200}, timeout=60
        )
        r.raise_for_status()
        for it in parse(r.json().get("notes", []), since):
            out.setdefault(it.key, it)
    log.info("OpenReview: %d", len(out))
    return list(out.values())

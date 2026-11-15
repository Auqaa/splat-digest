"""Reddit via public RSS. Reddit often returns 403 to datacenter IPs (incl. GitHub Actions)."""
from __future__ import annotations

import logging

from ..models import COMMUNITY, Item, is_on_topic, make_key
from .base import Ctx, parse_feed

log = logging.getLogger(__name__)


async def fetch(ctx: Ctx) -> list[Item]:
    conf = ctx.conf("reddit")
    since = ctx.since()
    items: list[Item] = []
    errors = []
    for sub in conf.get("subreddits", [{"name": "GaussianSplatting", "filter": False}]):
        name, need_kw = sub["name"], sub.get("filter", True)
        r = await ctx.client.get(f"https://www.reddit.com/r/{name}/new/.rss", params={"limit": 100}, timeout=30)
        if r.status_code in (403, 429):
            errors.append(f"r/{name}: HTTP {r.status_code}")
            continue
        r.raise_for_status()
        n = 0
        for e in parse_feed(r.text):
            if e["published"] and e["published"] < since:
                continue
            if need_kw and not is_on_topic(e["title"], e["summary"]):
                continue
            pid = e["id"].rsplit("_", 1)[-1] if e["id"] else e["link"]
            items.append(Item(
                key=make_key(None, f"reddit:{pid}", e["summary"]),
                title=e["title"],
                url=e["link"],
                abstract=e["summary"],
                published=e["published"],
                sources=["reddit"],
                kind=COMMUNITY,
                extra={"reddit_url": e["link"], "subreddit": name},
            ))
            n += 1
            if n >= sub.get("max_items", 25):
                break
    if errors and not items:
        raise RuntimeError("; ".join(errors))
    log.info("Reddit: %d (%s)", len(items), "; ".join(errors) or "ok")
    return items

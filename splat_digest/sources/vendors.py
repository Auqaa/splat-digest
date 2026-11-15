"""Company/news sites: RSS feeds or "new links on a listing page" watchers."""
from __future__ import annotations

import asyncio
import logging
import re

from ..models import INDUSTRY, Item, is_on_topic
from .base import Ctx, extract_links, page_meta, parse_feed

log = logging.getLogger(__name__)

# Vendor blogs are mostly not about 3DGS; keep reality-capture adjacent posts too.
VENDOR_KW = re.compile(
    r"gaussian|splat|3dgs|radiance field|nerf|reconstruct|digital twin|reality capture|photogrammetr|"
    r"lidar|3d scan|point cloud|mesh|mapping",
    re.I,
)


async def _rss(ctx: Ctx, v: dict) -> list[Item]:
    r = await ctx.client.get(v["url"], timeout=30)
    r.raise_for_status()
    since = ctx.since(v.get("lookback_days", 7))
    items = []
    for e in parse_feed(r.text):
        if e["published"] and e["published"] < since:
            continue
        items.append(_item(v, e["link"], e["title"], e["summary"], e["published"]))
    return items


async def _links(ctx: Ctx, v: dict) -> list[Item]:
    r = await ctx.client.get(v["url"], timeout=30)
    r.raise_for_status()
    pattern = re.compile(v["link_pattern"])
    links = {u: t for u, t in extract_links(r.text, str(r.url)) if pattern.search(u)}
    if not links:
        raise RuntimeError(f"0 ссылок по шаблону {v['link_pattern']!r} (сайт рендерится JS или сменил вёрстку)")
    kv = f"links:{v['url']}"
    prev = ctx.store.get_kv(kv)
    ctx.stage_kv(kv, sorted(set(links) | set(prev or [])))
    if prev is None:
        log.info("vendor %s: baseline of %d links", v["name"], len(links))
        return []
    new = [u for u in links if u not in set(prev)][: v.get("max_new", 10)]
    items = []
    for u in new:
        title, desc = links[u], ""
        try:  # fetch the article for title/description
            pr = await ctx.client.get(u, timeout=30)
            if pr.status_code == 200:
                t, desc = page_meta(pr.text)
                title = t or title
        except Exception as e:  # noqa: BLE001 - best effort enrichment
            log.debug("meta fetch failed %s: %r", u, e)
        items.append(_item(v, u, title or u, desc, None))
    return items


def _item(v: dict, url: str, title: str, summary: str, published) -> Item:
    return Item(
        key=f"web:{url.rstrip('/')}",
        title=f"{v['name']}: {title}",
        url=url,
        abstract=summary,
        published=published,
        sources=[v["name"]],
        kind=v.get("kind", INDUSTRY),
    )


async def fetch(ctx: Ctx) -> list[Item]:
    vendors = ctx.conf("vendors").get("sites", [])

    async def one(v: dict) -> list[Item]:
        items = await (_rss(ctx, v) if v.get("type", "links") == "rss" else _links(ctx, v))
        if not v.get("all", False):
            items = [i for i in items if is_on_topic(i.title, i.abstract, pattern=VENDOR_KW)]
        return items

    res = await asyncio.gather(*(one(v) for v in vendors), return_exceptions=True)
    items: list[Item] = []
    failed = []
    for v, r in zip(vendors, res):
        if isinstance(r, BaseException):
            log.warning("vendor %s failed: %s", v["name"], r)
            failed.append(v["name"])
        else:
            items += r
    if failed and len(failed) == len(vendors):
        raise RuntimeError(f"все сайты недоступны: {failed}")
    log.info("Vendors: %d items (%d sites failed)", len(items), len(failed))
    return items


async def check(ctx: Ctx) -> list[tuple[str, str]]:
    """Per-site status for `check-sources`."""
    out = []
    for v in ctx.conf("vendors").get("sites", []):
        try:
            if v.get("type", "links") == "rss":
                r = await ctx.client.get(v["url"], timeout=30)
                r.raise_for_status()
                n = len(parse_feed(r.text))
                out.append((v["name"], f"OK, {n} записей в RSS"))
            else:
                r = await ctx.client.get(v["url"], timeout=30)
                r.raise_for_status()
                pattern = re.compile(v["link_pattern"])
                links = [u for u, _ in extract_links(r.text, str(r.url)) if pattern.search(u)]
                out.append((v["name"], f"{'OK' if links else 'ПУСТО'}, {len(links)} ссылок; пример: {links[:2]}"))
        except Exception as e:  # noqa: BLE001
            out.append((v["name"], f"ОШИБКА: {type(e).__name__}: {str(e)[:150]}"))
    return out

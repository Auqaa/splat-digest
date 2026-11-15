"""Bluesky searchPosts. Auth (app password) is optional but public search may be restricted."""
from __future__ import annotations

import logging

from ..models import COMMUNITY, Item, is_on_topic, make_key
from .base import Ctx, parse_dt

log = logging.getLogger(__name__)
PUBLIC = "https://public.api.bsky.app/xrpc"
PDS = "https://bsky.social/xrpc"


async def _auth(ctx: Ctx) -> tuple[str, dict]:
    if not (ctx.cfg.bluesky_handle and ctx.cfg.bluesky_app_password):
        return PUBLIC, {}
    r = await ctx.client.post(
        f"{PDS}/com.atproto.server.createSession",
        json={"identifier": ctx.cfg.bluesky_handle, "password": ctx.cfg.bluesky_app_password},
        timeout=30,
    )
    r.raise_for_status()
    return PDS, {"Authorization": f"Bearer {r.json()['accessJwt']}"}


def parse(posts: list[dict], min_likes: int) -> list[Item]:
    items = []
    for p in posts:
        rec = p.get("record") or {}
        text = rec.get("text") or ""
        ext = ((p.get("embed") or {}).get("external") or {})
        link = ext.get("uri") or ""
        if not is_on_topic(text, ext.get("title"), ext.get("description"), link):
            continue
        if (p.get("likeCount") or 0) < min_likes:
            continue
        handle = (p.get("author") or {}).get("handle", "")
        rkey = p.get("uri", "").rsplit("/", 1)[-1]
        post_url = f"https://bsky.app/profile/{handle}/post/{rkey}"
        items.append(Item(
            key=make_key(link, f"bsky:{p.get('uri')}", text),
            title=(ext.get("title") or text.split("\n")[0])[:200],
            url=link or post_url,
            abstract=f"{text}\n{ext.get('description') or ''}".strip(),
            published=parse_dt(rec.get("createdAt") or p.get("indexedAt")),
            sources=["bluesky"],
            kind=COMMUNITY,
            extra={"bsky_url": post_url, "bsky_likes": p.get("likeCount")},
        ))
    return items


async def fetch(ctx: Ctx) -> list[Item]:
    conf = ctx.conf("bluesky")
    base, headers = await _auth(ctx)
    since = ctx.since().strftime("%Y-%m-%dT%H:%M:%SZ")
    out: dict[str, Item] = {}
    for q in conf.get("queries", ["gaussian splatting", "3DGS"]):
        r = await ctx.client.get(
            f"{base}/app.bsky.feed.searchPosts",
            params={"q": q, "sort": "top", "since": since, "limit": 100},
            headers=headers,
            timeout=30,
        )
        if r.status_code in (401, 403) and not headers:
            raise RuntimeError("Bluesky search требует авторизации: задайте BLUESKY_HANDLE и BLUESKY_APP_PASSWORD")
        r.raise_for_status()
        for it in parse(r.json().get("posts", []), conf.get("min_likes", 5)):
            out.setdefault(it.key, it)
    log.info("Bluesky: %d", len(out))
    return list(out.values())

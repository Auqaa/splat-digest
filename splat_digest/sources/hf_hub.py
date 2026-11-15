"""New Hugging Face models / Spaces about gaussian splatting."""
from __future__ import annotations

import logging

from ..models import TOOL, Item, is_on_topic
from .base import Ctx, parse_dt

log = logging.getLogger(__name__)


async def fetch(ctx: Ctx) -> list[Item]:
    conf = ctx.conf("hf_hub")
    since = ctx.since()
    min_likes = conf.get("min_likes", 0)
    out: dict[str, Item] = {}
    for kind in conf.get("kinds", ["spaces", "models"]):
        for term in conf.get("terms", ["gaussian-splatting", "gaussian splatting", "3dgs", "gsplat"]):
            r = await ctx.client.get(
                f"https://huggingface.co/api/{kind}",
                params={"search": term, "sort": "createdAt", "direction": -1, "limit": 50},
                timeout=30,
            )
            r.raise_for_status()
            for m in r.json():
                rid = m.get("id") or m.get("modelId")
                created = parse_dt(m.get("createdAt"))
                if not rid or (created and created < since) or (m.get("likes") or 0) < min_likes:
                    continue
                tags = [t for t in m.get("tags") or [] if ":" not in t][:15]
                if not is_on_topic(rid, " ".join(tags), term):
                    continue
                url = f"https://huggingface.co/{'spaces/' if kind == 'spaces' else ''}{rid}"
                out.setdefault(url, Item(
                    key=f"hf-{kind}:{rid}",
                    title=f"HF {kind[:-1]}: {rid}",
                    url=url,
                    abstract=f"Tags: {', '.join(tags)}. SDK/pipeline: {m.get('sdk') or m.get('pipeline_tag') or ''}",
                    published=created,
                    sources=["hf_hub"],
                    kind=TOOL,
                    extra={"hf_likes": m.get("likes")},
                ))
    items = list(out.values())[: conf.get("max_items", 20)]
    log.info("HF Hub: %d", len(items))
    return items

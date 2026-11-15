"""Hugging Face Daily Papers: GET https://huggingface.co/api/daily_papers?date=YYYY-MM-DD"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import httpx

from ..models import Item, is_on_topic

log = logging.getLogger(__name__)

API_URL = "https://huggingface.co/api/daily_papers"


def _dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse(entries: list[dict]) -> list[Item]:
    items: list[Item] = []
    for entry in entries:
        paper = entry.get("paper") or {}
        pid = paper.get("id")
        title = " ".join((paper.get("title") or entry.get("title") or "").split())
        abstract = " ".join((paper.get("summary") or entry.get("summary") or "").split())
        if not pid or not is_on_topic(title, abstract):
            continue
        items.append(
            Item(
                key=f"arxiv:{pid}",
                title=title,
                url=f"https://arxiv.org/abs/{pid}",
                abstract=abstract,
                published=_dt(paper.get("publishedAt") or entry.get("publishedAt")),
                authors=[a.get("name", "") for a in paper.get("authors", []) if isinstance(a, dict)],
                sources=["hf"],
                extra={
                    "hf_url": f"https://huggingface.co/papers/{pid}",
                    "hf_upvotes": paper.get("upvotes"),
                },
            )
        )
    return items


async def fetch(client: httpx.AsyncClient, lookback_days: int) -> list[Item]:
    today = datetime.now(timezone.utc).date()
    out: dict[str, Item] = {}
    for d in range(lookback_days + 1):
        day: date = today - timedelta(days=d)
        r = await client.get(API_URL, params={"date": day.isoformat()}, timeout=30)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        data = r.json()
        for it in parse(data if isinstance(data, list) else []):
            out.setdefault(it.key, it)
    log.info("HF Daily Papers: %d on-topic items", len(out))
    return list(out.values())

from __future__ import annotations

from typing import Awaitable, Callable

from ..models import Item
from . import arxiv, bluesky, github, hf, hf_hub, hn, openreview, reddit, s2, vendors
from .base import Ctx

Fetcher = Callable[[Ctx], Awaitable[list[Item]]]

REGISTRY: dict[str, tuple[str, Fetcher]] = {
    "arxiv_rss": ("arXiv RSS (сегодняшние объявления)", arxiv.fetch_rss),
    "arxiv": ("arXiv API (поиск, страховка)", arxiv.fetch),
    "hf_daily": ("HF Daily Papers", hf.fetch),
    "hn": ("Hacker News", hn.fetch),
    "s2": ("Semantic Scholar", s2.fetch),
    "openreview": ("OpenReview", openreview.fetch),
    "github_releases": ("GitHub releases", github.releases),
    "github_repos": ("GitHub new repos", github.new_repos),
    "awesome": ("awesome-lists", github.awesome),
    "hf_hub": ("HF models/Spaces", hf_hub.fetch),
    "reddit": ("Reddit", reddit.fetch),
    "bluesky": ("Bluesky", bluesky.fetch),
    "vendors": ("Сайты компаний", vendors.fetch),
}

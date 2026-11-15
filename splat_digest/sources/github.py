"""GitHub: release feeds of key tools, newly created repos gaining stars, awesome-list diffs."""
from __future__ import annotations

import asyncio
import logging
import re

from ..models import TOOL, Item, arxiv_ids_in, is_on_topic
from . import arxiv
from .base import Ctx, parse_feed

log = logging.getLogger(__name__)


async def releases(ctx: Ctx) -> list[Item]:
    conf = ctx.conf("github_releases")
    repos: list[str] = conf.get("repos", [])
    since = ctx.since(conf.get("lookback_days", ctx.cfg.lookback_days))

    async def one(repo: str) -> list[Item]:
        r = await ctx.client.get(f"https://github.com/{repo}/releases.atom", timeout=30)
        if r.status_code == 404:
            log.warning("github releases: repo %s not found", repo)
            return []
        r.raise_for_status()
        out = []
        for e in parse_feed(r.text):
            if e["published"] and e["published"] < since:
                continue
            out.append(Item(
                key=f"gh-release:{repo}:{e['id'].rsplit('/', 1)[-1]}",
                title=f"{repo} — {e['title']}",
                url=e["link"] or f"https://github.com/{repo}/releases",
                abstract=e["summary"],
                published=e["published"],
                sources=["github"],
                kind=TOOL,
                extra={"repo": repo},
            ))
        return out

    res = await asyncio.gather(*(one(r) for r in repos), return_exceptions=True)
    items: list[Item] = []
    failed = []
    for repo, r in zip(repos, res):
        if isinstance(r, BaseException):
            log.warning("github releases %s failed: %r", repo, r)
            failed.append(repo)
        else:
            items += r
    if repos and len(failed) == len(repos):
        raise RuntimeError(f"все {len(repos)} лент релизов недоступны")
    log.info("GitHub releases: %d", len(items))
    return items


def _gh_headers(ctx: Ctx) -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if ctx.cfg.github_token:
        h["Authorization"] = f"Bearer {ctx.cfg.github_token}"
    return h


async def new_repos(ctx: Ctx) -> list[Item]:
    """Repos created in the last N days that already have some stars."""
    conf = ctx.conf("github_repos")
    days = conf.get("created_within_days", 14)
    min_stars = conf.get("min_stars", 15)
    queries = conf.get("queries", ['"gaussian splatting"', "3dgs", "topic:gaussian-splatting"])
    since = ctx.since(days).date().isoformat()
    out: dict[str, Item] = {}
    for q in queries:
        r = await ctx.client.get(
            "https://api.github.com/search/repositories",
            params={"q": f"{q} created:>={since} stars:>={min_stars}", "sort": "stars", "order": "desc", "per_page": 50},
            headers=_gh_headers(ctx),
            timeout=30,
        )
        if r.status_code in (403, 429):
            log.warning("GitHub search rate-limited (задайте GITHUB_TOKEN)")
            break
        r.raise_for_status()
        for repo in r.json().get("items", []):
            name = repo["full_name"]
            desc = repo.get("description") or ""
            topics = repo.get("topics") or []
            if not is_on_topic(name, desc, " ".join(topics), repo.get("homepage")):
                continue
            out.setdefault(name, Item(
                key=f"gh-repo:{name}",
                title=f"{name}: {desc[:150]}" if desc else name,
                url=repo["html_url"],
                abstract=f"{desc}\nTopics: {', '.join(topics)}\nLanguage: {repo.get('language')}\n"
                         f"Homepage: {repo.get('homepage') or ''}",
                sources=["github"],
                kind=TOOL,
                extra={"gh_stars": repo.get("stargazers_count"), "arxiv_refs": arxiv_ids_in(
                    f"{desc} {repo.get('homepage') or ''}")},
            ))
    log.info("GitHub new repos: %d", len(out))
    return list(out.values())


ENTRY_SPLIT = re.compile(r"^- id:", re.M)
FIELD = re.compile(r"^  (title|paper|code|project_page): *(.+)$", re.M)


def parse_awesome(text: str) -> dict[str, dict]:
    """arXiv id -> {title, code, project_page}. Understands MrNeRF's YAML; falls back to any arXiv links."""
    out: dict[str, dict] = {}
    if ENTRY_SPLIT.search(text):
        for block in ENTRY_SPLIT.split(text)[1:]:
            f = {k: v.strip().strip("'\"") for k, v in FIELD.findall(block)}
            ids = arxiv_ids_in(f.get("paper", ""))
            if ids:
                out[ids[0]] = {k: v for k, v in f.items() if v and v != "null"}
    for aid in arxiv_ids_in(text):
        out.setdefault(aid, {})
    return out


async def awesome(ctx: Ctx) -> list[Item]:
    """New arXiv IDs appearing in curated awesome lists. First run only stores a baseline."""
    conf = ctx.conf("awesome")
    items: list[Item] = []
    for url in conf.get("urls", []):
        r = await ctx.client.get(url, timeout=60)
        r.raise_for_status()
        entries = parse_awesome(r.text)
        ids = list(entries)
        if not ids:
            log.warning("awesome %s: no arXiv links found", url)
            continue
        kv = f"awesome:{url}"
        prev = ctx.store.get_kv(kv)
        ctx.stage_kv(kv, ids)
        if prev is None:
            log.info("awesome %s: baseline of %d IDs stored", url, len(ids))
            continue
        new = [i for i in ids if i not in set(prev)]
        if new:
            for it in await arxiv.fetch_by_ids(ctx.client, new[:50]):
                aid = it.key.split(":", 1)[1]
                it.sources = ["awesome"]
                code = entries.get(aid, {}).get("code")
                if code:
                    it.extra["code_url"] = code
                items.append(it)
    log.info("awesome lists: %d new", len(items))
    return items

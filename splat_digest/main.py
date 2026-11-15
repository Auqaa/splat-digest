from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

import httpx

from . import digest, telegram
from .config import Config
from .llm import Analysis, LLMClient
from .models import PAPER, Item, norm_title
from .sources import REGISTRY, vendors
from .sources.base import Ctx
from .store import Store

log = logging.getLogger("splat_digest")
UA = "splat-digest/0.2 (personal research digest bot)"


def dedupe(raw: list[Item]) -> list[Item]:
    by_key: dict[str, Item] = {}
    for it in raw:
        if it.key.startswith("arxiv:"):
            it.kind = PAPER
        if it.key in by_key:
            by_key[it.key].merge(it)
        else:
            by_key[it.key] = it
    # Second pass: same title from sources without arXiv IDs (OpenReview, S2, blogs).
    by_title: dict[str, Item] = {}
    out: list[Item] = []
    for it in sorted(by_key.values(), key=lambda i: not i.key.startswith("arxiv:")):
        t = norm_title(it.title)
        if len(t) >= 20 and t in by_title:
            by_title[t].merge(it)
            continue
        if len(t) >= 20:
            by_title[t] = it
        out.append(it)
    # Repos whose description links a paper in this batch: attach as code link instead of separate item.
    papers = {i.key: i for i in out if i.key.startswith("arxiv:")}
    result = []
    for it in out:
        refs = [f"arxiv:{a}" for a in it.extra.get("arxiv_refs", [])]
        target = next((papers[r] for r in refs if r in papers), None)
        if target is not None and it.key.startswith("gh-repo:"):
            target.extra.setdefault("code_url", it.url)
            target.extra.setdefault("gh_stars", it.extra.get("gh_stars"))
            continue
        result.append(it)
    return result


async def collect(ctx: Ctx, names: tuple[str, ...]) -> tuple[list[Item], list[str], int]:
    """Returns (deduped items, notes, number of sources that worked)."""
    enabled = [(n, *REGISTRY[n]) for n in names]
    results = await asyncio.gather(*(fn(ctx) for _, _, fn in enabled), return_exceptions=True)
    raw: list[Item] = []
    notes: list[str] = []
    ok = 0
    for (_, title, _), res in zip(enabled, results):
        if isinstance(res, BaseException):
            log.error("source %s failed: %r", title, res)
            notes.append(f"{title}: {type(res).__name__}: {str(res)[:120]}")
            continue
        ok += 1
        raw += res
    return dedupe(raw), notes, ok


async def score_all(
    llm: LLMClient, store: Store, items: list[Item], concurrency: int
) -> tuple[list[tuple[Item, Analysis]], int]:
    sem = asyncio.Semaphore(concurrency)
    failed = 0

    async def one(it: Item) -> tuple[Item, Analysis] | None:
        nonlocal failed
        cached = store.get_analysis(it.key)
        if cached:
            return it, Analysis.from_json(cached)
        async with sem:
            try:
                a = await llm.analyze(it)
            except RuntimeError as e:
                log.error("%s", e)
                failed += 1
                return None
        store.save_analysis(it.key, it.title, a.to_dict())
        return it, a

    res = await asyncio.gather(*(one(i) for i in items))
    return [r for r in res if r], failed


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(headers={"User-Agent": UA}, follow_redirects=True)


def select_new(items: list[Item], store: Store, cap: int) -> tuple[list[Item], list[str]]:
    sent = store.sent_keys([i.key for i in items])
    new = [i for i in items if i.key not in sent]
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    # Papers first (the main point of the digest), then by recency.
    new.sort(key=lambda i: (i.kind != PAPER, -(i.published or oldest).timestamp() if i.published else 0))
    notes = []
    if len(new) > cap:
        notes.append(f"новых {len(new)}, обработано только {cap} (MAX_ITEMS_LLM); остальные — в следующий раз")
        new = new[:cap]
    return new, notes


async def run(cfg: Config, dry_run: bool = False) -> int:
    store = Store(cfg.state_path)
    async with _client() as client:
        llm = LLMClient(client, cfg.llm_api_key, cfg.llm_model, cfg.llm_base_url, cfg.llm_thinking,
                        cfg.llm_max_tokens)
        await llm.check_model()
        ctx = Ctx(client, cfg, store)

        items, notes, ok = await collect(ctx, cfg.sources)
        if ok == 0:
            raise SystemExit("Все источники недоступны — дайджест не отправлен")
        new, cap_notes = select_new(items, store, cfg.max_items_llm)
        notes += cap_notes
        log.info("collected=%d new=%d", len(items), len(new))

        if not new and not cfg.send_empty_digest:
            ctx.commit_kv()
            return 0

        scored, failed = await score_all(llm, store, new, cfg.llm_concurrency)
        log.info("LLM usage: %s", llm.usage)
        chunks = digest.build_digest(scored, cfg.min_relevance, datetime.now(timezone.utc).date(), failed, notes)

        if dry_run:
            print("\n\n=====\n\n".join(chunks))
            return 0
        await telegram.send_messages(client, cfg.telegram_bot_token, cfg.telegram_chat_id, chunks)
        # Failed-LLM items are NOT marked sent: they will be retried next run.
        store.mark_sent([i.key for i, _ in scored])
        ctx.commit_kv()
        log.info("sent %d messages, %d items", len(chunks), len(scored))
    return 0


async def seed(cfg: Config) -> int:
    """Mark everything currently visible as already sent, without LLM/Telegram (first launch)."""
    store = Store(cfg.state_path)
    async with _client() as client:
        ctx = Ctx(client, cfg, store)
        items, notes, _ = await collect(ctx, cfg.sources)
        store.mark_sent([i.key for i in items])
        ctx.commit_kv()
    print(f"Помечено как отправленное: {len(items)}. Проблемы: {notes or 'нет'}")
    return 0


async def check_sources(cfg: Config) -> int:
    store = Store(":memory:")
    async with _client() as client:
        ctx = Ctx(client, cfg, store)
        for name in cfg.sources:
            title, fn = REGISTRY[name]
            try:
                items = await fn(ctx)
                print(f"✅ {title}: {len(items)}")
                for it in items[:3]:
                    print(f"     · {it.title[:100]}")
            except Exception as e:  # noqa: BLE001
                print(f"❌ {title}: {type(e).__name__}: {str(e)[:200]}")
        if "vendors" in cfg.sources:
            print("\nСайты компаний (первый запуск watcher'а только запоминает ссылки):")
            for n, status in await vendors.check(ctx):
                print(f"   {n}: {status}")
    return 0


def cli() -> None:
    p = argparse.ArgumentParser(prog="splat-digest")
    p.add_argument("command", nargs="?", default="run", choices=["run", "dry-run", "seed", "check-sources", "chat-id"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # don't log URLs with the bot token

    if args.command == "chat-id":
        cfg = Config.from_env(require_secrets=False)
        if not cfg.telegram_bot_token:
            sys.exit("Задайте TELEGRAM_BOT_TOKEN")

        async def _ids() -> None:
            async with httpx.AsyncClient() as c:
                await telegram.print_chat_ids(c, cfg.telegram_bot_token)

        asyncio.run(_ids())
        return

    cfg = Config.from_env(require_secrets=args.command == "run")
    if args.command == "seed":
        sys.exit(asyncio.run(seed(cfg)))
    if args.command == "check-sources":
        sys.exit(asyncio.run(check_sources(cfg)))
    sys.exit(asyncio.run(run(cfg, dry_run=args.command == "dry-run")))


if __name__ == "__main__":
    cli()

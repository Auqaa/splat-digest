from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

import httpx

from . import digest, telegram
from .config import Config
from .llm import Analysis, DeepSeekClient
from .models import Item
from .sources import arxiv, hf, hn
from .store import Store

log = logging.getLogger("splat_digest")
UA = "splat-digest/0.1 (personal research digest)"


async def collect(client: httpx.AsyncClient, lookback_days: int) -> tuple[list[Item], list[str]]:
    names = ["arXiv", "HF Daily Papers", "Hacker News"]
    results = await asyncio.gather(
        arxiv.fetch(client, lookback_days),
        hf.fetch(client, lookback_days),
        hn.fetch(client, lookback_days),
        return_exceptions=True,
    )
    merged: dict[str, Item] = {}
    notes: list[str] = []
    for name, res in zip(names, results):
        if isinstance(res, BaseException):
            log.error("source %s failed: %r", name, res)
            notes.append(f"источник {name} недоступен: {type(res).__name__}")
            continue
        for it in res:
            if it.key in merged:
                merged[it.key].merge(it)
            else:
                merged[it.key] = it
    return list(merged.values()), notes


async def score_all(
    ds: DeepSeekClient, store: Store, items: list[Item], concurrency: int
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
                a = await ds.analyze(it)
            except RuntimeError as e:
                log.error("%s", e)
                failed += 1
                return None
        store.save_analysis(it.key, it.title, a.to_dict())
        return it, a

    res = await asyncio.gather(*(one(i) for i in items))
    return [r for r in res if r], failed


async def run(cfg: Config, dry_run: bool = False) -> int:
    store = Store(cfg.state_path)
    async with httpx.AsyncClient(headers={"User-Agent": UA}, follow_redirects=True) as client:
        ds = DeepSeekClient(client, cfg.deepseek_api_key, cfg.deepseek_model, cfg.deepseek_base_url)
        await ds.check_model()

        items, notes = await collect(client, cfg.lookback_days)
        if len(notes) == 3:
            raise SystemExit("Все источники недоступны — дайджест не отправлен")
        sent = store.sent_keys([i.key for i in items])
        new = [i for i in items if i.key not in sent]
        new.sort(key=lambda i: i.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        if len(new) > cfg.max_items_llm:
            notes.append(f"новых {len(new)}, обработано только {cfg.max_items_llm} (MAX_ITEMS_LLM)")
            new = new[: cfg.max_items_llm]
        log.info("collected=%d new=%d", len(items), len(new))

        if not new and not cfg.send_empty_digest:
            return 0

        scored, failed = await score_all(ds, store, new, cfg.llm_concurrency)
        chunks = digest.build_digest(scored, cfg.min_relevance, datetime.now(timezone.utc).date(), failed, notes)

        if dry_run:
            print("\n\n=====\n\n".join(chunks))
            return 0
        await telegram.send_messages(client, cfg.telegram_bot_token, cfg.telegram_chat_id, chunks)
        # Failed-LLM items are NOT marked sent: they will be retried next run.
        store.mark_sent([i.key for i, _ in scored])
        log.info("sent %d messages, %d items", len(chunks), len(scored))
    return 0


def cli() -> None:
    p = argparse.ArgumentParser(prog="splat-digest")
    p.add_argument("command", nargs="?", default="run", choices=["run", "dry-run", "chat-id"])
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
    sys.exit(asyncio.run(run(cfg, dry_run=args.command == "dry-run")))


if __name__ == "__main__":
    cli()

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from splat_digest import digest, main
from splat_digest.config import Config
from splat_digest.llm import Analysis
from splat_digest.models import Item, arxiv_id_from, is_on_topic, make_key
from splat_digest.sources import arxiv, hf, hn
from splat_digest.store import Store

FIX = Path(__file__).parent / "fixtures"
NOW = datetime.now(timezone.utc)


def arxiv_xml() -> str:
    recent = (NOW - timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (FIX / "arxiv.xml").read_text().replace("__RECENT__", recent)


HF_JSON = [
    {"paper": {"id": "2611.01234", "title": "AeroGS: Gaussian Splatting for Large-Scale UAV Scenes",
               "summary": "We present AeroGS, a 3DGS method for drone imagery, with a longer HF abstract.",
               "upvotes": 42, "publishedAt": NOW.isoformat(), "authors": [{"name": "A. Author"}]}},
    {"paper": {"id": "2611.09999", "title": "An LLM paper", "summary": "Transformers.", "upvotes": 100}},
]
HN_JSON = {"hits": [
    {"objectID": "111", "title": "AeroGS: splatting from drones", "url": "https://arxiv.org/abs/2611.01234v1",
     "points": 120, "num_comments": 30, "created_at_i": int(NOW.timestamp())},
    {"objectID": "222", "title": "Show HN: My Gaussian splat viewer", "url": "https://example.com/viewer",
     "points": 50, "created_at_i": int(NOW.timestamp())},
    {"objectID": "333", "title": "Gaussian process regression tutorial", "url": "https://example.com/gp",
     "created_at_i": int(NOW.timestamp())},
]}


def test_helpers():
    assert arxiv_id_from("http://arxiv.org/abs/2611.01234v2") == "2611.01234"
    assert arxiv_id_from("https://arxiv.org/pdf/2611.01234") == "2611.01234"
    assert make_key("https://example.com", "hn:1") == "hn:1"
    assert is_on_topic("Fast 3DGS rendering") and is_on_topic("Gaussian Splatting")
    assert not is_on_topic("Gaussian process regression")


def test_arxiv_parse_filters_old_and_normalizes():
    items = arxiv.parse_feed(arxiv_xml(), NOW - timedelta(days=3))
    assert [i.key for i in items] == ["arxiv:2611.01234", "arxiv:2611.05555"]
    assert items[0].title == "AeroGS: Gaussian Splatting for Large-Scale UAV Scenes"
    assert "github" in items[0].extra["arxiv_comment"]


def test_hf_and_hn_parse():
    assert [i.key for i in hf.parse(HF_JSON)] == ["arxiv:2611.01234"]
    keys = [i.key for i in hn.parse(HN_JSON["hits"])]
    assert keys == ["arxiv:2611.01234", "hn:222"]  # GP tutorial filtered out


@pytest.mark.parametrize("raw,ok", [
    ({"relevance": "8", "summary_ru": "x"}, 8),
    ({"relevance": 15, "summary_ru": "x"}, 10),
    ({"relevance": 7.6, "summary_ru": "x", "practical_value": "HUGE", "code_available": "yes"}, 8),
])
def test_analysis_coercion(raw, ok):
    a = Analysis.from_json(raw)
    assert a.relevance == ok and a.practical_value in ("low", "medium", "high")
    assert a.code_available is None or isinstance(a.code_available, bool)


@pytest.mark.parametrize("raw", [{"summary_ru": "x"}, {"relevance": "?", "summary_ru": "x"}, {"relevance": 5}, []])
def test_analysis_rejects(raw):
    with pytest.raises(ValueError):
        Analysis.from_json(raw)


def test_pack_respects_limit():
    blocks = ["a" * 3000, "b" * 3000, "c" * 100, "d" * 9000]
    chunks = digest.pack(blocks, limit=4000)
    assert all(len(c) <= 4000 for c in chunks)
    assert "".join(chunks).count("a") == 3000


def test_render_escapes_html():
    it = Item(key="hn:1", title="<script> & co", url="https://x.com/?a=1&b=2", sources=["hn"])
    a = Analysis(relevance=6, relevance_reason="r", summary_ru="1 < 2")
    out = digest.render_item(it, a)
    assert "&lt;script&gt; &amp; co" in out and "1 &lt; 2" in out and "a=1&amp;b=2" in out


class FakeWorld:
    def __init__(self, llm_fail_keys=(), tg_fail=False):
        self.llm_calls = 0
        self.tg_messages: list[dict] = []
        self.llm_fail_keys = set(llm_fail_keys)
        self.tg_fail = tg_fail

    def handler(self, req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        if "export.arxiv.org" in u:
            return httpx.Response(200, text=arxiv_xml())
        if "huggingface.co/api/daily_papers" in u:
            return httpx.Response(200, json=HF_JSON)
        if "hn.algolia.com" in u:
            return httpx.Response(200, json=HN_JSON)
        if u.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}]})
        if u.endswith("/chat/completions"):
            self.llm_calls += 1
            body = json.loads(req.content)
            assert body["response_format"] == {"type": "json_object"}
            user = body["messages"][1]["content"]
            if any(k in user for k in self.llm_fail_keys):
                return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
            rel = 9 if "UAV" in user else 2
            content = json.dumps({"relevance": rel, "relevance_reason": "причина", "summary_ru": "пересказ",
                                  "drone_relevant": rel > 5, "tags": ["aerial"], "code_available": True})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        if "api.telegram.org" in u:
            if self.tg_fail:
                return httpx.Response(500, json={"ok": False})
            self.tg_messages.append(json.loads(req.content))
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)


@pytest.fixture
def world(monkeypatch, tmp_path):
    w = FakeWorld()
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(lambda r: w.handler(r))
        return real(*a, **kw)

    monkeypatch.setattr(main.httpx, "AsyncClient", factory)

    async def nosleep(*_):
        return None

    monkeypatch.setattr("asyncio.sleep", nosleep)
    cfg = Config(deepseek_api_key="k", telegram_bot_token="t", telegram_chat_id="1",
                 state_path=str(tmp_path / "s.db"), lookback_days=2)
    return w, cfg


async def test_end_to_end_dedup_and_state(world):
    w, cfg = world
    assert await main.run(cfg) == 0
    # 3 unique: AeroGS (arxiv+hf+hn merged), GaussAvatar, HN viewer
    assert w.llm_calls == 3
    text = "\n".join(m["text"] for m in w.tg_messages)
    assert "9/10" in text and "AeroGS" in text and "HF ▲42" in text and "HN ▲120" in text
    assert "Ниже порога (2)" in text
    # Second run: nothing new, no LLM calls, short message only
    w.tg_messages.clear()
    await main.run(cfg)
    assert w.llm_calls == 3
    assert "Новых: 0" in w.tg_messages[0]["text"]


async def test_llm_failure_retried_next_run(world):
    w, cfg = world
    w.llm_fail_keys = {"GaussAvatar"}
    await main.run(cfg)
    assert "не обработано LLM: 1" in w.tg_messages[0]["text"]
    w.llm_fail_keys = set()
    w.tg_messages.clear()
    await main.run(cfg)
    assert "Новых: 1" in w.tg_messages[0]["text"]


async def test_telegram_failure_does_not_mark_sent_but_caches_llm(world):
    w, cfg = world
    w.tg_fail = True
    with pytest.raises(RuntimeError):
        await main.run(cfg)
    calls = w.llm_calls
    w.tg_fail = False
    await main.run(cfg)
    assert w.llm_calls == calls  # analyses reused from cache
    assert "Новых: 3" in w.tg_messages[0]["text"]


async def test_bad_model_fails_fast(world):
    w, cfg = world
    from dataclasses import replace
    with pytest.raises(SystemExit, match="deepseek-chat"):
        await main.run(replace(cfg, deepseek_model="deepseek-chat"))
    assert w.llm_calls == 0


def test_store_roundtrip(tmp_path):
    s = Store(str(tmp_path / "x.db"))
    s.save_analysis("k", "t", {"relevance": 1})
    assert s.get_analysis("k") == {"relevance": 1} and s.sent_keys(["k"]) == set()
    s.mark_sent(["k"])
    assert s.sent_keys(["k", "z"]) == {"k"} and s.get_analysis("k") == {"relevance": 1}

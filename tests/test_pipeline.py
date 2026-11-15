import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from splat_digest import digest, main
from splat_digest.config import Config
from splat_digest.llm import Analysis
from splat_digest.models import Item, arxiv_id_from, is_on_topic, make_key
from splat_digest.sources import arxiv, bluesky, hf, hn, openreview
from splat_digest.sources.base import Ctx, extract_links, parse_feed
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


RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>SuperSplat 2.0 released</title><link>https://blog.playcanvas.com/supersplat-2/</link>
<guid>https://blog.playcanvas.com/supersplat-2/</guid><pubDate>{now}</pubDate>
<description>&lt;p&gt;Gaussian splat editor update&lt;/p&gt;</description></item>
</channel></rss>"""
ATOM_RELEASES = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>tag:github.com,2008:Repository/1/v1.5.0</id><updated>{now}</updated>
<link rel="alternate" type="text/html" href="https://github.com/nerfstudio-project/gsplat/releases/tag/v1.5.0"/>
<title>v1.5.0</title><content type="html">&lt;p&gt;Faster rasterization, 2DGS support&lt;/p&gt;</content></entry>
<entry><id>tag:github.com,2008:Repository/1/v1.0.0</id><updated>2024-01-01T00:00:00Z</updated>
<link rel="alternate" href="https://github.com/x/old"/><title>v1.0.0</title></entry>
</feed>"""
VENDOR_PAGE_1 = '<html><a href="/news/detail/old-drone">Old drone</a><a href="/about">About</a></html>'
VENDOR_PAGE_2 = VENDOR_PAGE_1 + '<a href="/news/detail/terra-3dgs">DJI Terra adds Gaussian Splatting</a>'
ARTICLE = ('<html><head><meta property="og:title" content="DJI Terra adds 3DGS">'
           '<meta name="description" content="Gaussian splatting reconstruction from drone flights"></head></html>')


def test_feed_parsers():
    rss = parse_feed(RSS.format(now="Sat, 14 Nov 2026 10:00:00 GMT"))
    assert rss[0]["title"] == "SuperSplat 2.0 released" and rss[0]["summary"] == "Gaussian splat editor update"
    assert rss[0]["published"].year == 2026
    atom = parse_feed(ATOM_RELEASES.format(now=NOW.isoformat()))
    assert atom[0]["link"].endswith("v1.5.0") and "2DGS" in atom[0]["summary"]


def test_extract_links_absolute_and_dedup():
    links = dict(extract_links(VENDOR_PAGE_2 + '<a href="/news/detail/terra-3dgs#x">dup</a>', "https://enterprise.dji.com/news"))
    assert "https://enterprise.dji.com/news/detail/terra-3dgs" in links
    assert links["https://enterprise.dji.com/news/detail/terra-3dgs"] == "DJI Terra adds Gaussian Splatting"


def test_dedupe_title_and_code_attach():
    a = Item(key="arxiv:2611.01234", title="AeroGS: Gaussian Splatting for UAV", url="https://arxiv.org/abs/2611.01234",
             sources=["arxiv"])
    o = Item(key="openreview:abc", title="AeroGS: Gaussian splatting for UAV!", url="https://openreview.net/forum?id=abc",
             sources=["openreview"], extra={"venue": "ICLR 2027 submission"})
    r = Item(key="gh-repo:x/aerogs", title="x/aerogs", url="https://github.com/x/aerogs", kind="tool",
             sources=["github"], extra={"arxiv_refs": ["2611.01234"], "gh_stars": 77})
    hn_only = Item(key="arxiv:2611.07777", title="Paper only on HN", url="https://arxiv.org/abs/2611.07777",
                   kind="community", sources=["hn"])
    out = main.dedupe([a, o, r, hn_only])
    assert len(out) == 2
    m = next(i for i in out if i.key == "arxiv:2611.01234")
    assert m.sources == ["arxiv", "openreview"] and m.extra["venue"] and m.extra["code_url"].endswith("aerogs")
    assert next(i for i in out if i.key == "arxiv:2611.07777").kind == "paper"


def test_openreview_and_bluesky_parse():
    notes = [{"id": "n1", "forum": "f1", "cdate": int(NOW.timestamp() * 1000),
              "content": {"title": {"value": "Indoor Gaussian Splatting"}, "abstract": {"value": "rooms"},
                          "venue": {"value": "ICLR 2027 Conference Submission"}}},
             {"id": "n2", "cdate": 0, "content": {"title": {"value": "Old 3DGS"}}}]
    items = openreview.parse(notes, NOW - timedelta(days=14))
    assert [i.key for i in items] == ["openreview:f1"] and "ICLR" in items[0].extra["venue"]
    posts = [{"uri": "at://did:x/app.bsky.feed.post/3abc", "author": {"handle": "res.bsky.social"}, "likeCount": 30,
              "record": {"text": "Our new 3DGS paper for drones!", "createdAt": NOW.isoformat()},
              "embed": {"external": {"uri": "https://arxiv.org/abs/2611.01234", "title": "AeroGS"}}},
             {"uri": "at://x/p/1", "likeCount": 1, "record": {"text": "gaussian splatting lol"}}]
    items = bluesky.parse(posts, min_likes=5)
    assert [i.key for i in items] == ["arxiv:2611.01234"]
    assert items[0].extra["bsky_url"] == "https://bsky.app/profile/res.bsky.social/post/3abc"


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
        self.vendor_page = VENDOR_PAGE_1
        self.awesome_ids = ["2601.00001"]
        self.llm_calls = 0
        self.tg_messages: list[dict] = []
        self.llm_fail_keys = set(llm_fail_keys)
        self.tg_fail = tg_fail

    def handler(self, req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        if "export.arxiv.org" in u and "id_list" in u:
            xml = arxiv_xml().replace("2611.05555", "2611.08888").replace("GaussAvatar: Animatable Human Avatars",
                                                                          "Curated Indoor GS")
            return httpx.Response(200, text=xml)
        if "export.arxiv.org" in u:
            return httpx.Response(200, text=arxiv_xml())
        if "releases.atom" in u:
            return httpx.Response(200, text=ATOM_RELEASES.format(now=NOW.isoformat()))
        if "raw.githubusercontent.com" in u:
            return httpx.Response(200, text="\n".join(f"- https://arxiv.org/abs/{i}" for i in self.awesome_ids))
        if "enterprise.dji.com/news/detail/terra-3dgs" in u:
            return httpx.Response(200, text=ARTICLE)
        if "enterprise.dji.com/news" in u:
            return httpx.Response(200, text=self.vendor_page)
        if "blog.playcanvas.com" in u:
            return httpx.Response(200, text=RSS.format(now=NOW.strftime("%a, %d %b %Y %H:%M:%S GMT")))
        if "api.github.com/search" in u:
            return httpx.Response(200, json={"items": [{"full_name": "x/aerogs", "html_url": "https://github.com/x/aerogs",
                "description": "Code for https://arxiv.org/abs/2611.01234 gaussian splatting", "stargazers_count": 77}]})
        if "semanticscholar.org" in u:
            return httpx.Response(200, json={"data": [{"paperId": "s2x", "title": "IndoorSplat: Gaussian Splatting for Rooms",
                "abstract": "indoor 3DGS", "externalIds": {}, "venue": "ICRA", "citationCount": 3,
                "publicationDate": NOW.date().isoformat(), "url": "https://www.semanticscholar.org/paper/s2x"}]})
        if "openreview.net" in u:
            return httpx.Response(200, json={"notes": [{"id": "f1", "forum": "f1", "cdate": int(NOW.timestamp() * 1000),
                "content": {"title": {"value": "IndoorSplat: Gaussian Splatting for Rooms"},
                            "abstract": {"value": "indoor"}}}]})
        if "reddit.com" in u:
            return httpx.Response(403)
        if "bsky" in u:
            return httpx.Response(200, json={"posts": []})
        if "huggingface.co/api/spaces" in u or "huggingface.co/api/models" in u:
            return httpx.Response(200, json=[{"id": "u/gs-demo", "createdAt": NOW.isoformat(), "likes": 3,
                                              "tags": ["gaussian-splatting"]}] if "spaces" in u else [])
        if "huggingface.co/api/daily_papers" in u:
            return httpx.Response(200, json=HF_JSON)
        if "hn.algolia.com" in u:
            return httpx.Response(200, json=HN_JSON)
        if u.endswith("/v1/models"):
            return httpx.Response(200, json={"data": [{"id": "kimi-k2.6"}, {"id": "kimi-k3"}]})
        if u.endswith("/chat/completions"):
            self.llm_calls += 1
            body = json.loads(req.content)
            assert body["response_format"] == {"type": "json_object"}
            assert "temperature" not in body and body["thinking"] == {"type": "disabled"}
            user = body["messages"][1]["content"]
            if any(k in user for k in self.llm_fail_keys):
                return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
            rel = 9 if "UAV" in user else 7 if any(k in user for k in ("DJI Terra", "gsplat", "IndoorSplat", "Curated")) else 2
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
    import tomllib
    conf = tomllib.loads("""
[github_releases]
repos = ["nerfstudio-project/gsplat"]
[awesome]
urls = ["https://raw.githubusercontent.com/a/b/main/README.md"]
[vendors]
sites = [
  { name = "DJI", type = "links", url = "https://enterprise.dji.com/news", link_pattern = '/news/detail/' },
  { name = "PlayCanvas", type = "rss", url = "https://blog.playcanvas.com/rss/", kind = "tool" },
]
""")
    cfg = Config(llm_api_key="k", telegram_bot_token="t", telegram_chat_id="1",
                 state_path=str(tmp_path / "s.db"), lookback_days=2, sources_conf=conf)
    return w, cfg


async def test_end_to_end_dedup_and_state(world):
    w, cfg = world
    assert await main.run(cfg) == 0
    # AeroGS (arxiv+hf+hn+github repo merged), GaussAvatar, HN viewer, IndoorSplat (s2+openreview by title),
    # gsplat release, PlayCanvas post, HF space. DJI page + awesome only store a baseline on first run.
    text = "\n".join(m["text"] for m in w.tg_messages)
    assert w.llm_calls == 7, text
    assert "Reddit" in text  # failed source reported in header
    assert "📄 Статьи" in text and "🛠 Инструменты и код" in text
    assert "HF ▲42" in text and "HN ▲120" in text and "код ★77" in text
    assert "IndoorSplat" in text and "🏛 ICRA" in text
    # Second run: DJI publishes a 3DGS post and the awesome list adds a paper
    w.vendor_page = VENDOR_PAGE_2
    w.awesome_ids = ["2601.00001", "2611.08888"]
    w.tg_messages.clear()
    await main.run(cfg)
    text = "\n".join(m["text"] for m in w.tg_messages)
    assert w.llm_calls == 9, text
    assert "🏭 Индустрия и продукты" in text and "DJI Terra adds 3DGS" in text and "Curated Indoor GS" in text
    # Third run: nothing new
    w.tg_messages.clear()
    await main.run(cfg)
    assert w.llm_calls == 9 and "Новых: 0" in w.tg_messages[0]["text"]


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
    assert "Новых: 7" in w.tg_messages[0]["text"]


async def test_bad_model_fails_fast(world):
    w, cfg = world
    from dataclasses import replace
    with pytest.raises(SystemExit, match="kimi-k2-typo"):
        await main.run(replace(cfg, llm_model="kimi-k2-typo"))
    assert w.llm_calls == 0


def test_store_roundtrip(tmp_path):
    s = Store(str(tmp_path / "x.db"))
    s.save_analysis("k", "t", {"relevance": 1})
    assert s.get_analysis("k") == {"relevance": 1} and s.sent_keys(["k"]) == set()
    s.mark_sent(["k"])
    assert s.sent_keys(["k", "z"]) == {"k"} and s.get_analysis("k") == {"relevance": 1}


async def test_seed_then_run_sends_only_new(world):
    w, cfg = world
    await main.seed(cfg)
    assert w.llm_calls == 0 and not w.tg_messages
    w.vendor_page = VENDOR_PAGE_2
    await main.run(cfg)
    text = w.tg_messages[0]["text"]
    assert "Новых: 1" in text and "DJI Terra" in text


async def test_failed_send_keeps_vendor_snapshot_uncommitted(world):
    w, cfg = world
    await main.run(cfg)            # baseline
    w.vendor_page = VENDOR_PAGE_2
    w.tg_fail = True
    with pytest.raises(RuntimeError):
        await main.run(cfg)
    w.tg_fail = False
    w.tg_messages.clear()
    await main.run(cfg)
    assert "DJI Terra" in "\n".join(m["text"] for m in w.tg_messages)


def test_config_k3_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for k in ("LLM_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("LLM_MODEL", "kimi-k3")
    c = Config.from_env()
    assert c.llm_thinking == "omit" and c.llm_max_tokens == 8000
    monkeypatch.setenv("SOURCES", "arxiv,bogus")
    with pytest.raises(SystemExit):
        Config.from_env()


def test_parse_awesome_yaml():
    from splat_digest.sources.github import parse_awesome
    text = """- id: a2026
  title: Robust Dynamic 3DGS Compression
  abstract: 'long
    text arxiv.org/abs/2000.00001 inside'
  paper: https://arxiv.org/pdf/2609.25633.pdf
  code: null
- id: b2026
  title: Indoor GS
  paper: https://arxiv.org/abs/2610.11111
  code: https://github.com/x/indoor
"""
    e = parse_awesome(text)
    assert e["2609.25633"]["title"].startswith("Robust") and "code" not in e["2609.25633"]
    assert e["2610.11111"]["code"] == "https://github.com/x/indoor"

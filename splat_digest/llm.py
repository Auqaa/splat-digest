"""DeepSeek (OpenAI-compatible Chat Completions) scoring.

Docs: https://api-docs.deepseek.com/ . As of the V4 release the legacy names
`deepseek-chat` / `deepseek-reasoner` were retired; the default here is
`deepseek-flash` and is validated against GET /models at startup.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

import httpx

from .models import Item

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — эксперт по 3D Gaussian Splatting (3DGS), нейронному рендерингу и фотограмметрии.
Ты оцениваешь новые статьи/посты для инженера, которого интересует тема:
«3DGS для дронов и реконструкции интерьеров».

Что считается релевантным:
- Дроны/БПЛА: аэросъёмка, крупномасштабные/городские сцены, разреженные или наклонные ракурсы,
  неточные позы/GPS, motion blur, rolling shutter, онлайн/бортовая реконструкция, SLAM на 3DGS,
  навигация и планирование траектории съёмки (active view selection), сжатие для передачи.
- Интерьеры: комнаты/здания, бестекстурные стены, отражающие и прозрачные поверхности, освещение,
  извлечение меша/геометрии/нормалей, планировки, семантика и сегментация, съёмка с телефона/360°,
  few-shot интерьеры, LiDAR+камера, SLAM в помещениях.
- Общие улучшения 3DGS, которые напрямую применимы к этим сценариям (скорость обучения, качество геометрии,
  работа без COLMAP-поз, anti-aliasing, большие сцены, LOD) — средняя релевантность.

Шкала relevance (целое 0–10):
- 9–10: статья прямо про 3DGS для дронов/аэросъёмки ИЛИ для интерьеров.
- 7–8: 3DGS-метод, явно протестированный на аэро- или indoor-датасетах (Mill-19, UrbanScene3D, ScanNet,
  ScanNet++, Replica, Matterport3D и т.п.), либо SLAM/геометрия, сразу применимые.
- 4–6: общее улучшение 3DGS, полезное косвенно.
- 1–3: 3DGS, но другая область (аватары, люди, генерация объектов, медицина, автономные авто без переноса).
- 0: не про 3DGS / не статья / шум.
Не завышай оценки. Большинство статей должно получать 1–6.

Отвечай ТОЛЬКО валидным JSON-объектом такого вида:
{
  "relevance": 0,
  "relevance_reason": "почему такая оценка, 1 предложение по-русски",
  "summary_ru": "краткий пересказ на русском, 2–3 предложения: задача, метод, результат",
  "key_contribution": "главный вклад одной фразой по-русски",
  "drone_relevant": false,
  "indoor_relevant": false,
  "practical_value": "low | medium | high",
  "code_available": null,
  "tags": ["slam", "aerial", "indoor", "mesh", "compression", "..."],
  "is_paper": true
}
code_available: true, если в тексте явно указан код/репозиторий; false, если явно сказано, что кода нет; иначе null.
is_paper: false, если это не научная работа (например, пост-продукт, демо, новость).
Не выдумывай факты, которых нет в тексте. Если информации мало — так и напиши в summary_ru."""


@dataclass
class Analysis:
    relevance: int
    relevance_reason: str
    summary_ru: str
    key_contribution: str = ""
    drone_relevant: bool = False
    indoor_relevant: bool = False
    practical_value: str = "low"
    code_available: bool | None = None
    tags: list[str] = field(default_factory=list)
    is_paper: bool = True

    @classmethod
    def from_json(cls, data: dict) -> "Analysis":
        """Validate and coerce model output. Raises ValueError on unusable output."""
        if not isinstance(data, dict):
            raise ValueError("not an object")
        try:
            rel = int(round(float(data["relevance"])))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"bad relevance: {data.get('relevance')!r}") from e
        summary = str(data.get("summary_ru") or "").strip()
        if not summary:
            raise ValueError("empty summary_ru")
        pv = str(data.get("practical_value") or "low").lower().strip()
        code = data.get("code_available")
        tags = data.get("tags") or []
        return cls(
            relevance=max(0, min(10, rel)),
            relevance_reason=str(data.get("relevance_reason") or "").strip(),
            summary_ru=summary,
            key_contribution=str(data.get("key_contribution") or "").strip(),
            drone_relevant=bool(data.get("drone_relevant")),
            indoor_relevant=bool(data.get("indoor_relevant")),
            practical_value=pv if pv in ("low", "medium", "high") else "low",
            code_available=code if isinstance(code, bool) else None,
            tags=[str(t) for t in tags if t][:6] if isinstance(tags, list) else [],
            is_paper=bool(data.get("is_paper", True)),
        )

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def build_user_prompt(item: Item) -> str:
    parts = [f"Заголовок: {item.title}", f"Источники: {', '.join(item.sources)}", f"URL: {item.url}"]
    if item.extra.get("arxiv_category"):
        parts.append(f"Категория arXiv: {item.extra['arxiv_category']}")
    if item.extra.get("arxiv_comment"):
        parts.append(f"Комментарий авторов: {item.extra['arxiv_comment']}")
    parts.append(f"Аннотация/текст:\n{item.abstract or '(нет текста, только заголовок)'}")
    parts.append("Верни JSON.")
    return "\n".join(parts)


class DeepSeekClient:
    def __init__(self, client: httpx.AsyncClient, api_key: str, model: str, base_url: str):
        self.client = client
        self.model = model
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def check_model(self) -> None:
        """Fail fast with a clear message if the configured model ID does not exist."""
        r = await self.client.get(f"{self.base_url}/models", headers=self.headers, timeout=30)
        if r.status_code == 401:
            raise SystemExit("DeepSeek: неверный DEEPSEEK_API_KEY (401)")
        r.raise_for_status()
        ids = [m.get("id") for m in r.json().get("data", [])]
        if self.model not in ids:
            raise SystemExit(
                f"DeepSeek: модели {self.model!r} нет в /models. Доступные: {ids}. Задайте DEEPSEEK_MODEL."
            )
        log.info("DeepSeek models: %s (using %s)", ids, self.model)

    async def analyze(self, item: Item, retries: int = 3) -> Analysis:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(item)},
            ],
            "response_format": {"type": "json_object"},
            # Thinking adds latency/cost and isn't needed for scoring an abstract.
            "thinking": {"type": "disabled"},
            "temperature": 0.2,
            "max_tokens": 1200,
        }
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                r = await self.client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=self.headers, timeout=120
                )
                if r.status_code in (429, 500, 502, 503):
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                if r.status_code == 400 and "thinking" in r.text and "thinking" in payload:
                    # Older/other models may reject the param; drop it and retry.
                    payload.pop("thinking")
                    raise ValueError(f"400, retrying without thinking param: {r.text[:200]}")
                r.raise_for_status()
                content = r.json()["choices"][0]["message"].get("content") or ""
                if not content.strip():
                    raise ValueError("empty content")  # known DeepSeek JSON-mode quirk
                return Analysis.from_json(json.loads(content))
            except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                log.warning("LLM attempt %d/%d failed for %s: %s", attempt, retries, item.key, e)
                await asyncio.sleep(min(2**attempt, 20))
        raise RuntimeError(f"LLM failed for {item.key}: {last_err}")

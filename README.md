# splat-digest

Раз в день собирает новое про 3D Gaussian Splatting: статьи, релизы инструментов, новости компаний и
обсуждения. Kimi оценивает каждую запись по теме **«3DGS для дронов и реконструкции интерьеров»**,
и в Telegram приходит дайджест, разбитый на разделы: 📄 Статьи · 🛠 Инструменты и код · 🏭 Индустрия · 💬 Сообщество.

## Источники
| `SOURCES` | Что | Ключ | Замечания |
|---|---|---|---|
| `arxiv` | arXiv API, `all:"gaussian splatting" OR 3DGS` | — | основной поток статей |
| `hf_daily` | HF Daily Papers | — | в основном дубли arXiv, ценен апвоутами |
| `s2` | Semantic Scholar bulk search, 21 день | `S2_API_KEY` (желательно) | конференционные статьи вне arXiv, площадка и цитирования |
| `openreview` | OpenReview, 14 дней | — | ICLR/NeurIPS-сабмишены; сезонно |
| `awesome` | новые записи в [awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) (YAML) | — | ручной отбор и ссылка на код |
| `github_releases` | Atom-ленты релизов (gsplat, COLMAP, OpenSplat, ODM, Brush, SuperSplat, LichtFeld, Spark…) | — | список в `sources.toml` |
| `github_repos` | новые репозитории (≤14 дней, ≥15★) | `GITHUB_TOKEN` | если репозиторий ссылается на статью из дайджеста, он становится ссылкой «код ★N» у статьи |
| `hf_hub` | новые модели и Spaces на HF | — | шумно, ограничено 20 записями |
| `hn` | Hacker News (Algolia) | — | редко |
| `reddit` | RSS r/GaussianSplatting, r/photogrammetry, r/UAVmapping | — | **с IP GitHub Actions часто 403** |
| `bluesky` | searchPosts, sort=top, ≥5 лайков | `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD` | без авторизации поиск может быть закрыт |
| `vendors` | RadianceFields, PlayCanvas, DJI, Pix4D, Skydio, Agisoft, XGRIDS, Matterport, Polycam, Niantic | — | **URL не проверены**, см. ниже |

Дубли склеиваются по arXiv ID, а затем по нормализованному заголовку (так OpenReview и S2 сливаются с arXiv).
Если один источник упал, дайджест всё равно уходит, а в шапке появляется ⚠️ с причиной.

### Сайты компаний
Для них два режима: `rss` и `links`. В режиме `links` бот следит за новыми ссылками на странице-списке
по regex `link_pattern`, а для каждой новой ссылки забирает `og:title` и `og:description`. При первом
запуске он только запоминает текущие ссылки. Сайты, которые рендерятся через JS, вернут 0 ссылок, и
`check-sources` покажет ПУСТО. Для них нужен RSS или нужно их убрать.

## LLM: Kimi (Moonshot)
По умолчанию используется `kimi-k2.6` через `https://api.moonshot.ai/v1` с выключенным thinking и
`response_format: json_object`. Параметр `temperature` не передаётся: у Kimi он фиксирован, и другие
значения API отклоняет. При запуске бот проверяет модель через `GET /models`.

Ориентир по цене (прайс kimi.com, октябрь 2026; проверьте актуальный): K2.6 стоит $0.95 за 1M входных
токенов и $4 за 1M выходных. Это примерно $0.003 на запись, то есть **около $5–15 в месяц** при 50–150
записях в день. `kimi-k3` стоит $3/$15 и всегда рассуждает, поэтому выходит **в 10–20 раз дороже**; для
оценки аннотаций это не окупается. Любой другой OpenAI-совместимый API подключается через
`LLM_BASE_URL` и `LLM_MODEL`.

JSON, который возвращает модель: `relevance` (0–10), `relevance_reason`, `summary_ru`, `key_contribution`,
`drone_relevant`, `indoor_relevant`, `practical_value`, `code_available`, `tags`, `is_paper`.
Критерии оценки для каждого типа записи заданы в `SYSTEM_PROMPT` (`splat_digest/llm.py`).

## Первый запуск
```bash
pip install -e .
cp .env.example .env          # заполнить
splat-digest chat-id          # сначала написать боту /start
splat-digest check-sources    # что работает, что нет; поправить sources.toml
splat-digest seed             # пометить всё текущее как прочитанное (иначе первый дайджест будет огромным)
splat-digest dry-run          # дайджест в консоль, без отправки
splat-digest run
```

## GitHub Actions
Workflow `.github/workflows/digest.yml` запускается ежедневно в 05:17 UTC и вручную через `workflow_dispatch`.
- Обязательные secrets: `LLM_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- Необязательные secrets: `S2_API_KEY`, `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`.
- Variables: `LLM_MODEL`, `LLM_BASE_URL`, `MIN_RELEVANCE`, `SOURCES`.
- Состояние (SQLite) хранится в Actions cache. Reddit оттуда, скорее всего, будет отдавать 403; тогда
  уберите его из `SOURCES` или запускайте бота на своём сервере.

## Надёжность
- Запись помечается отправленной только после того, как Telegram принял сообщение. Снимки страниц
  компаний и awesome-списка тоже сохраняются только после успешной отправки.
- Ответы LLM кешируются: повторный запуск после сбоя не платит второй раз.
- Если LLM не справился с записью, она попадёт в следующий запуск.

## Тесты
```bash
pip install -e '.[dev]' && pytest
```

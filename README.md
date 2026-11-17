# splat-digest

Раз в день собирает новое про 3D Gaussian Splatting: статьи, релизы инструментов, новости компаний и
обсуждения. DeepSeek оценивает каждую запись по теме **«3DGS для дронов и реконструкции интерьеров»**,
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

## LLM: DeepSeek (по умолчанию) или Kimi
По умолчанию используется `deepseek-flash` (DeepSeek-V4.1-Flash) через `https://api.deepseek.com`.
Thinking выключен, ответ запрашивается в JSON-режиме. При запуске бот проверяет модель через `GET /models`.
Ошибку генерации, которую DeepSeek возвращает с кодом 200 и телом `{"error": ...}`, бот распознаёт и
повторяет запрос.

**Полный текст статьи (`DEEP_READ`).** Аннотации пишутся продающим языком, поэтому оценка по ним завышена.
Для 5 самых релевантных arXiv-статей (оценка ≥ 7) модель вторым проходом читает полный текст через
серверный инструмент `deepseek_read_url` (HTML-версию arXiv или PDF) и заново ставит оценку. В карточке
тогда видно, например, «6/10 (по аннотации 9)» и блок «🔎 По полному тексту»: метод, данные, цифры,
ресурсы, ограничения, применимость к дронам и интерьерам. Если инструмент не подтвердил, что страница
прочитана, результат отбрасывается: модель не должна пересказывать статью, которую не читала.
Ограничение: длинные страницы инструмент обрезает, и раздел с результатами может не попасть в контекст.

**Цена** (прайс DeepSeek; Flash вне пика — $0.15 за 1M входных токенов и $0.6 за 1M выходных, в пик вдвое дороже):
- оценка: примерно $0.0005 за запись, $1–3 в месяц при 50–150 записях в день;
- полный текст: примерно $0.005 за статью (около 30 тыс. токенов страницы плюс $0.00075 за загрузку), около $1 в месяц;
- cron стоит на 04:23 UTC, внутри окна off-peak (пик — 01–04 и 06–10 UTC по будням).

**Kimi.** Задайте `LLM_PROVIDER=kimi` и ключ Moonshot. Тогда по умолчанию будет `kimi-k2.6`, а
`DEEP_READ` отключится, потому что это инструмент только DeepSeek. Kimi выходит примерно в 5–10 раз дороже.

JSON, который возвращает модель: `relevance` (0–10), `relevance_reason`, `summary_ru`, `key_contribution`,
`drone_relevant`, `indoor_relevant`, `practical_value`, `code_available`, `tags`, `is_paper`.
Критерии оценки для каждого типа записи заданы в `SYSTEM_PROMPT` (`splat_digest/llm.py`).

## Первый запуск

Пошаговая инструкция без установки на компьютер (Telegram + GitHub): **[SETUP.md](SETUP.md)**.
Ниже — вариант для запуска локально.
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
Workflow `.github/workflows/digest.yml` запускается ежедневно в 04:23 UTC и вручную через `workflow_dispatch`.
- Обязательные secrets: `LLM_API_KEY` (ключ DeepSeek), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- Необязательные secrets: `S2_API_KEY`, `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`.
- Variables: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `DEEP_READ`, `MIN_RELEVANCE`, `SOURCES`.
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

# Agent In Rust Scratch

Rust scaffold for building a coding assistant agent inspired by Claude Code and OpenClaw.

## Current capabilities

- Thin chat loop with harness-side controls
- Append-only JSONL session memory
- Permission gating for privileged tools
- Tool registry with pluggable tools
- SQLite event bus for inbound/outbound dispatch
- Routing + agent-to-agent dispatch (`/route`, `/dispatch`)
- Delivery workers for `cli`, `file`, and websocket server transport (`/ws send`, `/ws broadcast`, `/ws poll`, `/ws clients`)
- Telegram Bot transport via polling + send API (`/tg send`, `/tg poll`, `/tg status`)
- Optional websocket auth handshake (`AUTH <token>`) via config
- Persistent websocket session ownership + ACL controls (`/ws bind`, `/ws allow`, `/ws acl`)
- Persistent Telegram chat ownership + ACL controls (`/tg bind`, `/tg allow`, `/tg acl`)
- Websocket hardening: idle timeout + ping/pong heartbeat + connection quota + token rotation (`/ws status`, `/ws rotate-token`)
- Provider adapters (`MODEL_PROVIDER=openai|anthropic|local`)
- OpenAI Responses API streaming with optional reasoning summaries + reasoning token usage metrics
- Planning/evaluator contracts (`/plan <goal>`, `/plan status|resume|start|done|fail ...`)
- Runtime config reload from `.agent/config.json` (`/reload-config`)
- Skill loading from `.agent/skills/*.json` (`/skills`)
- Deterministic policy engine + audit log (`/policy reload`)
- Policy rule matchers (`exact|prefix|wildcard|regex`) and scope (`source/channel/agent`)
- Wall-clock heartbeat scheduler daemon + manual `/tick`
- Multi-layer prompt composition (identity/channel/skills/memory)
- Long-term memory snapshots in SQLite with semantic/hybrid retrieval + compaction (`/memory latest`, `/memory search ...`, `/memory compact`)
- ANN-style memory candidate indexing for larger stores (threshold-based fast path with fallback scan)
- Retrieval diagnostics for ANN path distribution, candidate volume, latency, and maintenance lag (`/memory status`)
- Memory health snapshot command and optional periodic snapshot logging (`/memory snapshot`, `memory_status_snapshot_interval_ticks`)

## Run

Create `.env` from the example and set provider credentials:

```bash
cp .env.example .env
cargo run
```

## CLI usage

- `/help`
- `/tools`
- `/compact`
- `/route <source> <agent_id>`
- `/dispatch <to_agent> <content>`
- `/ws send <content>`
- `/ws broadcast <content>`
- `/ws poll`
- `/ws clients`
- `/ws bind <session_id> <agent_id>`
- `/ws allow <session_id> <agent_id>`
- `/ws acl <session_id>`
- `/ws status`
- `/ws rotate-token <new_token>`
- `/tg send <chat_id> <content>`
- `/tg poll`
- `/tg bind <chat_id> <agent_id>`
- `/tg allow <chat_id> <agent_id>`
- `/tg acl <chat_id>`
- `/tg status`
- `/reload-config`
- `/skills [list|reload]`
- `/policy reload`
- `/plan <goal>`
- `/plan status <plan_id>`
- `/plan resume <plan_id>`
- `/plan start <plan_id> <step_id>`
- `/plan done <plan_id> <step_id>`
- `/plan fail <plan_id> <step_id> <reason>`
- `/tick`
- `/post <cli|file|websocket|telegram> <content>`
- `/memory latest`
- `/memory search <query>`
- `/memory compact`
- `/memory maintain`
- `/memory status`
- `/memory snapshot`
- `/approve on`
- `/approve off`
- `/thinking on`
- `/thinking off`
- `/exit`
- `tool:<name> <args>` to execute tools directly

## Telegram setup

Telegram runtime is configured in `.agent/config.json`:

```json
{
  "telegram_enabled": true,
  "telegram_bot_token": "123456:your_bot_token",
  "telegram_api_base_url": "https://api.telegram.org",
  "telegram_poll_interval_secs": 2,
  "telegram_poll_timeout_secs": 0,
  "telegram_allowed_chat_ids": []
}
```

`telegram_allowed_chat_ids` empty means all chats are allowed. Set explicit chat ids to restrict access.

## OpenAI reasoning summary mode

Set `OPENAI_REASONING_SUMMARY` in `.env` to enable reasoning summaries from OpenAI Responses API:

- `off` (default behavior)
- `auto`
- `concise`
- `detailed`

When enabled, the CLI streams transient `thinking>` reasoning summary updates during inference and prints usage metrics including `reasoning_tokens`.

## Config files

- `.agent/config.json`: runtime settings (bounds, scheduler, websocket auth/hardening, telegram runtime, memory retention + retrieval/embedding/ANN tuning, snapshot interval, skill/policy/artifact paths)
- `.agent/policy.json`: deterministic allow/deny rules for actions (e.g. `tool:read_file`)
- `.agent/skills/*.json`: skill manifests

## Architecture notes

See [`docs/architecture.md`](docs/architecture.md) for layer-by-layer mapping.

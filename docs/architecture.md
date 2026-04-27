# Architecture Mapping

This project starts with a "thin model loop + heavy harness" baseline, aligning with the wiki's coding-agent design-space.

## Implemented now

1. Core loop (`src/agent/harness.rs`)
- Interactive loop
- Slash command handling
- Tool call dispatch (`tool:<name> <args>`)
- Model-driven tool-call rounds

2. Permission boundary (`src/agent/permissions.rs`)
- Privileged vs non-privileged tool split
- Per-tool approval
- Optional approval bypass

3. Tooling (`src/agent/tools.rs`)
- Tool trait and registry
- Built-ins: `echo`, `time`, `read_file`
- Workspace-bound file reads (path escape protection)

4. Persistence (`src/agent/session.rs`)
- Append-only JSONL session log
- Boot-time replay of prior session

5. Context management (`src/agent/harness.rs`)
- Manual compaction via `/compact`
- Auto-compaction by message count

6. Model abstraction + provider adapters (`src/agent/model.rs`)
- Unified `ModelBackend`
- Provider selection via `MODEL_PROVIDER`
- Implemented adapters: `openai`, `anthropic`, `local`
- Structured response protocol support (`text` / `tool_calls` JSON envelope)
- OpenAI uses Responses API streaming with optional reasoning summaries and usage metrics (`reasoning_tokens`)

7. Event bus + routing + dispatch (`src/agent/{events,harness,router}.rs`)
- SQLite-backed inbound/outbound queues
- Source-based routing (`EventSource -> agent_id`)
- Agent handler registry and dispatch-task handoff
- Per-event failure isolation (failed handler does not stop queue processing)

8. Channels + delivery workers + websocket + telegram transport (`src/agent/{harness,websocket,telegram}.rs`)
- Channel targets: `cli`, `file`, `websocket`, `telegram`
- Delivery worker behavior with per-channel handlers
- Delivery failure tracking with retry-attempt metadata in SQLite
- Real websocket server transport (bind address configurable) with per-connection session ids
- Authentication handshake support (`AUTH <token>`) when enabled by config
- Session-aware outbound routing for websocket replies (no implicit network broadcast from agent replies)
- Persistent websocket session owner routing in SQLite (`websocket_sessions`)
- Explicit websocket ACLs in SQLite (`websocket_session_acl`)
- Explicit websocket broadcast command: `/ws broadcast <content>`
- CLI websocket commands: `/ws send`, `/ws broadcast`, `/ws poll`, `/ws clients`, `/ws bind`, `/ws allow`, `/ws acl`
- Telegram bot polling transport (`getUpdates`) + outbound send API (`sendMessage`) with runtime config
- Persistent telegram chat owner routing in SQLite (`telegram_chats`)
- Explicit telegram ACLs in SQLite (`telegram_chat_acl`)
- CLI telegram commands: `/tg send`, `/tg poll`, `/tg bind`, `/tg allow`, `/tg acl`, `/tg status`

9. Concurrency bounds (`src/agent/harness.rs`)
- Bounded inbound processing per cycle (`max_inbound_events_per_cycle`)
- Bounded outbound delivery per cycle (`max_outbound_events_per_cycle`)

10. Planning + evaluator artifacts (`src/agent/planner.rs`)
- File-backed plan artifact contracts (`.agent/artifacts/*.json`)
- Evaluation artifact generated alongside plans
- Step execution lifecycle: `pending`, `in_progress`, `completed`, `failed`
- Dependency graph validation + cycle checks
- Resume API for next unblocked step
- CLI command family: `/plan <goal>`, `/plan status|resume|start|done|fail ...`

11. Config hot reload (`src/agent/config.rs`)
- Config source: `.agent/config.json`
- Runtime reload when file changes
- Manual reload command: `/reload-config`

12. Skill loading (`src/agent/skills.rs`)
- Skill manifest loading from `.agent/skills/*.json`
- Validation for empty/duplicate names
- Skill metadata injection into prompt layers
- CLI command: `/skills [list|reload]`

13. Cron + heartbeat scheduling (`src/agent/scheduler.rs`)
- Wall-clock background scheduler daemon writing heartbeats into the event bus
- Backpressure-aware tick suppression with max pending inbound bound
- Manual `/tick` still available for forced enqueue

14. Deterministic policy engine (`src/agent/policy.rs`)
- Explicit allow/deny rules from `.agent/policy.json`
- Matchers: `exact`, `prefix`, `wildcard`, `regex`
- Scope-aware rules: `source`, `channel`, `agent`
- Policy audit trail in `.agent/policy_audit.jsonl`
- CLI command: `/policy reload`

15. Multi-layer prompt assembly + outbound posting (`src/agent/{prompt,harness}.rs`)
- Prompt built by layers: identity/channel/skills/memory
- Agent-initiated outbound command: `/post <channel> <content>`

16. Long-term memory worker (`src/agent/memory.rs`)
- Typed snapshot summaries into SQLite (`.agent/memory.db`)
- External embedding-provider integration (OpenAI embeddings API with local hashed fallback)
- Persisted per-memory embedding vectors in SQLite (`embedding_json`) for retrieval-time reuse
- Embedding-style semantic retrieval using cosine similarity over persisted vectors
- Optional search modes: `lexical`, `semantic`, `hybrid` with configurable hybrid weights
- ANN-style candidate pruning index (`memory_ann_index`) with threshold-based fast path and scan fallback
- Incremental ANN maintenance worker (bounded backfill/reindex with cursor checkpoint in `memory_meta`)
- Manual maintenance command: `/memory maintain`
- Retention compaction jobs (TTL + max-record cap), auto-run on snapshot
- CLI memory commands: `/memory latest`, `/memory search <q>`, `/memory compact`
- Append-only derived memory writes from harness events (no free-form mutation)

17. Websocket production hardening (`src/agent/{websocket,harness,config}.rs`)
- Runtime connection quota controls (`websocket_max_clients`)
- Idle timeout + heartbeat ping/pong controls (`websocket_idle_timeout_secs`, `websocket_ping_interval_secs`)
- Auth token rotation strategy with previous-token grace window (`websocket_auth_previous_token`, `websocket_auth_rotation_grace_secs`, `/ws rotate-token`)
- Runtime policy introspection via CLI (`/ws status`)
- Config reload applies runtime websocket policy updates without process restart

## Next layer (recommended)

1. Add observability + diagnostics for memory retrieval
- Add ANN hit-rate/scan-fallback counters and expose via CLI/status
- Add maintenance lag metrics (`remaining_unindexed`, cursor progress, last-run timestamp)

Status: implemented via durable counters in `memory_meta` and `/memory status`.

2. Next layer (recommended)
- Add retrieval quality diagnostics: expose top-k candidate count and query latency percentiles for semantic/hybrid search paths.
- Add periodic operator summary output (or status snapshot command) for memory subsystem health over long-running sessions.

Status: implemented via `/memory status` (candidate totals/avg/p95, latency avg/p95, path counters) and `/memory snapshot` with optional periodic emission (`memory_status_snapshot_interval_ticks`).

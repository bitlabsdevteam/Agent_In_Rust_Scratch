# Agent In Rust Scratch

Rust scaffold for building a coding assistant agent inspired by Claude Code and OpenClaw.

This init follows your wiki guidance from:

- `/Users/davidbong/Documents/my_second_brain_vault/index.md`
- `/Users/davidbong/Documents/my_second_brain_vault/README.md`
- `wiki/entities/claude-code.md`
- `wiki/entities/openclaw.md`
- `wiki/concepts/coding-agent-design-space.md`
- `wiki/analyses/building-openclaw-style-agents.md`

## Current MVP

- Thin chat loop with harness-side controls
- Append-only JSONL session memory
- Permission gating for privileged tools
- Tool registry with pluggable tools
- Model-driven tool calling (OpenAI tool calls -> local tools)
- Explicit compaction command for context management
- OpenAI chat backend via `.env` (`OPENAI_API_KEY`)

## Run

Create `.env` from the example and set your key:

```bash
cp .env.example .env
```

```bash
cargo run
```

## CLI usage

- `/help`
- `/tools`
- `/compact`
- `/approve on`
- `/approve off`
- `/exit`
- `tool:<name> <args>` to execute tools directly

The model can also call tools automatically when needed.

Example:

```text
tool:time
tool:echo hello
tool:read_file Cargo.toml
```

## Architecture notes

See [`docs/architecture.md`](docs/architecture.md) for how this maps to your wiki's design space.
# Agent_In_Rust_2
# Agent_In_Rust_Scratch
# Agent_In_Rust_Scratch

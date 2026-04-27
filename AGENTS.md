# Repository Guidelines

## Purpose
This repository is an **AI agentic lab** for learning how to build an AI agent from scratch in Rust. Treat each change as a learning step, not just a feature drop. The reference knowledge base for decisions is:
- `/Users/davidbong/Documents/my_second_brain_vault/index.md`

Use the wiki pages on OpenClaw, Claude Code, and related concepts (permissions, context, memory, skills) before major architecture changes.

## Project Structure & Learning Modules
- `src/main.rs`: bootstraps the harness and runtime wiring.
- `src/agent/harness.rs`: interactive loop, slash commands, tool execution.
- `src/agent/tools.rs`: tool interface and tool registry.
- `src/agent/permissions.rs`: approval model for privileged actions.
- `src/agent/session.rs`: append-only JSONL session persistence.
- `src/agent/{model,types,mod}.rs`: model backend and shared types.

Add new capabilities as isolated modules under `src/agent/` and re-export via `src/agent/mod.rs`.

## Build, Test, and Dev Commands
- `cargo run`: run the local harness.
- `cargo build`: compile and catch build errors.
- `cargo test`: run all tests.
- `cargo fmt --all`: format code.
- `cargo clippy --all-targets --all-features -D warnings`: strict lint gate.

## Standards (OpenClaw + Claude Code + Codex)
- **OpenClaw-style progression**: build in layers (loop -> tools -> persistence -> compaction -> routing/delegation).
- **Claude Code-style safety**: enforce deny-first permission checks in the harness, not prompt text alone.
- **Codex-style execution**: small, verifiable increments; clear module boundaries; explicit command-based validation.
- Keep state durable and inspectable (JSONL append-only patterns).
- Design for context pressure early (compact history, keep prompts/tool outputs bounded).

## Testing Guidelines
Write tests for every behavioral change, especially:
- tool-call parsing,
- privileged tool approval flows,
- path safety and workspace boundary checks,
- session load/append behavior.

Name tests by behavior, for example: `read_file_rejects_path_escape`.

## Commit & PR Guidelines
Use Conventional Commit style (`feat:`, `fix:`, `test:`, `refactor:`). Keep PRs small and include:
- what changed,
- why it matches wiki guidance,
- validation output (`cargo test`, `cargo clippy`, `cargo fmt --check`).

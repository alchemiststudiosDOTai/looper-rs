# AGENTS.md

## Project Overview
- `looper-rs` is a Rust library for a lightweight, headless agent loop with both non-streaming and streaming APIs.
- Supported provider families in current code: OpenAI Completions, OpenAI Responses, Anthropic, and Gemini.
- The crate ships library code under `src/`, architecture notes under `docs/`, and runnable integration examples under `examples/`.

## Where To Start
- Read `README.md` first for positioning, setup, and quick usage snippets.
- Read `docs/core-agent-loop-architecture.md` next for the real runtime flow and invariants.
- Use the focused docs when changing a layer:
  - `docs/services-layer.md`
  - `docs/mapping-layer.md`
  - `docs/tools-layer.md`
- Check recent history with `git log --oneline -n 12` before carrying forward old paths or assumptions.

## Repository Map
- `Cargo.toml` — crate manifest; no workspace, task runner, or alternate build system is present.
- `src/lib.rs` — public module exports.
- `src/looper.rs` — non-streaming entry point returning `TurnResult`.
- `src/looper_stream.rs` — streaming entry point forwarding `LooperToInterfaceMessage` events.
- `src/services/` — provider-facing handler traits plus concrete handlers in `src/services/handlers/`.
- `src/mapping/tools/` — active provider tool-schema mappings.
- `src/mapping/turn/` — partial turn-normalization layer; docs note it is not the main runtime path today.
- `src/tools/` — `LooperTool`, `LooperTools`, `EmptyToolSet`, and `SubAgentTool`.
- `src/types/` — shared handler, message history, tool, and turn types.
- `prompts/system_prompt.txt` — Tera template used by both builders.
- `examples/cli.rs` — streaming example with buffered output, multiple file-system tools, and a sub-agent.
- `examples/cli_non_streaming.rs` — non-streaming example using OpenAI Responses.
- `docs/` — deeper architecture docs; prefer these over restating internals here.
- `assets/Demo.mp4` — demo asset linked from the README.

## Commands
- Setup:
  - `cp .env.example .env`
- Build:
  - `cargo build`
- Test:
  - `cargo test`
    - Verified in this repo on 2026-03-20; there are currently 0 unit tests and 0 doc tests, so this is mainly a compile/regression gate.
- Run examples:
  - `cargo run --example cli`
  - `cargo run --example cli_non_streaming`

## Provider / Environment Notes
- `.env.example` currently contains `OPENAI_API_KEY` only.
- Gemini handlers explicitly require `GEMINI_API_KEY` or `GOOGLE_API_KEY` in code (`src/services/handlers/gemini.rs` and `src/services/handlers/gemini_non_streaming.rs`).
- `README.md` mentions `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`; when provider setup looks unclear, prefer the handler code over prose.

## Architecture Boundaries
- Entry points (`src/looper.rs`, `src/looper_stream.rs`) choose a `Handlers` variant, render `prompts/system_prompt.txt`, and inject `SubAgentTool` when configured.
- `src/services/handlers/` owns the real agent loop: restore history, call providers, parse text/thinking/tool calls, execute tools concurrently, and recurse until done.
- `src/mapping/tools/` is the outbound translation boundary from `LooperToolDefinition` to provider SDK tool types.
- `src/tools/` is the capability boundary; application code owns the concrete tool registry implementation.
- `src/types/handlers.rs` defines the critical history split:
  - `MessageHistory::Messages` for OpenAI Completions, Anthropic, and Gemini
  - `MessageHistory::ResponseId` for OpenAI Responses

## Change Guardrails
- Keep streaming and non-streaming behavior aligned when changing a provider: most providers have paired files in `src/services/handlers/`.
- If tool-definition shape changes, inspect both `src/tools/` and `src/mapping/tools/`.
- Do not mix `MessageHistory` variants across handler families.
- Sub-agent support depends on a mutable `LooperTools` registry; builders only inject `SubAgentTool` when `.tools(...)` is supplied.
- `LooperStream` integrations should provide an interface receiver/sender path; the architecture doc calls out undrained streaming channels as a failure mode.
- Prefer updating the architecture docs in `docs/` when behavior changes materially; keep this file as a compact map.

## Sources Of Truth
- `README.md` — public overview, setup, and examples.
- `docs/core-agent-loop-architecture.md` — best single doc for control flow and invariants.
- `docs/services-layer.md` — provider execution model.
- `docs/mapping-layer.md` — translation layer status and boundaries.
- `docs/tools-layer.md` — tool contracts and sub-agent behavior.
- `Cargo.toml` — package shape and example target declarations.
- `src/` — current implementation truth when docs lag.

## Validation Checklist
- Confirm every path above still exists.
- Re-run `cargo test` after code changes.
- If you touched examples or env handling, re-check `README.md`, `.env.example`, and the relevant handler files together.
- If you changed provider/tool behavior, verify matching docs in `docs/` still describe the active path.
- Keep this file concise; link to deeper docs instead of copying them.

# STAC-Builder — Agent Coding Guidelines

> **Mandatory reference** for any AI agent working on this codebase.
>
> This is a **government-auditable, professional-grade system**. Every line of code must reflect that standard.
>
> Last updated: 2026-03-08

---

## Core Principles

### 1. Correctness Over Convenience

- **Never take shortcuts.** The correct solution is the only acceptable solution, even if it requires more effort or more code.
- **Never hardcode values.** All thresholds, paths, model parameters, and tuning constants belong in `config.yaml`. If a value doesn't exist there yet, **add it** before using it.
- **Never skip or mask problems.** If something doesn't work, diagnose the root cause and fix it properly. Swallowing exceptions, commenting out broken code, or adding workarounds that hide bugs are all prohibited.
- **Never guess.** If you are unsure about how a module works, **read the code first**. If you are unsure about the user's intent, **ask**.

### 2. Analyze Before Acting

- **Read before writing.** Before modifying any file, read the relevant code thoroughly. Understand the existing flow, the callers, the call chain, and the side effects.
- **Trace the full path.** A change in one module may cascade through workers, the pipeline manager, the API, and the frontend. Verify all touchpoints.
- **Check the config.** Before adding any parameter, verify whether it already exists in `config.yaml`, `reconstruction_config_builder.py`, or other config pathways.
- **Check the vendor code.** `Reconstruction_Streaming`, SAM3, and CloudCompPy are vendored with specific expectations. Understand their contracts before wrapping.

### 3. Always Consult the User

- **No unilateral architectural decisions.** Propose changes; wait for approval before implementing.
- **No surprise refactors.** If the scope of a fix grows beyond what was asked, stop and explain what's needed.
- **No assumptions about behavior.** If a feature request is ambiguous, ask for clarification rather than guessing.
- **Report findings honestly.** If a proposed approach won't work or has trade-offs, say so upfront.

---

## Code Quality Standards

### Readability

- **Code is documentation.** Write code that reads clearly without needing excessive comments. When comments are needed, they should explain *why*, not *what*.
- **Consistent naming.** Follow existing conventions in the codebase (`snake_case` for Python, `camelCase` for TypeScript/React).
- **Structured imports.** Group by standard library → third party → vendor → local modules.
- **No dead code.** Remove commented-out code, unused imports, and unreachable branches.

### Robustness

- **Type hints everywhere** in Python function signatures.
- **Graceful error handling.** Use specific exception types, log context, and propagate errors with meaningful messages.
- **Validate inputs** at module boundaries (API endpoints, worker entry points, config loading).
- **Use `.get()` with defaults** when reading dictionaries that may have missing keys, and document what the default means.

### Configuration

- **Single source of truth:** `config.yaml` → read by `config.py` or specific builders (e.g., `reconstruction_config_builder.py`).
- **No magic numbers in logic.** Extract to config or to named constants with docstrings.
- **Defaults must match vendor internals.** When exposing a vendor parameter, the default in `config.yaml` must exactly replicate the vendor's original behavior.

### Testing & Verification

- **Verify before declaring done.** After any change, confirm: Does it build? Does the pipeline run? Do the logs show the expected flow?
- **Check edge cases.** Missing files, empty frame directories, single-frame segments, non-consecutive segments — these are production scenarios.
- **Log tracing.** Any significant operation should produce a log line that a human can follow in the console.

---

## Architecture Rules

### Pipeline & Workers

- The server process **never** loads GPU models. All ML inference happens inside subprocess workers.
- Workers communicate exclusively via `WorkerPipe` IPC (progress, log, done, error).
- Workers must call `pipe.check_cancel()` periodically and exit cleanly on cancellation.
- Workers must clean up GPU memory (`torch.cuda.empty_cache()`, `gc.collect()`) on exit.

### Data Flow

- All path resolution goes through `project_paths.py` (`ProjectPaths` / `SourceContext`). Never build paths manually.
- Point clouds carry origin traceability (`frame_global`, `pixel_row`, `pixel_col`). This chain must never be broken.
- Gravity alignment is computed once (first chunk) and shared. The transform is persisted to `floor_transform.npz`.

### Frontend

- State that affects rendering must go through React refs (not state) to avoid re-renders.
- WebSocket messages follow a typed protocol (`type` field as discriminator).
- Potree LOD is managed by `PotreeLoader` — never load the full cloud at once.

---

## Audit Compliance

This codebase is subject to governmental audit. Therefore:

- **Traceability**: Every processing step must produce datable, identifiable artifacts (JSON metadata, NPZ with origins, timestamped logs).
- **Reproducibility**: Given the same inputs and config, the pipeline must produce identical results.
- **Transparency**: No hidden logic, no obfuscated algorithms, no undocumented side effects.
- **Change history**: Significant decisions should be documented in context (commit messages, this file, ROADMAP.md).

---

## Living Document

This file evolves with the project. When adding new modules, changing architectural patterns, or discovering new gotchas, **update this document**. It is the contract between the development team and any AI agent assisting the project.

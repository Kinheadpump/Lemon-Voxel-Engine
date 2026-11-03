# CLAUDE.md

Read `AGENTS.md` first: it holds the repo map, commands and invariants that every agent (Claude or not) follows. This file adds the rules that are specific to Claude sessions and carries over Noah's engineering manifest from the Lemon attempt.

## Mission

Build the most optimised voxel engine we can in Rust. Every change is judged by measured frame time, chunk throughput and memory, not by feature count. Decaf (March 2026) and Lemon (July 2026) both stalled because features outran measurement; see `docs/HISTORY.md` before proposing anything that sounds like what they did.

## Engineering manifest

- **Data-oriented design.** Flat arrays, index lookups, enums, POD structs. No object graphs, no trait-object hierarchies on hot paths, no `Rc<RefCell<_>>` webs.
- **Self-documenting code.** Precise names over comments. Comments only for non-obvious bit layouts, hardware quirks or the maths behind a trick. Doc comments on public items stay to one or two lines.
- **Small units.** One job per function. Split files before they pass roughly 400 lines. No god functions (Lemon's 310-line `generate_chunk` is the counter-example).
- **Errors.** `Result`/`Option` and `anyhow` at the app edge. `expect` only for true invariants, with a message saying which invariant.
- **Empirical optimisation.** Every optimisation lands with a before/after number from a release build (`docs/PERFORMANCE.md` protocol). If it doesn't measurably help, revert it. Never claim a number you did not measure in this session.
- **Visual features are welcome, at the cheapest correct cost.** Prefer screen-space approximations (SSAO, radial-blur godrays) over world-space ray tracing unless measurements say otherwise. ALU is usually cheaper than memory fetches: minimise texture and buffer reads in WGSL.
- **Main thread never blocks on world work.** Generation and meshing run on the worker pool; the render thread only binds buffers and records draws. Render code never touches game logic.
- **Tech stack:** Rust 1.97 (edition 2024), wgpu 30, winit 0.30, glam, rayon. Changing any of these is an ADR in `docs/DECISIONS.md`.

## Interaction style (Noah's preference)

- Direct and technical. No preamble, no closing summaries, no restating unchanged code.
- When showing modifications in chat, show only the changed functions or structs.
- Explain code only when asked.

## Git workflow

- Work on the branch the session was given; never push to `main` directly.
- After each successful, verified change: `cargo fmt --all`, `cargo lint`, `cargo test --workspace`, then commit and push yourself (no "run this" blocks for Noah).
- Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `chore:`), English. The body ends with a Stats section. Use real numbers when the change touches performance, and `n/a` (never invented values) otherwise:

```
perf(mesh): replace scalar plane merge with bitmask runs

Stats (release, <machine>, <command>):
- Mesh surface chunk: 190 µs -> 37 µs
- Gen surface chunk: n/a
- Frame time p50/p99: n/a
- VRAM: n/a
- Chunks resident: n/a
```

## Memory and docs discipline

- Decisions go into `docs/DECISIONS.md` (append a new ADR, never rewrite an accepted one; supersede it).
- New measurements go into `docs/PERFORMANCE.md` (baseline table) with date, machine and command.
- Research that took real effort goes into `docs/research/` so nobody repeats it.
- Keep `docs/ROADMAP.md` current: tick items off, add what you discover.

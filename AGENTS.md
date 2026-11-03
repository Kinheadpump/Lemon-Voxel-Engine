# AGENTS.md

Guide for any coding agent (and human) working in this repo. Read this, then `docs/ARCHITECTURE.md`. Claude sessions also read `CLAUDE.md`.

## What this is

A performance-first blocky voxel engine in Rust on wgpu. It is the third attempt; the first two (Decaf, Lemon) are preserved in their own repos and summarised in `docs/HISTORY.md`. Their full audits live in `docs/research/`.

## Repo map

| Path | Purpose | Depends on |
|---|---|---|
| `crates/vx-core` | Chunk constants, `BlockId`, `ChunkPos`/`LocalPos`, `Axis`/`FaceDir`, padded indexing | glam, bytemuck |
| `crates/vx-world` | Palette chunk storage, ring-buffer `ChunkWindow`, terrain generator, padded-neighbour gather | vx-core, fastnoise-lite |
| `crates/vx-mesh` | Binary greedy mesher, 8-byte `PackedQuad`, culled reference mesher for tests | vx-core |
| `crates/vx-render` | wgpu device/tier negotiation, terrain pipeline (vertex pulling + multi-draw indirect), camera, offscreen target | vx-core, vx-mesh, wgpu |
| `crates/vx-app` | `vx` binary: winit window, region build on rayon, headless `--probe` / `--screenshot` | all |
| `docs/` | Architecture, decisions (ADRs), roadmap, performance log, history, research | |

Rule: `vx-core`, `vx-world` and `vx-mesh` never depend on wgpu or winit, so CPU benches compile fast and run anywhere.

## Commands

```sh
cargo build --release                 # builds the vx binary (default member)
cargo run --release -- --radius 16    # windowed fly-cam (click to grab mouse, WASD/Space/Ctrl, Shift = fast)
cargo probe                           # print adapter, render tier, features, limits
cargo run --release -- --screenshot out.png --radius 16   # headless render, works on lavapipe
cargo test --workspace
cargo lint                            # clippy pedantic, -D warnings
cargo fmt-check
cargo bench-cpu                       # divan benches for mesh + world
```

Headless GPU: install `mesa-vulkan-drivers` (lavapipe). It exposes every feature we use, so it validates the full path; its timings are meaningless for GPU performance.

## Invariants

- `CHUNK_SIZE` is 32 and `PADDED_SIZE = CHUNK_SIZE + 2` must stay ≤ 64 (a padded column is one `u64`). Changing chunk size is an experiment with an ADR, not a drive-by.
- Voxel index order is `x + S*(z + S*y)` for both `LocalPos::index` and `padded_index`. Rows along x are contiguous.
- `PackedQuad` bit layout (`vx-mesh/src/quad.rs`) is mirrored in `vx-render/src/shaders/terrain.wgsl`. Change both together; the mesher-vs-reference test and the WGSL validation test must stay green.
- Meshing only reads a fully gathered padded chunk. A chunk is meshed only when its six face neighbours exist. Never predict neighbour data from the generator (that was Lemon's biggest cost and bug source).
- Reverse-Z: depth clears to 0, compare `Greater`, infinite far plane.
- No allocation in per-frame code paths; workers reuse scratch buffers (`map_init`, scratch structs).
- Large arrays go on the heap (`new_dense_chunk`, `new_padded_voxels`); clippy's `large_stack_arrays` lint enforces this.

## Workflow for a change

1. Find the roadmap item in `docs/ROADMAP.md`, or add one.
2. If it is a design choice with alternatives, write an ADR in `docs/DECISIONS.md` first (status Proposed), with the measurement that will decide it.
3. Implement with tests. Hot-path changes need a divan bench or an end-to-end number.
4. Measure in release before and after. Record in `docs/PERFORMANCE.md`.
5. `cargo fmt --all && cargo lint && cargo test --workspace`.
6. Conventional commit with the Stats section (see `CLAUDE.md`).

## Parallel agents

When splitting work across agents: give each a disjoint crate or doc, a separate `CARGO_TARGET_DIR` if they build concurrently, and require a written report in `docs/research/` (sources linked, uncertain claims marked). The orchestrator reviews and merges conclusions into `docs/DECISIONS.md`.

# History: Decaf and Lemon

Both earlier attempts are preserved untouched in their own repos. Full audits with file:line references: `docs/research/decaf-audit.md`, `docs/research/lemon-audit.md`. This page is the summary of what they taught us.

## Timeline

| | Decaf | Lemon |
|---|---|---|
| Repo | Kinheadpump/Decaf-Voxel-Engine | Kinheadpump/Lemon-Voxel-Engine |
| Active | 2026-03-16 → 03-20 (5 days, 44 commits, 4 PRs) | 2026-07-06 → 07-13 (8 days, 50 commits) |
| Stack | wgpu 0.20, winit 0.29, std threads + crossbeam, tracy | wgpu 30, winit 0.30, rayon, noiz + bevy_math |
| Size | ~13k lines Rust | ~12k lines Rust + WGSL |
| Tests | 85 pass | 39 pass |
| Ended on | Feature work: textures, HUD, biomes; Hi-Z disabled | Terrain features: mountains, snow, pyramids; stats "n/a" since 07-08 |

## Decaf (first attempt)

Good: 8-byte packed quads with storage-buffer vertex pulling and `multi_draw_indirect`; `Arc` copy-on-write chunk snapshots for meshing; request-id stale-result rejection; atomic save pattern; solid lifecycle tests.

Why it was slow (measured single-thread, release):
- Generation ~4.95 ms/chunk, meshing ~1.22 ms/chunk (scalar greedy with `Instant::now()` inside the slice loop).
- Voxels stored as `u32` (128 KiB per chunk, even for air).
- Streaming, mesh orchestration, upload and culling all on the main thread every frame; per-frame O(5625) rebuild of the load set with sorting and a HashSet.
- CPU culling rebuilt and re-uploaded the whole draw list every frame; Hi-Z read back to the CPU with 1–3 frames latency, so it was disabled.
- Missing neighbours treated as air, so most chunks were meshed twice.

## Lemon (second attempt)

Good (we port these ideas): 32-bit face format with vertex pulling and `draw_index`; GPU compute culling writing indirect args plus a count for `multi_draw_indirect_count`; the pool slot doubling as GPU index; reverse-Z; HZB min-reduction; column-bitfield exposure (`col & !(col >> 1)`); rayon backpressure; cascade fitting for CSM. Lemon reached 500–750 FPS at render distance 32 (4225 chunks, ~217 visible) on an Intel iGPU (commit e97f5c9), the only real frame numbers in either repo.

Why it stalled:
- The async mesher had no neighbour data and re-evaluated the procedural generator to predict borders (`boundary_planes`): 1.4–2.4 ms per chunk against 0.15–0.22 ms for the mesh itself, and the source of most bug-fix commits (gaps, races with edits).
- The mesher was only half binary: column bitfields, then a scalar 2D greedy merge. A commit claimed 643 µs; the bench now reports 6.2 ms per batch.
- Fixed 64 KiB `u16` arrays for every chunk including air, pool sized to the whole window: ~2.4 GB RAM for 38k chunks. LOD was attempted in debug builds and deleted two days later.
- Single-phase occlusion with a 48-block radius hack; all 6 face directions drawn for every visible chunk; shadows redrew the union of chunks in every cascade.
- A "SIMD terrain" commit that was not SIMD (noiz sampled point by point).
- `config.rs` (918 lines) held each setting three times; heavy German doc comments despite the "no comments" rule.
- Commit-message claims were not re-verified; measurements were often taken in debug builds.

## Lessons that shape this attempt

1. Mesh from real neighbour data only (padded gather), never from predictions.
2. Storage must scale with content: uniform chunks cost nothing, palettes for the rest.
3. Measure every claim in release, record it, and re-check old numbers when touching the code.
4. Keep the main thread free: workers produce, main thread only uploads and draws.
5. Ship performance infrastructure (benches, headless render, timers) before visual features.
6. One source of truth per setting; no triple config structs.

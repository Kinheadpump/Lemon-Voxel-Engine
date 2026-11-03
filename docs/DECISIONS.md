# Architecture decision records

Append-only. Status is Accepted, Proposed (needs the stated measurement), or Superseded by ADR-n.

## ADR-0001 New workspace instead of continuing Decaf or Lemon (Accepted, 2026-11-03)

Both prior engines' cores (storage layout, neighbour handling, world index, mesher) need replacing, and those are the pieces everything else hangs off. Lemon's renderer ideas are good, but its code is tied to the 64 KiB pool, the prediction mesher and a 918-line config. A fresh Cargo workspace lets us port ideas deliberately, keeps both old repos intact as references, and gives a clean git log for performance tracking. Recommended home: a new repo `Kinheadpump/Espresso-Voxel-Engine` (needs Noah's approval).

## ADR-0002 Workspace split core / world / mesh / render / app (Accepted)

CPU crates stay free of wgpu/winit so benches compile in seconds and run headless. The render crate knows nothing about terrain generation. The app crate wires them together.

## ADR-0003 32³ chunks, padded 34³ mesher input, u64 columns (Accepted, revisit with measurement)

32³ keeps chunk-level culling granular and the padded column fits a `u64`. The alternative from the research is 62³ data padded to 64³ (cgerikj / TanTanDev), which halves per-chunk overhead (draws, cull work, upload count) at the cost of coarser culling and remeshing more on edits. `CHUNK_SIZE` is a single constant with `PADDED_SIZE ≤ 64` asserted, so the experiment is cheap. Decide with: mesh µs per voxel, draws visible, and frame time at the same view distance.

## ADR-0004 8-byte quads with vertex pulling (Accepted; 4-byte variant Proposed)

`u64` quads leave 15 free bits for AO/light and a 16-bit block id. Lemon's 4-byte format fits 32³ chunks with ≤128 block types and halves vertex-fetch bandwidth. Decide with: GPU time of the opaque pass on a real GPU at radius 32 for both formats.

## ADR-0005 Job system: rayon + main-thread priority heap (Proposed)

Research (`docs/research/rust-tooling.md`): rayon has no priorities or cancellation. Plan: dedicated `ThreadPool` (cores − 1), main thread keeps a distance-sorted heap and submits at most ~2× worker count jobs per frame with `spawn_fifo`, results over crossbeam, stale results dropped by a per-slot generation counter. Replace only if profiling shows rayon overhead.

## ADR-0006 Mesh only with real neighbours (Accepted)

`gather_padded` returns `MissingNeighbour` instead of guessing. A chunk is meshed once all 6 face neighbours are resident; edits remesh the chunk and any neighbour whose border changed. This removes Lemon's `boundary_planes` cost (5–10× the mesh itself) and its bug class.

## ADR-0007 Palette storage with uniform shortcut (Accepted)

See `docs/ARCHITECTURE.md` for numbers. Open follow-ups: palette compaction after many edits, and a `u8` fast path if decode shows in profiles (decode is 17–24 µs per surface chunk today).

## ADR-0008 Ring-buffer chunk window instead of a hash map (Accepted)

O(1) neighbour lookup without hashing, trivial eviction on camera movement. A hash map (foldhash, packed `u64` key via `ChunkPos::packed_key`) is reserved for sparse sets such as persisted edits.

## ADR-0009 Noise: fastnoise-lite now, SIMD later (Accepted for now)

Bench (`docs/research/rust-tooling.md`): fastnoise-lite was the fastest scalar crate (44 ns per 2D sample with 4 octaves); simdnoise (SIMD) was ~3× faster but is unmaintained. Terrain is heightmap-first (1024 2D samples per column chunk); 3D noise for caves will be sampled on a coarse grid and interpolated. A hand-written `wide::f32x8` fBm is the next step if generation dominates the profile.

## ADR-0010 GPU-driven culling path (Proposed)

Port Lemon's compute cull (frustum + per-direction back-face rejection + Hi-Z), extended to two-phase occlusion, writing indirect args and a count for `multi_draw_indirect_count`. Fallback without the count feature: same compute pass writes zero-instance draws, drawn with plain `multi_draw_indirect`. Decide on the Hi-Z phase only once overdraw shows in GPU timings.

## ADR-0011 Far terrain / LOD (Proposed, research)

Candidates: meshed LOD rings with downsampled chunks (Distant Horizons style), or ray-marching a coarse brick map for the far field. Aokana (2025) reports 10¹⁰ voxels in ~6 ms but is read-only. Decide after the near field is GPU-driven and measured, with a prototype of each at the same view distance.

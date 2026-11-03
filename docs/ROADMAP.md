# Roadmap

Ordered by expected impact on the core goal (view distance and frame time per watt), measurement infrastructure first. Each item names how it will be verified.

## Done (2026-11-03)

- [x] Workspace, toolchain pin, lints, CI.
- [x] Palette chunk storage with uniform shortcut, ring-buffer window, heightmap terrain.
- [x] Binary greedy mesher with reference-equivalence tests (~35 µs per surface chunk).
- [x] wgpu 30 vertex-pulling terrain pipeline with one `multi_draw_indirect`.
- [x] Windowed fly-cam, headless `--probe` and `--screenshot` on lavapipe.

## Next: measurement first

1. **GPU timestamps for every pass** plus an `--bench <path>` mode that flies a scripted camera path for N frames and prints JSON (frame p50/p99, GPU pass times, chunks resident, VRAM). The commit Stats section is filled from this. Verify: runs headless on lavapipe, numbers on Noah's GPU recorded in `docs/PERFORMANCE.md`.
2. **Tracy behind a feature flag** via the `profiling` crate (zones in gen, mesh, upload, frame). Verify: zero cost with the feature off (bench unchanged).

## Then: streaming core

3. **Chunk streaming** around the camera using `ChunkWindow`: dedicated rayon pool, priority heap, bounded in-flight jobs, generation-counter cancellation (ADR-0005), and meshing only when 6 neighbours exist (ADR-0006). Verify: no frame over budget while flying at sprint speed; chunks/s throughput.
4. **GPU buffer suballocation** with `offset-allocator` for quads (one big buffer, O(1) alloc/free) and staged uploads with a byte budget per frame. Verify: upload ms per frame, fragmentation after long flights.
5. **GPU culling** (ADR-0010): frustum + per-direction back-face rejection in compute, `multi_draw_indirect_count` where available. Verify: GPU ms and draws issued at radius 32/48.

## Then: scale

6. **Chunk-size experiment** 32³ vs 62³ (ADR-0003) and **4-byte vs 8-byte quads** (ADR-0004), using the bench mode.
7. **Two-phase Hi-Z occlusion** if overdraw shows. Verify: GPU ms in caves and mountains.
8. **Far-field LOD** prototypes (ADR-0011).
9. **SIMD terrain noise** (ADR-0009) if generation dominates the streaming profile.

## Then: looks (each with a cost budget)

10. Editing with remesh of affected neighbours; persistence (region files, palette-compressed).
11. Per-vertex AO in the free quad bits (only merge equal-AO faces; measure the quad-count increase) or SSAO.
12. Flood-fill sky and block light.
13. Texture array, transparent pass (water/glass), cascaded shadows with per-cascade GPU culling and cached far cascades.
14. Camera-relative rendering for large coordinates (world origin rebasing).

## Known issues

- Greedy meshing can create T-junctions; a faint sub-pixel seam was visible in one headless frame. Check on real GPUs before deciding whether it needs fixing.
- Region build in the app is a fixed square; no streaming yet.

# Architecture

Status as of 2026-11-03: a vertical slice. A fixed region is generated and meshed on rayon workers, then drawn from one storage buffer with `multi_draw_indirect`. Streaming, GPU culling, editing and lighting are roadmap items (`docs/ROADMAP.md`).

## Data flow

```
TerrainGenerator::generate (worker, rayon map_init scratch)
        │  ChunkVoxels: Uniform(BlockId) | Packed{palette, 1/2/4/8/16-bit indices}
        ▼
ChunkWindow<ChunkVoxels>  (ring buffer, slot = pos.rem_euclid(extent), no hashing)
        │  gather_padded: centre + 6 face-neighbour slabs → 34³ dense BlockId array
        ▼
BinaryGreedyMesher::mesh (worker, reusable scratch)
        │  ChunkMesh { quads: Vec<PackedQuad /*u64*/>, direction_ranges: [Range; 6] }
        ▼
TerrainRenderer::upload_static (main thread)
        │  one quad storage buffer + chunk-origin storage buffer + DrawIndirectArgs per (chunk, direction)
        ▼
terrain.wgsl: vertex pulling (vertex_index / 6 → quad, instance_index → chunk slot)
        │  reverse-Z, infinite far plane, back-face culling, fog
        ▼
multi_draw_indirect (one call for the whole world)
```

## Storage (`vx-world/src/storage.rs`)

- Chunks are 32³. `Uniform` chunks (all air, all stone) have no heap allocation.
- `Packed` chunks hold a palette and power-of-two bit widths (1, 2, 4, 8, 16) so an index never straddles a `u64`. A palette that outgrows its width doubles the width.
- Measured: radius-32 region (22,445 chunks) uses 58.6 MiB of voxel heap. The same chunks in Lemon's fixed `u16` layout would take 1.37 GiB, and 2.74 GiB in Decaf's `u32` layout.

## Meshing (`vx-mesh/src/binary_greedy.rs`)

1. One pass over the padded 34³ array builds opaque bit-columns along X, Y and Z (`u64` each).
2. Face masks per direction: `col & !(col >> 1)` (positive) or `col & !(col << 1)` (negative), shifted to drop the padding bits.
3. Each face bit is scattered into a per-layer, per-block-type plane of 32 `u64` rows.
4. Each plane is merged with bit runs: `trailing_zeros` finds the run start, `trailing_ones` its width, and rows below are absorbed while they contain the whole run.
5. Quads are emitted grouped by direction, so the renderer can cull whole directions per chunk.

Correctness is pinned by a test that expands every quad to unit faces and compares against a naive culled mesher on random and edge-case chunks.

## Quad format (`vx-mesh/src/quad.rs`, mirrored in `terrain.wgsl`)

| Bits | Field |
|---|---|
| 0–5, 6–11, 12–17 | x, y, z of the quad's min voxel (chunk-local) |
| 18–23, 24–29 | width − 1 (along u), height − 1 (along v) |
| 30–32 | face direction (0 = +X, 1 = −X, 2 = +Y, 3 = −Y, 4 = +Z, 5 = −Z) |
| 33–48 | block id |
| 49–63 | free: reserved for AO / light (ADR-0004) |

Axis mapping (`Axis::depth_u_v_to_xyz`): X → (d, u = y, v = z), Y → (u = x, d, v = z), Z → (u = x, v = y, d).

## Rendering (`vx-render`)

- `GpuContext` requires `INDIRECT_FIRST_INSTANCE` and opportunistically enables `MULTI_DRAW_INDIRECT_COUNT`, timestamp queries and `SHADER_DRAW_INDEX`. `RenderTier` records whether the GPU-driven count path is available. Plain `multi_draw_indirect` is native on Vulkan/DX12 and emulated on Metal/GL/WebGPU.
- Per-draw chunk data is fetched via `instance_index` (= `first_instance` = chunk slot). Note: on DX12, `instance_index` may not include `first_instance`; this must be tested on DX12 before relying on it there (see `docs/research/wgpu-capabilities.md`).
- The indirect buffer is built once on upload; there is no per-frame CPU work besides the camera uniform.

## Threads

- rayon global pool for region generation and meshing (`par_iter().map_init(scratch)`), with results sent to the main thread over a crossbeam channel. The main thread never waits on it.
- Planned: a dedicated pool with a distance-priority heap on the main thread, bounded in-flight jobs and generation-counter cancellation (ADR-0005).

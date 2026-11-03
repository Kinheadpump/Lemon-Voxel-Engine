# Voxel engine techniques survey (Rust/wgpu, Minecraft-like) - 2026-11-03

Legend: [S] = verified from a fetched page (URL given). [U] = uncertain / from background knowledge, NOT verified this session. Several areas had thin search results; gaps are listed at the end.

## 1. Meshing

| Mesher | Evidence |
|---|---|
| Naive culled (visible faces) | block-mesh-rs `visible_block_faces` ~40 M quads/s single core (2.5 GHz i7) [S] https://github.com/bonsairobo/block-mesh-rs |
| Greedy (scalar) | block-mesh-rs `greedy_quads`: ~1/3 the quads of culled for spherical input but ~3x slower [S] same URL. Rust binary-greedy crate author reports block-mesh-rs ~3 ms/chunk [S] https://docs.rs/crate/binary-greedy-meshing/latest |
| Binary/bitmask greedy | C lib: 50-200 us per chunk typical; 74 us avg single thread, 108 us in thread pool, Ryzen 3800X [S] https://github.com/cgerikj/binary-greedy-meshing . Rust port: `fast_mesh` 65 us (opaque), 90 us (opaque+transparent); `mesh` 300/340 us, Ryzen 5 5500 [S] https://docs.rs/crate/binary-greedy-meshing/latest |

- Chunk size: binary-greedy-meshing is fixed to 62^3 data + 1 voxel padding = 64^3 so one row = one u64 [S] (docs.rs link above; C repo says max 64^3 incl. neighbours). block-mesh-rs example uses 16^3+padding = 18^3 [S].
- Quad format: C lib 8 B/quad, 6-bit x,y,z,w,h + 8-bit voxel type, rendered by vertex pulling [S]; Rust crate u64 `(type<<32)|(h<<24)|(w<<18)|(z<<12)|(y<<6)|x`, six `Vec<u64>` (one per face direction), README recommends instanced rendering [S]. Faces grouped by direction gives you per-direction culling for free.
- T-junctions: not eliminated; mitigated by drawing quads ~1 px larger and eye-space vertex math [S] (cgerikj README).
- AO: **v2 of cgerikj mesher has no AO; baked AO only in old v1.0.0 branch** [S]. Rust crate: no AO mention [S]. Mikola Lysenko: merge quads only when lighting values at vertices are equal [S] https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/ . Consequence (inference, [U] magnitude): per-face AO fragments merges and erodes greedy's gain. Common alternative: compute AO in shader / screen-space (SSAO) or from a voxel-occupancy lookup in the fragment shader, so merging is unaffected.
- Greedy theory: <=8x optimal quads (conjectured ~4x) [S] 0fps link. Merge gain: 8^3 solid cube 3072 naive -> 384 culled -> 6 greedy [S].
- TanTanDev demo (Bevy, binary greedy vs simple culled, criterion benchmarks) exists but its README gives no numbers [S] https://github.com/TanTanDev/binary_greedy_mesher_demo
- Vercidium: https://github.com/vercidium-patreon/meshing is a C#/Silk.NET greedy mesher repo; no numbers on the page [S]. His videos' claims were not verifiable here [U].
- Face-direction splitting: a forum reply claims rendering only camera-facing direction groups roughly halves render cost [S, anecdotal] https://nyc1.lr.ggtyler.dev/r/VoxelGameDev/comments/1f93uke/voxel_game_optimizations/llye0cn/?context=3

## 2. Storage

- Minecraft: 16^3 sections; per-section paletted container: single-valued (0 bits), indirect 4-8 bits/entry, direct (global registry, 15 bits+); light stored separately as 4 bits/voxel [S] https://wiki.vg/Chunk_Format . Memory for an indirect-4bit section = 4096*0.5 B = 2 KiB + palette (arithmetic).
- Vintage Story: 32^3 chunks, protobuf blobs on disk, compressed block+light fields [S] https://wiki.vintagestory.at/index.php/Chunk_Data_Storage ; in-memory layout not documented there.
- Veloren: dev blog on site2 shows chunk generation costs (~40 us to 3.3 ms per chunk for structures) [S] https://veloren.net/devblog-178 ; storage format not verified [U].
- Teardown: dense 3D texture, 1 byte/voxel -> 256-entry palette, two extra mip levels for empty-space skipping [S] https://acko.net/blog/teardown-frame-teardown/
- Aokana: SVDAG per 256^3 chunk, nodes 8-40 B, 4^3 leaves as 64-bit bitmaps, colors stored separately; **no runtime editing, no transparency** [S] https://arxiv.org/html/2505.02017v1
- Sodium/SVO/brickmap tradeoffs: not verified [U]. Practical conclusion: for editable blocky terrain use palette-compressed 32^3 (or 64^3-aligned) sections + single-value fast path; DAGs only for read-only far LOD.

## 3. Rendering

- wgpu: `MULTI_DRAW_INDIRECT_COUNT` is native-only, DX12 + Vulkan only (not Metal/WebGPU) [S] https://wgpu.rs/doc/wgpu_types/struct.FeaturesWGPU.html . Mesh shaders experimental, naga compiles them only for Vulkan [S] same. Plan a fallback (compute-built indirect buffer + plain MDI, or one indirect draw per region).
- offset-allocator crate: Rust port of Aaltonen's OffsetAllocator, O(1) alloc/free, TLSF-like, 256 bins, <=12.5% waste, returns offsets (metadata kept CPU-side) [S] https://github.com/pcwalton/offset-allocator . Fits one big vertex/quad buffer with per-chunk suballocation.
- Sodium: release notes confirm multi-draw chunk rendering "can drastically improve frame rates"; chunk graph/culling CPU optimisations [S] https://github.com/CaffeineMC/sodium-fabric/releases/tag/mc1.16.5-0.2.0 . Region-based buffer arenas and per-face culling details: not verified [U].
- GPU-driven pattern from Aokana (voxel, but ray-marched): frustum cull chunks -> Hi-Z tile cull using previous frame -> render -> rebuild Hi-Z -> second pass for wrongly culled tiles (two-phase) [S] arXiv link above. Same two-phase structure applies to rasterised chunks [U for voxel-specific numbers].
- Nanite-like clusters: no voxel-specific evidence found; for 64^3 chunk quads the chunk is already the cluster [U].

## 4. LOD / far terrain

- Distant Horizons: LODs visual only, stored in SQLite, async distant generation, default LOD radius 64 chunks / vanilla 8 [S] https://craftdownunder.co/guides/mods/distant-horizons . Internal octree/mesher details not verified [U].
- bevy_voxel_world: per-chunk LOD with downsampled (nearest-neighbour) meshing shapes; uses `Unset` border voxels as skirts to hide LOD cracks [S] https://github.com/splashdust/bevy_voxel_world
- Aokana (2025): ~6 ms/frame at 64K^3 (~10^10 voxels) on RTX 3060 Ti; 424 MB VRAM vs 3475 MB HashDAG; ~5% of scene resident; 2-4x faster than HashDAG above 32K; LOD n = 8 children -> 1 voxel (>=2 non-empty), streaming error `(ChunkSize*Factor) - dist` [S] arXiv link. Caveat: pre-built, static, Unity, single GPU.
- Teardown-style ray marching: great for dynamic voxel objects, shadows/AO via 1-bit coarse shadow volume (262 MB for the Marina level), expensive volumetrics at half res [S] acko link. Not designed for Minecraft-scale worlds.
- Douglas Dwyer, John Lin, Voxel Space: no primary sources retrieved [U].
- Judgement ([U], derived from the above): biggest view distance per ms = rasterised chunk meshes (greedy, direction-culled) near; coarser-resolution meshed LOD rings (octree/clipmap of 2x downsampled meshes) mid/far; SVDAG ray-marched far field is the research frontier (Aokana) but conflicts with editing.

## 5. Lighting

- Voxel-grid light in Minecraft: 4-bit block+sky light per voxel [S] wiki.vg link. Flood-fill propagation details/costs: not sourced [U].
- Teardown: stochastic sky AO via volumetric tracing at ~2 spp + small screen-space step, spatial+temporal denoising [S] acko link; cost drivers are volumetric lighting and multiple temporal passes.
- Vertex/baked AO vs greedy: see section 1. SSAO/ray-traced GI costs in blocky engines: no primary numbers found [U]. CLAUDE.md prefers screen-space approximations; SSAO/GTAO on depth+normal is consistent with that.

## 6. Rust projects

- vx_bevy: block-mesh greedy, async task pool, 16-chunk render distance, ~100 fps on GTX 1060, one mesh per chunk, no stitching/LOD, unmaintained [S] https://github.com/Game4all/vx_bevy
- bevy_voxel_world: procedural + persistent HashMap override layers, block-mesh simple mesher, per-LOD shapes, procedural cache grows with spawn distance, "various hard-coded assumptions", perf not prioritised, tied to Bevy versions [S] https://github.com/splashdust/bevy_voxel_world
- block-mesh-rs: solid, but ~3 ms/chunk greedy vs ~65-300 us for bitmask [S].
- Veloren: server-side worldgen cost work (sub-ms structures) [S] devblog-178.
- all-is-cubes: not researched [U].
- Mistakes to avoid (evidence-backed): one entity/mesh per chunk (vx_bevy), unbounded procedural caches (bevy_voxel_world), engine-framework coupling to version churn, scalar greedy on the hot path.

## 7. Numbers cheat sheet

- Bitmask greedy: 65-340 us/chunk (62^3) Rust, 74-108 us C [S].
- Scalar greedy: ~3 ms/chunk [S, crate author's claim].
- Culled: 40 M quads/s/core [S].
- Quad: 8 B (u64) vs 4-6 vertices; vertex pulling/instancing [S].
- Aokana: 6 ms @ 10^10 voxels, 424 MB [S].
- Veloren structure gen: 40 us-3.3 ms/chunk [S].
- Chunks visible at N render distance / VRAM per chunk: not found in sources; compute yourself: (2N+1)^3 sections before culling; e.g. 32 chunks of 32 -> ~1 km radius [U].

## Recommendation (order to evaluate)

1. Binary greedy meshing on 62^3(+2) chunks, u64 quads, grouped per face direction, rayon workers. Benchmark vs naive culled in-engine; target <=0.3 ms/chunk. Keep AO out of the mesher.
2. Vertex pulling / instanced quads from one big storage buffer, suballocated with offset-allocator. Measure vs conventional vertex buffers.
3. Per-chunk face-direction masks + CPU/compute frustum culling, then indirect draws (MDI-count where supported, fallback otherwise).
4. Compute Hi-Z occlusion (two-phase) - evaluate only if overdraw/draw count is the bottleneck; cave-heavy terrain benefits most [U].
5. Palette-compressed storage with single-value sections (Minecraft style).
6. AO: screen-space (GTAO/SSAO) or shader-side lookup; flood-fill light in 4-bit grid.
7. Far field: meshed LOD rings (2x downsample per ring) first; prototype SVDAG ray-marched far field (Aokana) only as research, since it lacks editing and transparency.

Avoid: vertex-baked AO inside greedy merge (fragmentation), per-chunk draw calls, scalar greedy, full voxel ray-tracing GI (Teardown-level cost), Nanite-style cluster hierarchy (no evidence it pays off for blocky quads), and relying solely on MDI-count (no Metal/web).

## Gaps (searches failed or returned nothing usable)
Sodium internals, Veloren storage, SVO/brickmap numbers, Vercidium/TanTanDev timings, Douglas Dwyer, John Lin, all-is-cubes, SSAO/GI costs in voxel engines, 2024-2026 papers beyond Aokana.

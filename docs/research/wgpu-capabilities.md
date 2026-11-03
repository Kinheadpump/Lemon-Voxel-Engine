# wgpu research (as of 2026-11-03)

Legend: [V]=verified from fetched source/local run, [U]=uncertain/from memory/not verified.

## 1. Versions and feature status
Latest on crates.io: **wgpu 30.0.1 (2026-08-21/22)** [V, https://crates.io/api/v1/crates/wgpu/versions]. Timeline: 26.0 (2025-07-10), 27.0 (2025-10-01), 28.0 (2025-12-18), 29.0.0 (2026-03-19; .1 03-26, .2/.3 05-01, .4 07-02), 30.0.0 (2026-07-01), 30.0.1 (2026-08-22). The previous attempt's wgpu 30 is therefore current; only a patch bump to 30.0.1 is available.
Sources: https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md , https://docs.rs/wgpu/30.0.1/wgpu/struct.Features.html

Notable changes (from CHANGELOG [V] for 28-30; 25-27 not read [U]):
- v30: immediates aligned with WebGPU spec (Features::IMMEDIATES is now in the *webgpu* feature set; present on llvmpipe); Metal full mesh shader support incl. WGSL; naga HLSL mesh writer (DX12 mesh complete); Metal bindless storage buffer binding arrays; DX12 partially-bound binding array fix; `enable wgpu_binding_array;`; ray-tracing pipelines SPIR-V out; query-set init tracking (unwritten slots resolve to 0).
- v29: `@builtin(draw_index)` on Vulkan (#8883); 29.0.2: only request shaderDrawParameters when SHADER_DRAW_INDEX requested (fixes V3DV/SwiftShader device creation); DX12 multi-draw stride debug_assert fix; wgsl RT pipelines; TLAS binding arrays; Metal RT acceleration structures; WebGPU occlusion queries.
- v28: mesh shaders "fully supported on Vulkan, supported on Metal and DX12 with passthrough shaders" (v30 added naga paths for Metal/DX12).
- Earlier (changelog): `Features::MULTI_DRAW_INDIRECT` was removed: multi_draw_indirect is unconditionally available when `DownlevelFlags::INDIRECT_EXECUTION` (emulated as loop of draw_indirect where no native MDI). `MULTI_DRAW_INDIRECT_COUNT` "cannot be emulated efficiently" so it is native-only. `DownlevelFlags::VERTEX_AND_INSTANCE_INDEX_RESPECTS_RESPECTIVE_FIRST_VALUE_IN_INDIRECT_DRAW`: enabled on all backends **except DX12** (instance_index/vertex_index start at 0 there) -- critical for the "instance_index as draw id" trick on DX12.

Feature/backend matrix (from docs.rs Features, [V]):
| Feature | Vulkan | DX12 | Metal | GL | WebGPU |
|---|---|---|---|---|---|
| multi_draw_indirect (no feature) | native | native | native (added wgpu 0.14-ish) [U] | emulated loop | emulated loop |
| MULTI_DRAW_INDIRECT_COUNT | yes (1.2 or KHR_draw_indirect_count) | yes | **no** | no | no |
| INDIRECT_FIRST_INSTANCE | mostly | yes | Apple3+/Mac1+ | desktop GL 4.2 + ARB_shader_draw_parameters | yes (web std) |
| SHADER_DRAW_INDEX | yes | not yet | not yet | GLES | no |
| IMMEDIATES (ex push constants) | yes | yes [U] | yes [U] | [U] | in WebGPU spec direction; limit max_immediate_size=256 on lavapipe |
| EXPERIMENTAL_MESH_SHADER | yes (EXT_mesh_shader) | yes | yes | no | no |
| Binding arrays (TEXTURE_/BUFFER_/STORAGE_RESOURCE_) | yes | textures only (+non-uniform idx) | textures; storage bufs since v30 | no | no |
| SHADER_INT64_ATOMIC_MIN_MAX | yes | SM6.6 | MSL2.4 | no | no |
| SHADER_INT64_ATOMIC_ALL_OPS | yes | SM6.6 | no | no | no |
| SUBGROUP (compute+fragment) | yes | yes | yes | no | no |
| SUBGROUP_VERTEX / BARRIER | V / V+M | - | barrier only | | |
| TIMESTAMP_QUERY (pass writes) | yes | yes | yes | ARB_timer_query | yes |
| TIMESTAMP_QUERY_INSIDE_ENCODERS/PASSES | yes | yes | AMD/Intel only (not Apple, tiled) | ARB_timer_query | no |
| PIPELINE_CACHE | **yes** | no (unimplemented) | no | no | no |
| EXPERIMENTAL_RAY_QUERY | yes | (HLSL path; UB fix in v29) [U] | yes since v29 acceleration structs [U] | no | no |
| PASSTHROUGH_SHADERS | V | D | M | | W |

## 2. GPU-driven rendering best practice in wgpu
- Pipeline: compute cull (frustum + Hi-Z + optional per-face-direction cull) -> writes `DrawIndexedIndirectArgs[]` + a u32 count -> `multi_draw_indexed_indirect_count` (Vulkan/DX12) with fallback `multi_draw_indexed_indirect` over max_count entries with instance_count=0 for culled draws (Metal/GL/Web). Compact the buffer with atomicAdd if using the count path; leave zero-instance entries for the fallback.
- Per-draw id: `@builtin(draw_index)` only Vulkan (needs SHADER_DRAW_INDEX). Portable alternative: put chunk id into `first_instance` (needs INDIRECT_FIRST_INSTANCE; works on V/M/GL-ish/Web) and read `@builtin(instance_index)`. **DX12 caveat:** instance_index does not include first_instance per the downlevel flag above [V from changelog line 3753; may have been fixed since [U]] -- test on DX12; fallback is immediates/ a per-draw-id lookup via base_vertex trick or splitting draws.
- Vertex pulling: Kitware benchmark (RTX A5000, GL) found indexed vertex pulling 2-11% faster than classic VBO, non-indexed pulling worse than indexed; no CPU cost (https://www.kitware.com/one-polydata-mapper-instead-of-two-measuring-vertex-pulling-on-the-desktop/ ). One GPU/driver only: treat as hint. For voxels: u32/u64-packed quad (pos, face, ao, tex) in a storage buffer, 4 verts from `vertex_index` (6 per quad with a shared static index buffer pattern: quad_id = vi/6 or use shared index buffer, `vertex_index/4`). Needs VERTEX_STORAGE downlevel (present). Mind `max_storage_buffer_binding_size` (llvmpipe 128 MiB; desktop 2 GiB+ typical; query limits and split into several big arenas).
- Suballocation: `offset-allocator` crate 0.2.0 (port of Aaltonen's OffsetAllocator, O(1) alloc/free, last published 2024-07) https://crates.io/crates/offset-allocator -- fits one giant quad arena + indirect-args slot allocator. Request limits.max_buffer_size / max_storage_buffer_binding_size up from adapter limits (defaults are conservative).
- Uploads: `queue.write_buffer` is fine for many small meshes (wgpu stages internally, extra copy); `Queue::write_buffer_with` avoids a copy; `mapped_at_creation` best for one-off large buffers; `StagingBelt` for per-frame ring uploads. wgpu 30 has MAPPABLE_PRIMARY_BUFFERS on llvmpipe (not desktop discrete in general [U]). Batch all chunk uploads into one pass per frame and cap bytes/frame. Measure -- per CLAUDE.md rule.
- Avoid stalls: never `device.poll(Wait)` per frame; read back (e.g. visible count, timestamps) with map_async two-three frames late; use `Queue::submit` once; keep pipelines pre-built; Vulkan PIPELINE_CACHE (persist `PipelineCache::get_data` to disk) cuts startup shader compile (Vulkan only).
- Hi-Z: render depth -> compute downsample chain (max-reduce 2x2, one dispatch per mip or single-pass with `SUBGROUP` / workgroup memory; wgpu has no storage-texture mip views limit issue but needs one view per mip; R32Float storage write needs `TEXTURE_FORMAT` support -- basic, no feature needed). Use previous-frame HZB + reproject (two-phase culling) to avoid popping. Existing code in <repos>/Lemon-Voxel-Engine/src/engine/render/hzb.rs is the pattern.

## 3. Mesh shaders for voxels
Status: experimental, API feedback in https://github.com/gfx-rs/wgpu/issues/7197 ; docs https://github.com/gfx-rs/wgpu/blob/trunk/docs/api-specs/mesh_shading.md. `enable wgpu_mesh_shader;`, `@task`/`@mesh`, `create_mesh_pipeline`, `draw_mesh_tasks(_indirect/_count)`. Features docs: "Some Mesa drivers (LLVMPIPE, not RADV) reportedly fail on Naga output", **native only** (no WebGPU), no queries in mesh passes, primitive-index pending. Hardware: needs NV Turing+/AMD RDNA2+/Intel Arc/Apple M3+? [U].
Verdict: **Not worth it now** for a blocky engine as primary path. Quad-based vertex pulling + multi_draw_indirect_count already removes the vertex-input bottleneck; mesh shaders' wins (per-meshlet culling, no index buffer, amplification) are real on NV/AMD but remove portability to Metal-older/WebGPU/GL, and the API is explicitly unstable. Worth a **later isolated experiment**: a task shader that culls 16^3 sub-chunk quad ranges + mesh shader that expands packed quads (no intermediate buffers), gated behind the feature with the indirect path as fallback; measure with timestamps before adopting.

## 4. Ray marching voxels (far LOD)
- Feasible in compute (and fragment) with plain WGSL: Amanatides-Woo DDA over a brick/64-tree or two-level grid in storage buffers; storage_buffer limit and the 128 MiB-ish default storage binding limit require bricking. No hardware RT needed. Hardware ray query (EXPERIMENTAL_RAY_QUERY) is Vulkan-only/experimental -- skip.
- Examples found: shocovox (sparse voxel octree w/ wgpu backend) https://github.com/davids91/shocovox ; roxlap-gpu (voxlap-style GPU renderer, docs.rs/crate/roxlap-gpu) [both only seen in search results, not inspected -- U]. Bevy/others: various "voxel DDA wgpu" demos [U].
- Practical design: render far LOD at half/quarter resolution in compute into a color+depth target, composite with depth test against the rasterised near field (write gl_FragDepth in a full-screen fragment pass or do depth compare in composite). Use mipped brick occupancy (64-bit bitmask per 4^3 brick: u64 needs SHADER_INT64 (present on llvmpipe, [U] elsewhere) otherwise 2x u32) for empty-space skipping. CLAUDE.md prefers screen-space/cheap approximations: coarse-LOD heightmap ray march (2.5D) is much cheaper than full 3D DDA for far terrain.

## 5. GPU profiling
- TIMESTAMP_QUERY (+ _INSIDE_ENCODERS/_INSIDE_PASSES where available) [V matrix above]; `timestamp_writes` on pass descriptors works everywhere TIMESTAMP_QUERY exists (incl. WebGPU); `write_timestamp` mid-pass not on Apple/tiled GPUs. Multiply by `queue.get_timestamp_period()`. Resolve to buffer, copy to mappable, read 2-3 frames later. Existing: src/engine/render/gpu_timer.rs.
- `wgpu-profiler` 0.28.0 (2026-07-31) depends on wgpu ^30.0.0 [V via crates.io API] https://crates.io/crates/wgpu-profiler : scopes, nested, chrome trace export, Tracy (`tracy` feature) and puffin integration [U on exact feature names]. Tracy GPU zones are fed through its `tracy-client` gpu context [U].
- Also: RenderDoc/Nsight capture; PIPELINE_STATISTICS_QUERY present on llvmpipe.

## 6. Headless testing / CI
- **Local container**: initially no Vulkan ICD (`/usr/share/vulkan/icd.d` missing, libvulkan1 present only). `apt-get update && apt-get install -y mesa-vulkan-drivers vulkan-tools` worked (I installed them system-wide; Mesa 25.2.8, lavapipe `lvp_icd.json`). `vulkaninfo`: llvmpipe (LLVM 20.1.2, 256 bits), Vulkan 1.4.318, 4 cores.
- **Probe crate**: <scratch>/wgpu-probe (wgpu 30.0.1 + pollster 0.4; note v30 API: `Instance::new(InstanceDescriptor::new_without_display_handle())`, `enumerate_adapters(..)` is async). Output in out.txt there. Result on llvmpipe/Vulkan (DeviceType::Cpu), builds in ~70 s:
  - Features present: TIMESTAMP_QUERY (+INSIDE_ENCODERS/PASSES), MULTI_DRAW_INDIRECT_COUNT, INDIRECT_FIRST_INSTANCE, IMMEDIATES, SHADER_DRAW_INDEX, SUBGROUP(+BARRIER, subgroup size 8), PIPELINE_CACHE, TEXTURE/BUFFER/STORAGE binding arrays + non-uniform + PARTIALLY_BOUND, SHADER_INT64 + all INT64 atomics, SHADER_F16/F64, EXPERIMENTAL_MESH_SHADER(+POINTS), EXPERIMENTAL_RAY_QUERY, PASSTHROUGH_SHADERS, MAPPABLE_PRIMARY_BUFFERS, MULTIVIEW, PIPELINE_STATISTICS_QUERY.
  - Limits: max_buffer_size 2 GiB-1, max_storage_buffer_binding_size 128 MiB, max_bind_groups 8, max_immediate_size 256, max_binding_array_elements 1,000,000, compute workgroup storage 32 KiB, mesh output 256 verts/256 prims.
  - => lavapipe exposes almost the whole GPU-driven feature set, so culling/MDI-count/Hi-Z/indirect compute paths can be integration-tested headless. It is slow (CPU), so use tiny worlds; timings are meaningless for perf.
- CI: wgpu's own CI installs Mesa (lavapipe) on Linux/Windows and WARP + DXC on Windows (.github/workflows/ci.yml lines ~696-739, actions install-mesa/install-warp) [V]. Recipe: Linux GH runner `sudo apt-get install -y mesa-vulkan-drivers libvulkan1`, set `WGPU_BACKEND=vulkan`, optionally `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` [file name U; here it is lvp_icd.json]; request adapter with `force_fallback_adapter` or pick DeviceType::Cpu. Windows: WARP via DX12 (`WGPU_ADAPTER_NAME=Microsoft Basic Render`/ force_fallback) [U]. Mesh shaders on llvmpipe: docs warn naga output may fail -- test before relying.

## Recommendations
1. Stay on wgpu 30 (bump to 30.0.1). 2. Core path: packed-quad storage arena + offset-allocator + compute cull -> multi_draw_indexed_indirect_count (Vulkan/DX12) / multi_draw_indexed_indirect fallback (Metal/GL). 3. Use first_instance->instance_index as chunk id, verify DX12 behaviour; SHADER_DRAW_INDEX as Vulkan-only optional. 4. Defer mesh shaders; prototype later behind feature flag. 5. Far LOD: heightmap/brick DDA compute at reduced resolution. 6. Add lavapipe job to CI for correctness tests; wgpu-profiler 0.28 for GPU timing.

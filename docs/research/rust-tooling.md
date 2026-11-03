# Rust voxel engine tooling research (2026-11-03)

Versions verified via crates.io API today. Claims marked (K) are from background knowledge, not re-verified today.

## Verified versions
| crate | version | updated |
|---|---|---|
| rayon / rayon-core | 1.12.0 / 1.13.0 | 2026-04 |
| crossbeam-channel / -deque / -queue | 0.5.17 / 0.8.8 / 0.3.14 | 2026-09 |
| bevy_tasks | 0.20.1 | 2026-10-28 |
| noiz | 0.6.0 (needs bevy_math 0.20; Lemon pins noiz 0.5 + bevy_math 0.19, outdated) | 2026-10-12 |
| fastnoise-lite | 1.1.1 | 2024-03 |
| noise (noise-rs) | 0.9.0 | 2024-03 |
| simdnoise | 3.1.6 | 2020-05 (unmaintained) |
| libnoise | 1.2.0 | 2025-04 |
| fastnoise2 (bindings) | 0.4.0 | 2026-01 |
| wide | 1.7.1 | 2026-09 |
| glam | 0.34.2 (Lemon 0.33, Decaf 0.32) | 2026-11-01 |
| wgpu | 30.0.1 | 2026-08 |
| winit | 0.30.13 stable; 0.31.0-beta.4 (2026-10-24) | |
| mimalloc / tikv-jemallocator | 0.1.52 / 0.7.0 | |
| ahash / foldhash / rustc-hash / hashbrown | 0.8.12 / 0.2.0 / 2.1.3 / 0.17.1 | |
| tracy-client / tracing-tracy | 0.19.0 / 0.12.0 (Decaf pins =0.18.4) | 2026-08 |
| profiling / puffin / puffin_http | 1.0.18 / 0.20.0 / 0.17.0 | |
| criterion / divan | 0.8.2 / 0.1.21 (no release since 2025-04) | |
| codspeed-divan-compat | 6.0.0 | 2026-10 |
| gungraun (renamed iai-callgrind) | 0.20.0 | 2026-09 |
| cargo-deny | 0.20.2 | 2026-07 |
| bumpalo | 3.20.3 | |

## 1. Job system
- rayon global pool has no priorities; `spawn` is LIFO per worker, `spawn_fifo`/`scope_fifo` give FIFO within a thread (https://docs.rs/rayon-core/latest/rayon_core/fn.spawn_fifo.html, https://rust-lang.github.io/hashbrown/rayon_core/fn.scope_fifo.html). No cancellation primitive.
- Recommended pattern (K): do NOT push every chunk into rayon. Keep a main-thread priority queue (BinaryHeap keyed by distance to camera / frustum bias, re-prioritised every N frames). Each frame, submit at most `workers*2` jobs with `rayon::spawn_fifo` on a dedicated `ThreadPoolBuilder` pool (n-1 or n-2 threads, leaving main + render free; keep it separate from the global pool). Results return via `crossbeam-channel` (or `std::sync::mpsc`, now crossbeam-based) to the main thread.
- Cancellation: per-chunk generation counter (`AtomicU32` in a flat slot array) or `Arc<AtomicBool>`; job checks it at start and between pipeline stages (noise -> mesh); main thread also drops results whose generation is stale. Because only a bounded number are in flight, the queue is re-sortable and stale work is just never submitted.
- bevy_tasks 0.20.1: has priority-aware pools (compute/async-compute/IO) and task handles that cancel on drop, built on async-executor. Heavier dependency and futures overhead; only worth it if adopting Bevy ECS. Not recommended for a standalone engine.
- Custom work-stealing (crossbeam-deque 0.8.8) only if rayon proves a bottleneck; measure first, per CLAUDE.md.
- Recommendation: rayon 1.12 dedicated pool + main-thread priority heap + generation-counter cancellation + crossbeam-channel 0.5.17 for results.

## 2. Noise
Local micro-benchmark: <scratch>/noise-bench (divan 0.1.21, `cargo bench`, lto=fat, 4 cores, default target-cpu i.e. baseline x86-64, noisy VM; single-threaded; 4 octaves fBm, freq 0.02, median).

| grid | crate / algorithm | median | per sample |
|---|---|---|---|
| 2D 64x64 (4096) | simdnoise fbm simplex (SIMD, runtime AVX2 dispatch) | 57 us | 14 ns |
| | fastnoise-lite Perlin | 181 us | 44 ns |
| | noiz 0.6 Perlin (MixCellGradients/Smoothstep/QuickGradients) | 191 us | 47 ns |
| | noise-rs 0.9 Perlin f64 | 228 us | 56 ns |
| 3D 32^3 (32768) | simdnoise fbm simplex | 0.73 ms (min 0.60) | 22 ns |
| | fastnoise-lite Perlin | 2.51 ms | 77 ns |
| | noiz Perlin | 3.00 ms | 92 ns |
| | noise-rs Perlin f64 | 3.95 ms | 121 ns |
| | fastnoise-lite OpenSimplex2 | 4.07 ms | 124 ns |

Caveats: simdnoise is simplex (different algorithm) and the only truly SIMD one here; others are scalar. noiz does per-sample scalar work; its strengths are API flexibility, derivatives, Bevy ties. Not benchmarked: FastNoise2 (fastnoise2 0.4.0; C++ SIMD via cmake, cmake+clang are installed) — upstream claims several-fold speedups over scalar libs, unverified here; libnoise 1.2 (noiz's own benches/compare/libnoise.rs compares them, see https://elliottjpierce.github.io/noiz/).
Recommendations:
- Best throughput without C++: hand-write a small 2D/3D simplex/value-noise fBm in `wide::f32x8` over a whole chunk column (gen heightmap 32x32 with 8 lanes). Only ~3-4x faster than scalar libs is plausible given simdnoise's result. Avoid simdnoise as a dependency (unmaintained since 2020, uses simdeez) but it demonstrates the ceiling.
- Cheap pragmatic: fastnoise-lite 1.1.1 (no deps, no_std, fastest scalar here) for terrain 2D heightmap (2D is only 1 sample per column, 1024 columns per 32x32 chunk => ~0.2 ms with 4 octaves) — 3D noise only for caves/overhangs, and cache by evaluating at coarse resolution + trilinear interpolation (4x4x4 stride cuts cost 64x; K, standard technique).
- Drop noiz+bevy_math if only used for noise (pulls bevy_reflect etc.; compile 1m52s total in bench). Keep only if derivatives/ cell-based features are needed.
- Or FastNoise2 bindings if C++ toolchain in CI is acceptable; benchmark before adopting.

## 3. SIMD in stable Rust
- `std::simd` (portable_simd) still nightly-only: tracking issue https://github.com/rust-lang/rust/issues/86656 — no stabilization PR, "still mostly blocked" on rust-lang/portable-simd#364, needs RFC (fetched 2026-11-03).
- Stable choices: `wide` 1.7.1 (f32x8, u32x8, u64x4 etc., fallbacks), `std::arch` with runtime `is_x86_feature_detected!`, glam (SIMD-backed Vec4/Mat4/Quat, Vec3A via SSE2/NEON/wasm; no wide vectors), `pulp`/`multiversion` for dispatch (K).
- Bitmask greedy meshing does not need SIMD types: u64 column masks (64 = 62 + 2 padding bits) use `trailing_zeros`, `leading_zeros`, `!`, `& >> <<`, `count_ones` which compile to TZCNT/BMI (needs `-C target-cpu` with bmi1, else BSF-based fallback; x86-64-v3 baseline is a good target if you accept AVX2 CPUs, K). Face culling = `col & !(col >> 1)`, `col & !(col << 1)` per axis; auto-vectorisation handles the loops across 64x64 columns. This is the approach popularised by cgerikj/binary-greedy-meshing (https://github.com/cgerikj/binary-greedy-meshing, K) and 0fps greedy meshing.
- Set `.cargo/config.toml` `rustflags = ["-C","target-cpu=x86-64-v3"]` for release only after measuring; keep a baseline build for shipping.

## 4. Memory
- Allocator: `mimalloc` 0.1.52 as `#[global_allocator]` generally wins on multithreaded alloc-heavy workloads; `tikv-jemallocator` 0.7.0 is the alternative (K). Chunk gen/mesh produce many Vec growths: eliminate the cause first with reusable per-thread scratch buffers (thread_local `Vec` with capacity retained) — measure before adding mimalloc.
- Arenas/pools: per-worker mesh scratch (`Vec<u32>` vertices) + `bumpalo` 3.20.3 only for short-lived temp data. For GPU data, suballocate large wgpu buffers (one vertex pool with free-list/buddy allocator, multi-draw-indirect) rather than per-chunk buffers.
- Voxel storage: palette-compressed or flat `[u16; 32768]` boxes; keep chunk padding for meshing.
- Hashing: foldhash 0.2.0 is now hashbrown's/std-adjacent default-quality fast hasher (README: https://github.com/orlp/foldhash, compares favorably to ahash, faster than SipHash; `foldhash::fast::FixedState` for deterministic). rustc-hash 2.1.3 (FxHasher v2, improved mixing) is fastest for small integer keys but has weaker distribution for IVec3 packed keys unless mixed. ahash 0.8.12 is stagnant (last release 2025-05). Recommend foldhash (or rustc-hash 2) with a packed i64/u64 key rather than IVec3.
- Best: skip hashing entirely for the loaded region. Use a flat ring buffer / clipmap: slot index = `(x mod W) + W*((z mod W) + W*(y mod H))` with a stored chunk coordinate for validation. O(1) neighbour lookup (needed for meshing borders), no hashing, cache-friendly, trivial eviction on camera movement. Use a hashmap only for the sparse far/unloaded/persisted set.

## 5. Profiling / benchmarking
- Tracy: `tracy-client` 0.19.0 (Decaf pins =0.18.4 — must match the Tracy GUI version; check the tracy-client README table) plus `tracing-tracy` 0.12.0 if using tracing. Best frame-timeline/zone tool, GPU zones via tracy-client GPU context (K). Use feature-gated: `profile-tracy = ["tracy-client/enable"]` so release is zero-cost.
- `profiling` 1.0.18 is a thin facade over tracy/puffin/optick/superluminal: recommended to keep instrumentation backend-agnostic (`profiling::scope!`, `finish_frame!`). `puffin` 0.20.0 + `puffin_http` is the pure-Rust fallback, no external GUI install issues.
- Microbench: criterion 0.8.2 (maintained, stats-heavy) vs divan 0.1.21 (nicer API, allocation counters via `AllocProfiler`, generics; no release since 2025-04 but CodSpeed ships `codspeed-divan-compat` 6.0.0 which is actively updated: https://codspeed.io/docs/benchmarks/rust/divan). Pick divan for ergonomics + CodSpeed, criterion if you want Bencher/criterion JSON.
- `cargo flamegraph` (K) / `samply` for sampling; set `[profile.profiling] inherits="release", debug=true` (and `-C force-frame-pointers=yes`).
- CI regression tracking: wall-clock on shared GitHub runners is noisy (we saw min-to-max 2x spreads locally). Options: CodSpeed (instruction-count simulation via valgrind, works with divan/criterion; https://codspeed.io/docs/benchmarks/rust/divan), Bencher (statistical thresholds, criterion adapter: https://bencher.dev/learn/track-in-ci/rust/criterion/), or `gungraun` 0.20.0 (ex iai-callgrind; instruction counts, deterministic, run locally in CI with valgrind). For deterministic results: fixed seeds, fixed world coords, single thread for microbenches, `black_box`, pinned runner (`runs-on` same image), compare against base branch on same job.
- End-to-end: add a headless `--bench-scenario` mode that flies a recorded camera path for N frames and prints JSON (frame ms p50/p99, chunks/s, VRAM) matching the CLAUDE.md commit "Stats" section.

## 6. Project hygiene
- Workspace (resolver = "3", edition 2024): `crates/{core (math, coords, ids), world (voxel storage, gen, streaming), mesh (greedy meshing, no wgpu dep), render (wgpu 30, WGSL), app (winit, main)}` + `benches/` crate. Keep mesh/world free of wgpu so benchmarks compile fast. Use `[workspace.dependencies]` for version pinning, `[workspace.lints]`.
- Release profile (K):
```
[profile.release]
lto = "fat"          # or "thin"; measure compile-time/perf
codegen-units = 1
panic = "abort"
debug = "line-tables-only"
[profile.dev]
opt-level = 1
[profile.dev.package."*"]
opt-level = 3
```
Note panic=abort prevents rayon panic isolation; acceptable for an engine. Only use `opt-level=3` for deps + `opt-level=1` for own crates in dev; avoids unplayable debug builds.
- `rust-toolchain.toml`: `channel = "1.97.0"`, `components = ["rustfmt","clippy"]` (pin exact for reproducible benchmarks).
- Clippy: `[workspace.lints.clippy] all={level="warn",priority=-1}, pedantic warn with selective allows, unwrap_used/expect_used = "warn"`, `cargo clippy --workspace --all-targets -- -D warnings` in CI. `rustfmt.toml`: `edition="2024"`, `max_width=100`.
- `cargo-deny` 0.20.2: `deny.toml` for advisories, licenses (MIT/Apache-2.0/BSD/Zlib), duplicate versions (bans). Note duplicate glam versions across bevy_math/glam directly — align glam to 0.34 if noiz is dropped.
- GitHub Actions: jobs fmt, clippy, test, deny (`EmbarkStudios/cargo-deny-action`), bench (CodSpeed or Bencher); `Swatinem/rust-cache`; `dtolnay/rust-toolchain` or rely on rust-toolchain.toml; headless GPU tests via lavapipe (mesa-vulkan-drivers) (K).

## 7. Windowing
- winit 0.30.13 is latest stable; 0.31.0-beta.4 (2026-10-24) exists — https://github.com/rust-windowing/winit/releases. Recommend 0.30.13 now (Lemon already uses 0.30; Decaf's 0.29 uses the old closure loop). Migrate to 0.31 once stable.
- ApplicationHandler model (0.30): `impl ApplicationHandler for App { fn resumed(&mut self, &ActiveEventLoop) (create window + wgpu surface), fn window_event(&mut self, &ActiveEventLoop, WindowId, WindowEvent), fn device_event (raw mouse motion for camera), fn about_to_wait (request_redraw) }`, `EventLoop::run_app(&mut app)`; set `ControlFlow::Poll` for game loops. User events through `EventLoopProxy<T>`.
- 0.31 beta changes (from release notes): surface creation separated from resume (use `can_create_surfaces`), `user_event` -> `user_wake_up` (no payload; use a channel), `run_app` takes handler by value, `dyn ActiveEventLoop`. Isolate winit in the `app` crate so migration touches one file.

## Final stack recommendation
rayon 1.12 (dedicated pool, spawn_fifo, main-thread priority heap, generation-counter cancel) + crossbeam-channel 0.5.17; fastnoise-lite 1.1.1 (or hand-written `wide` 1.7.1 SIMD noise / FastNoise2 after measurement); u64 bitmask greedy meshing; flat ring-buffer chunk slots + foldhash 0.2 for sparse maps; mimalloc 0.1.52 only if measured; profiling 1.0.18 + tracy-client 0.19.0; divan + codspeed or gungraun in CI; winit 0.30.13; glam 0.34.2; wgpu 30.0.1.

## Sources
- https://github.com/rust-lang/rust/issues/86656
- https://github.com/rust-windowing/winit/releases
- https://crates.io/api/v1/crates/<name> (all versions)
- https://docs.rs/rayon-core/latest/rayon_core/fn.spawn_fifo.html
- https://rust-lang.github.io/hashbrown/rayon_core/fn.scope_fifo.html
- https://github.com/orlp/foldhash
- https://codspeed.io/docs/benchmarks/rust/divan
- https://bencher.dev/learn/track-in-ci/rust/criterion/
- https://elliottjpierce.github.io/noiz/ and noiz-0.6.0 README (version table)

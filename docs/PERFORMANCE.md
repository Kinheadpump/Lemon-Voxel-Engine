# Performance log

## Protocol

- Release builds only (`cargo bench-cpu`, `cargo run --release`). Debug numbers are never recorded.
- Record date, commit, machine (CPU, cores, GPU, driver) and the exact command.
- CPU micro-benches: divan medians. Shared VMs are noisy (2× min–max spread seen), so compare against a baseline from the same machine and session.
- GPU numbers only from real hardware. Lavapipe proves correctness, not speed.
- Budgets to hold as features land: frame ≤ 4 ms CPU and ≤ 8 ms GPU at radius 32 on a mid-range GPU; no frame spike above 16 ms while streaming at sprint speed.

## Baselines

### 2026-11-03 — first slice (cloud VM, Intel Xeon @ 2.30 GHz, 4 vCPUs, no GPU; lavapipe)

`cargo bench-cpu` (single thread):

| Bench | Median |
|---|---|
| Binary greedy mesh, terrain surface chunk (640 quads) | 33–39 µs |
| Binary greedy mesh, 3D checkerboard worst case (98,304 quads) | 0.77–1.05 ms |
| Naive culled reference mesh, surface chunk | 211–241 µs |
| Padded gather (centre + 6 neighbour slabs) | 29–35 µs |
| Terrain generate, surface chunk | 140–149 µs |
| Palette decode, surface chunk | 17–24 µs |

`vx --screenshot out.png --radius R` (rayon, 4 threads, region = (2R+3)² × 5 chunks):

| Radius | Chunks generated | Gen | Voxel heap | Meshes / quads | Quad buffer | Mesh |
|---|---|---|---|---|---|---|
| 12 | 3,645 | 94 ms | 9.5 MiB | 1,611 / 565,859 | 4.3 MiB | 40 ms |
| 16 | 6,125 | 212 ms | 16.1 MiB | 2,826 / 986,380 | 7.5 MiB | 62 ms |
| 32 | 22,445 | 594 ms | 58.6 MiB | 10,858 / 3,792,695 | 28.9 MiB | 241 ms |

### Prior attempts, same VM class (from the audits)

| | Decaf | Lemon | This repo |
|---|---|---|---|
| Mesh one surface chunk | 1.22 ms | 155–224 µs (+1.4 ms border prediction) | ~35 µs (+~32 µs gather) |
| Generate one surface chunk | 4.95 ms | 297–350 µs | ~145 µs |
| Voxel memory per non-uniform chunk | 128 KiB | 64 KiB | 4–16 KiB typical |
| Voxel memory per air chunk | 128 KiB | 64 KiB | 0 |

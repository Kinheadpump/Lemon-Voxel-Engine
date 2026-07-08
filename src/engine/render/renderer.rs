use std::sync::{Arc, Mutex};

use wgpu::util::{BufferInitDescriptor, DeviceExt, DrawIndirectArgs};

use crate::engine::config::EngineConfig;
use crate::engine::core::mesher::DirectionalMesh;
use crate::game::math::cascades::MAX_SHADOW_CASCADES;

use super::cull_pipeline::{self, ChunkMetaGpu, CullUniformData};
use super::hzb::HzbPass;
use super::pipeline::{self, ChunkPipeline, DIRECTION_VECTORS};
use super::shadow::{ShadowDrawData, ShadowPass};

const FACE_STRIDE_BYTES: u64 = 4;
/// Groesse von `ChunkData` (WGSL: `vec4<f32>`) in Bytes.
const CHUNK_DATA_STRIDE_BYTES: u64 = 16;
const INDIRECT_ARGS_STRIDE_BYTES: u64 = std::mem::size_of::<DrawIndirectArgs>() as u64;
const CHUNK_META_STRIDE_BYTES: u64 = std::mem::size_of::<ChunkMetaGpu>() as u64;
/// 6 Richtungen * u32-Atomic-Counter.
const COUNTERS_BUFFER_SIZE: u64 = 6 * 4;

/// Handle auf die persistent hochgeladene Geometrie eines Chunks. Pro Richtung ein
/// (first_instance, face_count)-Paar im jeweiligen Richtungs-Buffer.
#[derive(Clone, Copy, Default)]
pub struct ChunkGpuHandle {
    slots: [Slot; 6],
}

#[derive(Clone, Copy, Default)]
struct Slot {
    offset: u32,
    count: u32,
}

/// First-Fit Free-List-Suballocator ueber einen linearen Instanz-Raum. Vergibt und recycelt
/// zusammenhaengende Regionen; benachbarte freie Bloecke werden beim Freigeben verschmolzen.
struct SubAllocator {
    free: Vec<(u32, u32)>,
}

impl SubAllocator {
    fn new(capacity: u32) -> Self {
        Self { free: vec![(0, capacity)] }
    }

    fn alloc(&mut self, size: u32) -> Option<u32> {
        if size == 0 {
            return Some(0);
        }
        for i in 0..self.free.len() {
            let (offset, block) = self.free[i];
            if block >= size {
                if block == size {
                    self.free.remove(i);
                } else {
                    self.free[i] = (offset + size, block - size);
                }
                return Some(offset);
            }
        }
        None
    }

    fn free_region(&mut self, offset: u32, size: u32) {
        if size == 0 {
            return;
        }
        let insert = self.free.partition_point(|&(o, _)| o < offset);
        self.free.insert(insert, (offset, size));

        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(self.free.len());
        for &(offset, size) in &self.free {
            if let Some(last) = merged.last_mut() {
                if last.0 + last.1 == offset {
                    last.1 += size;
                    continue;
                }
            }
            merged.push((offset, size));
        }
        self.free = merged;
    }
}

struct DirectionArena {
    faces_buffer: wgpu::Buffer,
    /// Opaque-Pass-Bindgroup - `binding 3` adressiert einen Sub-Bereich des von `ChunkRenderer`
    /// gehaltenen `combined_chunk_data_buffer` (s. dortigen Kommentar): der Cull-Compute-Pass
    /// schreibt Indirect-Argumente/Chunk-Origin GPU-seitig kompaktiert direkt hinein, die CPU
    /// sieht diesen Buffer nie.
    bind_group: wgpu::BindGroup,
    /// Bind-Group gegen das Shadow-Pipeline-Layout - adressiert dieselben `faces_buffer` wie oben
    /// (die Geometrie selbst ist fuer Opaque- und Schatten-Pass identisch), aber EIGENE
    /// `shadow_chunk_data_buffer`/`shadow_indirect_buffer` unten: die Schatten-sichtbare Chunk-Menge
    /// (Licht-Kugel-Kullung, siehe `ChunkManager::update_shadow_visibility`) unterscheidet sich von
    /// der kamera-sichtbaren und bleibt (anders als der Opaque-Pass) CPU-seitig kompaktiert.
    shadow_bind_group: wgpu::BindGroup,
    shadow_chunk_data_buffer: wgpu::Buffer,
    shadow_indirect_buffer: wgpu::Buffer,
    allocator: SubAllocator,
    shadow_indirect_scratch: Vec<DrawIndirectArgs>,
    shadow_chunk_data_scratch: Vec<[f32; 4]>,
    shadow_draw_count: u32,
}

struct StatsSlot {
    buffer: wgpu::Buffer,
    ready: Arc<Mutex<bool>>,
    busy: bool,
}

const STATS_READBACK_SLOTS: usize = 2;

pub struct ChunkRenderer {
    camera_buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    directions: [DirectionArena; 6],
    wireframe_enabled: bool,
    max_draws_per_direction: usize,
    chunk_pool_size: usize,

    /// Ein Eintrag pro Chunk-Pool-Slot (Index = `pool_slot` aus `ChunkManager`) - AABB + Face-Slots
    /// aller 6 Richtungen. Input des Cull-Compute-Passes.
    chunk_meta_buffer: wgpu::Buffer,
    /// Kompaktierte Indirect-Draw-Argumente ALLER 6 Richtungen in einem Buffer (Richtung `d`
    /// belegt den Bereich `[d*max_draws_per_direction .. (d+1)*max_draws_per_direction)`) - so
    /// kann der Cull-Shader mit nur EINER Storage-Buffer-Bindung pro Ressource auskommen, statt
    /// 6 einzelne Bindings gegen das Storage-Buffer-Limit zu verbrauchen.
    combined_indirect_buffer: wgpu::Buffer,
    /// Analog zu `combined_indirect_buffer`, aber fuer die Chunk-Origin-Daten (`ChunkData` in
    /// `shader.wgsl`) - vom Cull-Shader an genau demselben kompaktierten Index geschrieben.
    combined_chunk_data_buffer: wgpu::Buffer,
    /// Byte-Abstand zwischen den Richtungs-Bereichen in `combined_chunk_data_buffer` - anders als
    /// beim Indirect-Buffer auf `min_storage_buffer_offset_alignment` aufgerundet, weil die
    /// Opaque-Pass-Bindgroup pro Richtung einen Sub-Bereich dieses Buffers per Offset bindet
    /// (`create_direction_arena`), und Storage-Buffer-Bind-Offsets diese Geraete-Grenze respektieren
    /// muessen. In Elementen (je `CHUNK_DATA_STRIDE_BYTES`) an den Cull-Shader durchgereicht.
    chunk_data_dir_stride_elems: u32,
    /// 6 Atomic-Counter (einer pro Richtung) - Basis fuer `multi_draw_indirect_count` UND
    /// gleichzeitig Schreib-Index-Quelle im Cull-Shader (`atomicAdd`).
    counters_buffer: wgpu::Buffer,

    cull_compute_pipeline: wgpu::ComputePipeline,
    cull_bind_group_layout: wgpu::BindGroupLayout,
    cull_uniform_buffer: wgpu::Buffer,
    /// `None` bis zum ersten `rebuild_cull_bind_group` (braucht die HZB-Textur, die erst NACH dem
    /// Renderer erzeugt wird, s. `GpuContext::new`).
    cull_bind_group: Option<wgpu::BindGroup>,

    /// Asynchroner, nicht-blockierender Readback der 6 Draw-Counter fuers HUD - identisches Muster
    /// zu `gpu_timer.rs` (zwei alternierende Puffer, `map_async` + Poll statt Stall).
    stats_slots: [StatsSlot; STATS_READBACK_SLOTS],
    stats_frame_parity: usize,
    last_draw_count: u32,
}

impl ChunkRenderer {
    pub fn new(
        device: &wgpu::Device,
        pipeline: &ChunkPipeline,
        initial_view_proj: glam::Mat4,
        config: &EngineConfig,
        shadow_pass: &ShadowPass,
    ) -> Self {
        let camera_data = pipeline::CameraUniformData {
            view_proj: initial_view_proj.to_cols_array_2d(),
            debug_mode: [0, 0, 0, 0],
            camera_pos: [0.0; 4],
            camera_forward: [0.0, 0.0, 1.0, 0.0],
        };
        let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("camera_uniform_buffer"),
            contents: bytemuck::bytes_of(&camera_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let lighting_data = pipeline::LightingUniformData {
            cascade_view_proj: [glam::Mat4::IDENTITY.to_cols_array_2d(); MAX_SHADOW_CASCADES],
            cascade_split_far: [f32::MAX; 4],
            sun_direction: [0.0, 1.0, 0.0, 0.0],
            sun_color_intensity: [1.0, 1.0, 1.0, 1.0],
            ambient_count_resolution: [0.2, 0.0, 0.0, 0.0],
        };
        let lighting_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("lighting_uniform_buffer"),
            contents: bytemuck::bytes_of(&lighting_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let chunk_pool_size = config.chunk_pool_size;
        let max_draws_per_direction = config.max_draws_per_direction;

        let chunk_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_meta_buffer"),
            size: chunk_pool_size as u64 * CHUNK_META_STRIDE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let combined_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_combined_indirect_buffer"),
            size: 6 * max_draws_per_direction as u64 * INDIRECT_ARGS_STRIDE_BYTES,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Storage-Buffer-Bind-Offsets muessen `min_storage_buffer_offset_alignment` respektieren
        // (typischerweise 256 Byte) - der reine `max_draws_per_direction * CHUNK_DATA_STRIDE_BYTES`
        // Abstand landet dort i.A. NICHT auf einer gueltigen Grenze, deshalb wird pro Richtung
        // aufgerundet.
        let storage_alignment = device.limits().min_storage_buffer_offset_alignment as u64;
        let chunk_data_dir_unpadded_bytes = max_draws_per_direction as u64 * CHUNK_DATA_STRIDE_BYTES;
        let chunk_data_dir_stride_bytes =
            chunk_data_dir_unpadded_bytes.div_ceil(storage_alignment) * storage_alignment;
        let chunk_data_dir_stride_elems = (chunk_data_dir_stride_bytes / CHUNK_DATA_STRIDE_BYTES) as u32;

        let combined_chunk_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_combined_chunk_data_buffer"),
            size: 6 * chunk_data_dir_stride_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_cull_counters_buffer"),
            size: COUNTERS_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cull_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("chunk_cull_uniform_buffer"),
            contents: bytemuck::bytes_of(&CullUniformData {
                view_proj: initial_view_proj.to_cols_array_2d(),
                screen_size: [1.0, 1.0, 0.0, 0.0],
                counts: [max_draws_per_direction as u32, chunk_pool_size as u32, 1, chunk_data_dir_stride_elems],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (cull_compute_pipeline, cull_bind_group_layout) = cull_pipeline::create(device);

        let directions = std::array::from_fn(|dir| {
            Self::create_direction_arena(
                device,
                pipeline,
                &camera_buffer,
                &lighting_buffer,
                shadow_pass,
                dir,
                config,
                &combined_chunk_data_buffer,
                chunk_data_dir_stride_bytes,
            )
        });

        let stats_slots = std::array::from_fn(|_| StatsSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk_cull_stats_readback_buffer"),
                size: COUNTERS_BUFFER_SIZE,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            ready: Arc::new(Mutex::new(false)),
            busy: false,
        });

        Self {
            camera_buffer,
            lighting_buffer,
            directions,
            wireframe_enabled: false,
            max_draws_per_direction,
            chunk_pool_size,
            chunk_meta_buffer,
            combined_indirect_buffer,
            combined_chunk_data_buffer,
            chunk_data_dir_stride_elems,
            counters_buffer,
            cull_compute_pipeline,
            cull_bind_group_layout,
            cull_uniform_buffer,
            cull_bind_group: None,
            stats_slots,
            stats_frame_parity: 0,
            last_draw_count: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_direction_arena(
        device: &wgpu::Device,
        pipeline: &ChunkPipeline,
        camera_buffer: &wgpu::Buffer,
        lighting_buffer: &wgpu::Buffer,
        shadow_pass: &ShadowPass,
        dir: usize,
        config: &EngineConfig,
        combined_chunk_data_buffer: &wgpu::Buffer,
        chunk_data_dir_stride_bytes: u64,
    ) -> DirectionArena {
        let direction_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("direction_uniform_buffer"),
            contents: bytemuck::bytes_of(&DIRECTION_VECTORS[dir]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let faces_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_faces_persistent"),
            size: config.max_faces_per_direction as u64 * FACE_STRIDE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_chunk_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_data_per_shadow_draw"),
            size: config.max_draws_per_direction as u64 * CHUNK_DATA_STRIDE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_shadow_indirect_batch"),
            size: (config.max_draws_per_direction * std::mem::size_of::<DrawIndirectArgs>()) as u64,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dir_chunk_data_size = config.max_draws_per_direction as u64 * CHUNK_DATA_STRIDE_BYTES;
        let dir_chunk_data_offset = dir as u64 * chunk_data_dir_stride_bytes;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk_direction_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: direction_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: faces_buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: combined_chunk_data_buffer,
                        offset: dir_chunk_data_offset,
                        size: wgpu::BufferSize::new(dir_chunk_data_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&pipeline.block_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&pipeline.block_texture_sampler),
                },
                wgpu::BindGroupEntry { binding: 6, resource: lighting_buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&shadow_pass.sampling_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&shadow_pass.comparison_sampler),
                },
            ],
        });

        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk_direction_shadow_bind_group"),
            layout: &shadow_pass.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: shadow_pass.direction_buffer(dir).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: faces_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: shadow_chunk_data_buffer.as_entire_binding() },
            ],
        });

        DirectionArena {
            faces_buffer,
            bind_group,
            shadow_bind_group,
            shadow_chunk_data_buffer,
            shadow_indirect_buffer,
            allocator: SubAllocator::new(config.max_faces_per_direction as u32),
            shadow_indirect_scratch: Vec::with_capacity(config.max_draws_per_direction),
            shadow_chunk_data_scratch: Vec::with_capacity(config.max_draws_per_direction),
            shadow_draw_count: 0,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, view_proj: glam::Mat4, camera_pos: glam::Vec3, camera_forward: glam::Vec3) {
        let camera_data = pipeline::CameraUniformData {
            view_proj: view_proj.to_cols_array_2d(),
            debug_mode: [self.wireframe_enabled as u32, 0, 0, 0],
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
            camera_forward: [camera_forward.x, camera_forward.y, camera_forward.z, 0.0],
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_data));
    }

    /// Aktualisiert Sonnen-/Kaskaden-Daten fuer den Fragment-Shader. `cascades` ist immer
    /// `MAX_SHADOW_CASCADES` lang; `cascade_count` sagt dem Shader, wie viele davon tatsaechlich
    /// befuellt sind (ungenutzte Slots behalten `split_far = f32::MAX` und werden nie getroffen).
    #[allow(clippy::too_many_arguments)]
    pub fn update_lighting(
        &self,
        queue: &wgpu::Queue,
        cascades: &[crate::game::math::cascades::Cascade; MAX_SHADOW_CASCADES],
        cascade_count: u32,
        shadow_map_resolution: u32,
        sun_direction_to_sun: glam::Vec3,
        sun_color: glam::Vec3,
        sun_intensity: f32,
        ambient: f32,
    ) {
        let lighting_data = pipeline::LightingUniformData {
            cascade_view_proj: std::array::from_fn(|i| cascades[i].view_proj.to_cols_array_2d()),
            cascade_split_far: std::array::from_fn(|i| cascades[i].split_far),
            sun_direction: [sun_direction_to_sun.x, sun_direction_to_sun.y, sun_direction_to_sun.z, 0.0],
            sun_color_intensity: [sun_color.x, sun_color.y, sun_color.z, sun_intensity],
            ambient_count_resolution: [ambient, cascade_count as f32, shadow_map_resolution as f32, 0.0],
        };
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&lighting_data));
    }

    /// Liefert pro Richtung die fuer den Schatten-Pass sichtbare Geometrie (Licht-Kugel-Kullung
    /// gegen ALLE geladenen Chunks, s. `ChunkManager::update_shadow_visibility` - NICHT die
    /// Kamera-Sichtbarkeitsmenge). Frueher wurde hier dieselbe Menge wie fuer den Opaque-Pass
    /// zurueckgegeben: da Kamera-Frustum-Sichtbarkeit sich bei reiner Kopfdrehung staendig aendert,
    /// aber die Schatten-Kaskaden eine davon unabhaengige, staendig um die Kamera liegende
    /// Kugel abdecken, poppte Schatten-Geometrie bei jeder Drehung rein/raus - Schatten "sprangen"
    /// oder tauchten an Stellen auf, die eben noch nicht beschattet waren.
    pub fn shadow_draw_data(&self) -> [ShadowDrawData<'_>; 6] {
        std::array::from_fn(|dir| ShadowDrawData {
            bind_group: &self.directions[dir].shadow_bind_group,
            indirect_buffer: &self.directions[dir].shadow_indirect_buffer,
            draw_count: self.directions[dir].shadow_draw_count,
        })
    }

    pub fn toggle_wireframe(&mut self) {
        self.wireframe_enabled = !self.wireframe_enabled;
    }

    /// Laedt die Geometrie eines Chunks EINMALIG persistent hoch und liefert ein Handle zurueck.
    /// Reicht der Buffer einer Richtung nicht, wird diese Richtung ausgelassen (count = 0).
    pub fn alloc_chunk(&mut self, queue: &wgpu::Queue, mesh: &DirectionalMesh) -> ChunkGpuHandle {
        let mut handle = ChunkGpuHandle::default();

        for (dir, arena) in self.directions.iter_mut().enumerate() {
            let faces = &mesh.faces[dir];
            if faces.is_empty() {
                continue;
            }
            let count = faces.len() as u32;
            let Some(offset) = arena.allocator.alloc(count) else {
                log::warn!("Face-Buffer Richtung {dir} voll - Chunk-Teil ausgelassen");
                continue;
            };

            queue.write_buffer(
                &arena.faces_buffer,
                offset as u64 * FACE_STRIDE_BYTES,
                bytemuck::cast_slice(faces),
            );

            handle.slots[dir] = Slot { offset, count };
        }

        handle
    }

    pub fn free_chunk(&mut self, handle: &ChunkGpuHandle) {
        for (dir, arena) in self.directions.iter_mut().enumerate() {
            let slot = handle.slots[dir];
            arena.allocator.free_region(slot.offset, slot.count);
        }
    }

    /// Schreibt/aktualisiert den Cull-Metadaten-Eintrag eines Chunks - Index ist der `pool_slot`
    /// aus `ChunkManager`, NICHT ein kompaktierter Zaehler (Alloc/Free adressieren so direkt per
    /// Index statt eine GPU-seitige Liste pflegen zu muessen).
    pub fn update_chunk_meta(&self, queue: &wgpu::Queue, pool_slot: usize, aabb_min: glam::Vec3, aabb_max: glam::Vec3, handle: &ChunkGpuHandle) {
        let meta = ChunkMetaGpu {
            aabb_min: [aabb_min.x, aabb_min.y, aabb_min.z, 1.0],
            aabb_max: [aabb_max.x, aabb_max.y, aabb_max.z, 0.0],
            slots: std::array::from_fn(|dir| [handle.slots[dir].offset, handle.slots[dir].count]),
        };
        queue.write_buffer(&self.chunk_meta_buffer, pool_slot as u64 * CHUNK_META_STRIDE_BYTES, bytemuck::bytes_of(&meta));
    }

    /// Markiert einen Pool-Slot als unbelegt - der Cull-Shader ueberspringt ihn dann komplett
    /// (`aabb_min.w < 0.5`), statt mit stehengebliebenen (potenziell freigegebenen) Face-Slots zu
    /// testen.
    pub fn clear_chunk_meta(&self, queue: &wgpu::Queue, pool_slot: usize) {
        queue.write_buffer(&self.chunk_meta_buffer, pool_slot as u64 * CHUNK_META_STRIDE_BYTES, bytemuck::bytes_of(&ChunkMetaGpu::INACTIVE));
    }

    /// Baut den Indirect-Draw-Batch fuer die schatten-sichtbaren Chunks (Licht-Kugel-Kullung, s.
    /// `shadow_draw_data`-Kommentar) neu auf - CPU-kompaktiert, da hierfuer (anders als der
    /// Opaque-Pass) noch keine GPU-Kullung existiert.
    pub fn set_shadow_visible(&mut self, queue: &wgpu::Queue, visible: &[(ChunkGpuHandle, glam::Vec3)]) {
        for (dir, arena) in self.directions.iter_mut().enumerate() {
            arena.shadow_indirect_scratch.clear();
            arena.shadow_chunk_data_scratch.clear();
            for (handle, origin) in visible {
                let slot = handle.slots[dir];
                if slot.count == 0 || arena.shadow_indirect_scratch.len() >= self.max_draws_per_direction {
                    continue;
                }
                arena.shadow_indirect_scratch.push(DrawIndirectArgs {
                    vertex_count: 6,
                    instance_count: slot.count,
                    first_vertex: 0,
                    first_instance: slot.offset,
                });
                arena.shadow_chunk_data_scratch.push([origin.x, origin.y, origin.z, 0.0]);
            }

            arena.shadow_draw_count = arena.shadow_indirect_scratch.len() as u32;
            if arena.shadow_draw_count > 0 {
                let indirect_bytes: &[u8] = bytemuck::cast_slice(&arena.shadow_indirect_scratch);
                queue.write_buffer(&arena.shadow_indirect_buffer, 0, indirect_bytes);

                let chunk_data_bytes: &[u8] = bytemuck::cast_slice(&arena.shadow_chunk_data_scratch);
                queue.write_buffer(&arena.shadow_chunk_data_buffer, 0, chunk_data_bytes);
            }
        }
    }

    fn build_cull_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        chunk_meta_buffer: &wgpu::Buffer,
        cull_uniform_buffer: &wgpu::Buffer,
        hzb_view: &wgpu::TextureView,
        combined_indirect_buffer: &wgpu::Buffer,
        combined_chunk_data_buffer: &wgpu::Buffer,
        counters_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk_cull_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: chunk_meta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cull_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(hzb_view) },
                wgpu::BindGroupEntry { binding: 3, resource: combined_indirect_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: combined_chunk_data_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: counters_buffer.as_entire_binding() },
            ],
        })
    }

    /// Muss einmalig nach der Erstellung UND nach jedem `HzbPass::resize` aufgerufen werden - die
    /// Bind-Group haelt sonst eine veraltete (bei Resize neu erzeugte) HZB-Textur-View fest.
    pub fn rebuild_cull_bind_group(&mut self, device: &wgpu::Device, hzb: &HzbPass) {
        self.cull_bind_group = Some(Self::build_cull_bind_group(
            device,
            &self.cull_bind_group_layout,
            &self.chunk_meta_buffer,
            &self.cull_uniform_buffer,
            hzb.sampled_view(),
            &self.combined_indirect_buffer,
            &self.combined_chunk_data_buffer,
            &self.counters_buffer,
        ));
    }

    /// GPU-Driven Frustum+Occlusion-Culling: 1 Thread pro Chunk-Pool-Slot (s. `cull.wgsl`).
    /// Ersetzt die vormals CPU-seitige `par_iter`-Frustum-Kullung des Opaque-Passes vollstaendig -
    /// die kompaktierten Indirect-Argumente landen direkt in `combined_indirect_buffer`, ohne dass
    /// die CPU die sichtbare Menge je zu Gesicht bekommt.
    pub fn dispatch_cull(&mut self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, view_proj: glam::Mat4, hzb: &HzbPass) {
        let Some(bind_group) = &self.cull_bind_group else {
            return;
        };

        let (width, height) = hzb.mip0_size();
        let uniform = CullUniformData {
            view_proj: view_proj.to_cols_array_2d(),
            screen_size: [width as f32, height as f32, 0.0, 0.0],
            counts: [
                self.max_draws_per_direction as u32,
                self.chunk_pool_size as u32,
                hzb.mip_count(),
                self.chunk_data_dir_stride_elems,
            ],
        };
        queue.write_buffer(&self.cull_uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        // Vorherige Frame-Zaehler muessen vor dem Kompaktieren auf 0 - sonst wuerden Indices immer
        // weiter aufaddiert statt pro Frame neu ab 0 zu zaehlen.
        encoder.clear_buffer(&self.counters_buffer, 0, None);

        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chunk_cull_pass"), timestamp_writes: None });
        pass.set_pipeline(&self.cull_compute_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups((self.chunk_pool_size as u32).div_ceil(64), 1, 1);
    }

    /// Stoesst den (nicht-blockierenden) Readback der 6 Draw-Counter fuers HUD an - muss NACH
    /// `dispatch_cull`, aber vor `queue.submit`, in denselben Encoder aufgenommen werden.
    pub fn record_stats_copy(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let slot = &mut self.stats_slots[self.stats_frame_parity];
        if !slot.busy {
            encoder.copy_buffer_to_buffer(&self.counters_buffer, 0, &slot.buffer, 0, COUNTERS_BUFFER_SIZE);
        }
    }

    /// Muss nach `queue.submit` aufgerufen werden - identisches Muster zu `GpuTimer::after_submit`.
    pub fn after_submit(&mut self, device: &wgpu::Device) {
        let slot = &mut self.stats_slots[self.stats_frame_parity];
        if !slot.busy {
            slot.busy = true;
            let ready = Arc::clone(&slot.ready);
            slot.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    *ready.lock().unwrap() = true;
                }
            });
        }

        device.poll(wgpu::PollType::Poll).ok();

        for slot in &mut self.stats_slots {
            let is_ready = *slot.ready.lock().unwrap();
            if !is_ready {
                continue;
            }

            let counters: [u32; 6] = {
                let view = slot.buffer.slice(..).get_mapped_range().expect("Buffer nicht gemappt");
                let mut counters = [0u32; 6];
                counters.copy_from_slice(bytemuck::cast_slice(&view));
                counters
            };
            slot.buffer.unmap();
            *slot.ready.lock().unwrap() = false;
            slot.busy = false;

            self.last_draw_count = counters.iter().sum();
        }

        self.stats_frame_parity = (self.stats_frame_parity + 1) % STATS_READBACK_SLOTS;
    }

    pub fn render<'pass>(&'pass self, render_pass: &mut wgpu::RenderPass<'pass>) {
        for (dir, arena) in self.directions.iter().enumerate() {
            render_pass.set_bind_group(0, &arena.bind_group, &[]);
            render_pass.multi_draw_indirect_count(
                &self.combined_indirect_buffer,
                dir as u64 * self.max_draws_per_direction as u64 * INDIRECT_ARGS_STRIDE_BYTES,
                &self.counters_buffer,
                dir as u64 * 4,
                self.max_draws_per_direction as u32,
            );
        }
    }

    /// GPU-kompaktierte Draw-Anzahl (nach Frustum+Occlusion-Culling), ein bis zwei Frames im
    /// Rueckstand (asynchroner Readback, s. `after_submit`) - fuers HUD.
    pub fn draw_call_count(&self) -> u32 {
        self.last_draw_count
    }
}

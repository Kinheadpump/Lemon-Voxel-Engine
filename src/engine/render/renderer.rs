use wgpu::util::{BufferInitDescriptor, DeviceExt, DrawIndirectArgs};

use crate::engine::config::EngineConfig;
use crate::engine::core::mesher::DirectionalMesh;
use crate::game::math::cascades::MAX_SHADOW_CASCADES;

use super::pipeline::{self, ChunkPipeline, DIRECTION_VECTORS};
use super::shadow::{ShadowDrawData, ShadowPass};

const FACE_STRIDE_BYTES: u64 = 4;
/// Groesse von `ChunkData` (WGSL: `vec4<f32>`) in Bytes.
const CHUNK_DATA_STRIDE_BYTES: u64 = 16;

/// Handle auf die persistent hochgeladene Geometrie eines Chunks. Pro Richtung ein
/// (first_instance, face_count)-Paar im jeweiligen Richtungs-Buffer.
#[derive(Clone, Copy, Default)]
pub struct ChunkGpuHandle {
    slots: [Slot; 6],
    origin: [f32; 3],
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
    chunk_data_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Bind-Group gegen das Shadow-Pipeline-Layout - adressiert dieselben `faces_buffer` wie oben
    /// (die Geometrie selbst ist fuer Opaque- und Schatten-Pass identisch), aber EIGENE
    /// `shadow_chunk_data_buffer`/`shadow_indirect_buffer` unten: die Schatten-sichtbare Chunk-Menge
    /// (Licht-Kugel-Kullung, siehe `ChunkManager::update_shadow_visibility`) unterscheidet sich von
    /// der kamera-sichtbaren und muss deshalb ein eigenes Indirect-Batch pflegen.
    shadow_bind_group: wgpu::BindGroup,
    shadow_chunk_data_buffer: wgpu::Buffer,
    shadow_indirect_buffer: wgpu::Buffer,
    allocator: SubAllocator,
    indirect_scratch: Vec<DrawIndirectArgs>,
    chunk_data_scratch: Vec<[f32; 4]>,
    draw_count: u32,
    shadow_indirect_scratch: Vec<DrawIndirectArgs>,
    shadow_chunk_data_scratch: Vec<[f32; 4]>,
    shadow_draw_count: u32,
}

pub struct ChunkRenderer {
    camera_buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    directions: [DirectionArena; 6],
    visible_face_count: usize,
    wireframe_enabled: bool,
    max_draws_per_direction: usize,
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

        let directions = std::array::from_fn(|dir| {
            Self::create_direction_arena(device, pipeline, &camera_buffer, &lighting_buffer, shadow_pass, dir, config)
        });

        Self {
            camera_buffer,
            lighting_buffer,
            directions,
            visible_face_count: 0,
            wireframe_enabled: false,
            max_draws_per_direction: config.max_draws_per_direction,
        }
    }

    fn create_direction_arena(
        device: &wgpu::Device,
        pipeline: &ChunkPipeline,
        camera_buffer: &wgpu::Buffer,
        lighting_buffer: &wgpu::Buffer,
        shadow_pass: &ShadowPass,
        dir: usize,
        config: &EngineConfig,
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

        let chunk_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_data_per_draw"),
            size: config.max_draws_per_direction as u64 * CHUNK_DATA_STRIDE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_indirect_batch"),
            size: (config.max_draws_per_direction * std::mem::size_of::<DrawIndirectArgs>()) as u64,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk_direction_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: direction_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: faces_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: chunk_data_buffer.as_entire_binding() },
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
            chunk_data_buffer,
            indirect_buffer,
            bind_group,
            shadow_bind_group,
            shadow_chunk_data_buffer,
            shadow_indirect_buffer,
            allocator: SubAllocator::new(config.max_faces_per_direction as u32),
            indirect_scratch: Vec::with_capacity(config.max_draws_per_direction),
            chunk_data_scratch: Vec::with_capacity(config.max_draws_per_direction),
            draw_count: 0,
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
    pub fn alloc_chunk(
        &mut self,
        queue: &wgpu::Queue,
        mesh: &DirectionalMesh,
        origin: glam::Vec3,
    ) -> ChunkGpuHandle {
        let mut handle = ChunkGpuHandle { slots: Default::default(), origin: [origin.x, origin.y, origin.z] };

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

    /// Baut den Indirect-Draw-Batch fuer die aktuell sichtbaren Chunks neu auf. Nur diese kleine
    /// Argument-Liste und die dazugehoerigen Chunk-Origins (ein `ChunkData`-Eintrag pro Draw, per
    /// `@builtin(draw_index)` im Shader adressiert) werden pro Frame hochgeladen - die Geometrie
    /// selbst bleibt persistent.
    pub fn set_visible(&mut self, queue: &wgpu::Queue, visible: &[ChunkGpuHandle]) {
        self.visible_face_count = 0;

        for (dir, arena) in self.directions.iter_mut().enumerate() {
            arena.indirect_scratch.clear();
            arena.chunk_data_scratch.clear();
            for handle in visible {
                let slot = handle.slots[dir];
                if slot.count == 0 || arena.indirect_scratch.len() >= self.max_draws_per_direction {
                    continue;
                }
                arena.indirect_scratch.push(DrawIndirectArgs {
                    vertex_count: 6,
                    instance_count: slot.count,
                    first_vertex: 0,
                    first_instance: slot.offset,
                });
                arena.chunk_data_scratch.push([handle.origin[0], handle.origin[1], handle.origin[2], 0.0]);
                self.visible_face_count += slot.count as usize;
            }

            arena.draw_count = arena.indirect_scratch.len() as u32;
            if arena.draw_count > 0 {
                let indirect_bytes: &[u8] = bytemuck::cast_slice(&arena.indirect_scratch);
                queue.write_buffer(&arena.indirect_buffer, 0, indirect_bytes);

                let chunk_data_bytes: &[u8] = bytemuck::cast_slice(&arena.chunk_data_scratch);
                queue.write_buffer(&arena.chunk_data_buffer, 0, chunk_data_bytes);
            }
        }
    }

    /// Baut den Indirect-Draw-Batch fuer die schatten-sichtbaren Chunks (Licht-Kugel-Kullung, s.
    /// `shadow_draw_data`-Kommentar) neu auf - vollstaendig analog zu `set_visible`, aber in ein
    /// eigenes Buffer-Paar pro Richtung, damit Opaque- und Schatten-Pass im selben Frame
    /// unterschiedliche Chunk-Mengen zeichnen koennen.
    pub fn set_shadow_visible(&mut self, queue: &wgpu::Queue, visible: &[ChunkGpuHandle]) {
        for (dir, arena) in self.directions.iter_mut().enumerate() {
            arena.shadow_indirect_scratch.clear();
            arena.shadow_chunk_data_scratch.clear();
            for handle in visible {
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
                arena.shadow_chunk_data_scratch.push([handle.origin[0], handle.origin[1], handle.origin[2], 0.0]);
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

    pub fn render<'pass>(&'pass self, render_pass: &mut wgpu::RenderPass<'pass>) {
        for arena in &self.directions {
            if arena.draw_count == 0 {
                continue;
            }
            render_pass.set_bind_group(0, &arena.bind_group, &[]);
            render_pass.multi_draw_indirect(&arena.indirect_buffer, 0, arena.draw_count);
        }
    }

    pub fn total_face_count(&self) -> usize {
        self.visible_face_count
    }

    pub fn draw_call_count(&self) -> usize {
        self.directions.iter().filter(|d| d.draw_count > 0).count()
    }
}

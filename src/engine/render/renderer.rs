use wgpu::util::{BufferInitDescriptor, DeviceExt, DrawIndirectArgs};

use crate::engine::core::mesher::DirectionalMesh;

use super::pipeline::{self, ChunkPipeline, DIRECTION_VECTORS};

/// Kapazitaet des persistenten Face-Buffers pro Richtung (in Faces/Instanzen). Grosszuegig fuer
/// grosse Render-Distanzen; eine fehlgeschlagene Allokation ueberspringt den Chunk graceful.
pub const MAX_FACES_PER_DIR: usize = 3_000_000;

/// Obergrenze gleichzeitig sichtbarer Chunks (Groesse des Indirect-Batch pro Richtung).
pub const MAX_DRAWS_PER_DIR: usize = 4300;

const FACE_STRIDE_BYTES: u64 = 4;
const ORIGIN_STRIDE_BYTES: u64 = 8;

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
    origins_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    allocator: SubAllocator,
    indirect_scratch: Vec<DrawIndirectArgs>,
    draw_count: u32,
}

pub struct ChunkRenderer {
    camera_buffer: wgpu::Buffer,
    directions: [DirectionArena; 6],
    origin_scratch: Vec<[f32; 2]>,
    visible_face_count: usize,
}

impl ChunkRenderer {
    pub fn new(device: &wgpu::Device, pipeline: &ChunkPipeline, initial_view_proj: glam::Mat4) -> Self {
        let camera_data =
            pipeline::CameraUniformData { view_proj: initial_view_proj.to_cols_array_2d() };
        let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("camera_uniform_buffer"),
            contents: bytemuck::bytes_of(&camera_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let directions = std::array::from_fn(|dir| {
            Self::create_direction_arena(device, pipeline, &camera_buffer, dir)
        });

        Self { camera_buffer, directions, origin_scratch: Vec::new(), visible_face_count: 0 }
    }

    fn create_direction_arena(
        device: &wgpu::Device,
        pipeline: &ChunkPipeline,
        camera_buffer: &wgpu::Buffer,
        dir: usize,
    ) -> DirectionArena {
        let direction_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("direction_uniform_buffer"),
            contents: bytemuck::bytes_of(&DIRECTION_VECTORS[dir]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let faces_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_faces_persistent"),
            size: MAX_FACES_PER_DIR as u64 * FACE_STRIDE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let origins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_origins_persistent"),
            size: MAX_FACES_PER_DIR as u64 * ORIGIN_STRIDE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_indirect_batch"),
            size: (MAX_DRAWS_PER_DIR * std::mem::size_of::<DrawIndirectArgs>()) as u64,
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
                wgpu::BindGroupEntry { binding: 3, resource: origins_buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&pipeline.block_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&pipeline.block_texture_sampler),
                },
            ],
        });

        DirectionArena {
            faces_buffer,
            origins_buffer,
            indirect_buffer,
            bind_group,
            allocator: SubAllocator::new(MAX_FACES_PER_DIR as u32),
            indirect_scratch: Vec::with_capacity(MAX_DRAWS_PER_DIR),
            draw_count: 0,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, view_proj: glam::Mat4) {
        let camera_data = pipeline::CameraUniformData { view_proj: view_proj.to_cols_array_2d() };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_data));
    }

    /// Laedt die Geometrie eines Chunks EINMALIG persistent hoch und liefert ein Handle zurueck.
    /// Reicht der Buffer einer Richtung nicht, wird diese Richtung ausgelassen (count = 0).
    pub fn alloc_chunk(
        &mut self,
        queue: &wgpu::Queue,
        mesh: &DirectionalMesh,
        origin: glam::Vec3,
    ) -> ChunkGpuHandle {
        let origin_xz = [origin.x, origin.z];
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

            self.origin_scratch.clear();
            self.origin_scratch.resize(faces.len(), origin_xz);
            queue.write_buffer(
                &arena.origins_buffer,
                offset as u64 * ORIGIN_STRIDE_BYTES,
                bytemuck::cast_slice(&self.origin_scratch),
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
    /// Argument-Liste wird pro Frame hochgeladen - die Geometrie bleibt persistent.
    pub fn set_visible(&mut self, queue: &wgpu::Queue, visible: &[ChunkGpuHandle]) {
        self.visible_face_count = 0;

        for (dir, arena) in self.directions.iter_mut().enumerate() {
            arena.indirect_scratch.clear();
            for handle in visible {
                let slot = handle.slots[dir];
                if slot.count == 0 || arena.indirect_scratch.len() >= MAX_DRAWS_PER_DIR {
                    continue;
                }
                arena.indirect_scratch.push(DrawIndirectArgs {
                    vertex_count: 6,
                    instance_count: slot.count,
                    first_vertex: 0,
                    first_instance: slot.offset,
                });
                self.visible_face_count += slot.count as usize;
            }

            arena.draw_count = arena.indirect_scratch.len() as u32;
            if arena.draw_count > 0 {
                let bytes: &[u8] = bytemuck::cast_slice(&arena.indirect_scratch);
                queue.write_buffer(&arena.indirect_buffer, 0, bytes);
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

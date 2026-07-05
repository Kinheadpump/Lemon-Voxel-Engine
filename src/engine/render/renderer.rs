use wgpu::util::{BufferInitDescriptor, DeviceExt, DrawIndirectArgs};

use crate::engine::core::mesher::DirectionalMesh;

use super::pipeline::{self, ChunkPipeline, DIRECTION_VECTORS};

pub const GPU_RENDER_SLOTS: usize = 128;
pub const MAX_FACES_PER_SLOT: usize = 8192;

const FACE_STRIDE_BYTES: u64 = 4;
const ORIGIN_STRIDE_BYTES: u64 = 16;
const INDIRECT_ARGS_STRIDE: u64 = 16;

struct DirectionArena {
    faces_buffer: wgpu::Buffer,
    origins_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    slot_bind_groups: Vec<wgpu::BindGroup>,
}

pub struct ChunkRenderer {
    camera_buffer: wgpu::Buffer,
    directions: [DirectionArena; 6],
}

fn slot_face_offset(slot: usize) -> u64 {
    (slot * MAX_FACES_PER_SLOT) as u64
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

        Self { camera_buffer, directions }
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
            label: Some("chunk_faces_arena"),
            size: (GPU_RENDER_SLOTS * MAX_FACES_PER_SLOT) as u64 * FACE_STRIDE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let origins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_origins_arena"),
            size: (GPU_RENDER_SLOTS * MAX_FACES_PER_SLOT) as u64 * ORIGIN_STRIDE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let initial_indirect_args: Vec<DrawIndirectArgs> = (0..GPU_RENDER_SLOTS)
            .map(|_| DrawIndirectArgs { vertex_count: 6, instance_count: 0, first_vertex: 0, first_instance: 0 })
            .collect();

        let indirect_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("chunk_indirect_buffer"),
            contents: bytemuck::cast_slice(&initial_indirect_args),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        });

        let slot_bind_groups = (0..GPU_RENDER_SLOTS)
            .map(|slot| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("chunk_slot_bind_group"),
                    layout: &pipeline.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: direction_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &faces_buffer,
                                offset: slot_face_offset(slot) * FACE_STRIDE_BYTES,
                                size: wgpu::BufferSize::new(
                                    MAX_FACES_PER_SLOT as u64 * FACE_STRIDE_BYTES,
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &origins_buffer,
                                offset: slot_face_offset(slot) * ORIGIN_STRIDE_BYTES,
                                size: wgpu::BufferSize::new(
                                    MAX_FACES_PER_SLOT as u64 * ORIGIN_STRIDE_BYTES,
                                ),
                            }),
                        },
                    ],
                })
            })
            .collect();

        DirectionArena { faces_buffer, origins_buffer, indirect_buffer, slot_bind_groups }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, view_proj: glam::Mat4) {
        let camera_data = pipeline::CameraUniformData { view_proj: view_proj.to_cols_array_2d() };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_data));
    }

    pub fn upload_chunk(
        &self,
        queue: &wgpu::Queue,
        slot: usize,
        mesh: &DirectionalMesh,
        origin: glam::Vec3,
    ) {
        let origin_data = [origin.x, origin.y, origin.z, 0.0f32];

        for (dir, arena) in self.directions.iter().enumerate() {
            let faces = &mesh.faces[dir];
            let count = faces.len().min(MAX_FACES_PER_SLOT);

            let face_byte_offset = slot_face_offset(slot) * FACE_STRIDE_BYTES;
            queue.write_buffer(&arena.faces_buffer, face_byte_offset, bytemuck::cast_slice(&faces[..count]));

            let origins: Vec<[f32; 4]> = std::iter::repeat(origin_data).take(count).collect();
            let origin_byte_offset = slot_face_offset(slot) * ORIGIN_STRIDE_BYTES;
            queue.write_buffer(&arena.origins_buffer, origin_byte_offset, bytemuck::cast_slice(&origins));

            let args =
                DrawIndirectArgs { vertex_count: 6, instance_count: count as u32, first_vertex: 0, first_instance: 0 };
            let indirect_byte_offset = slot as u64 * INDIRECT_ARGS_STRIDE;
            queue.write_buffer(&arena.indirect_buffer, indirect_byte_offset, args.as_bytes());
        }
    }

    pub fn clear_slot(&self, queue: &wgpu::Queue, slot: usize) {
        for arena in &self.directions {
            let args =
                DrawIndirectArgs { vertex_count: 6, instance_count: 0, first_vertex: 0, first_instance: 0 };
            let indirect_byte_offset = slot as u64 * INDIRECT_ARGS_STRIDE;
            queue.write_buffer(&arena.indirect_buffer, indirect_byte_offset, args.as_bytes());
        }
    }

    pub fn render<'pass>(&'pass self, render_pass: &mut wgpu::RenderPass<'pass>) {
        for arena in &self.directions {
            for slot in 0..GPU_RENDER_SLOTS {
                render_pass.set_bind_group(0, &arena.slot_bind_groups[slot], &[]);
                render_pass.draw_indirect(&arena.indirect_buffer, slot as u64 * INDIRECT_ARGS_STRIDE);
            }
        }
    }
}

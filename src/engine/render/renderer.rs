use wgpu::util::{BufferInitDescriptor, DeviceExt, DrawIndirectArgs};

use crate::engine::core::mesher::DirectionalMesh;

use super::pipeline::{self, ChunkPipeline, DIRECTION_VECTORS};

/// Obergrenze der insgesamt (ueber alle sichtbaren Chunks kombiniert) darstellbaren Faces pro
/// Richtung. Grosszuegig bemessen fuer Render Distance 4-6 mit vollstaendig sichtbarem Himmel.
pub const MAX_COMBINED_FACES: usize = 200_000;

const FACE_STRIDE_BYTES: u64 = 4;
const ORIGIN_STRIDE_BYTES: u64 = 16;

struct DirectionArena {
    faces_buffer: wgpu::Buffer,
    origins_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub struct ChunkRenderer {
    camera_buffer: wgpu::Buffer,
    directions: [DirectionArena; 6],
    face_scratch: [Vec<u32>; 6],
    origin_scratch: [Vec<[f32; 4]>; 6],
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

        Self {
            camera_buffer,
            directions,
            face_scratch: std::array::from_fn(|_| Vec::new()),
            origin_scratch: std::array::from_fn(|_| Vec::new()),
        }
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
            label: Some("chunk_faces_frame_buffer"),
            size: MAX_COMBINED_FACES as u64 * FACE_STRIDE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let origins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_origins_frame_buffer"),
            size: MAX_COMBINED_FACES as u64 * ORIGIN_STRIDE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("chunk_indirect_buffer"),
            contents: DrawIndirectArgs { vertex_count: 6, instance_count: 0, first_vertex: 0, first_instance: 0 }
                .as_bytes(),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
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

        DirectionArena { faces_buffer, origins_buffer, indirect_buffer, bind_group }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, view_proj: glam::Mat4) {
        let camera_data = pipeline::CameraUniformData { view_proj: view_proj.to_cols_array_2d() };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_data));
    }

    /// Kompaktiert alle sichtbaren Chunk-Meshes pro Richtung in EINEN zusammenhaengenden
    /// Frame-Buffer und aktualisiert die Indirect-Args entsprechend. Ergebnis: genau ein
    /// draw_indirect-Aufruf pro Richtung (6 insgesamt) fuer die komplette sichtbare Welt,
    /// unabhaengig von der Anzahl geladener/sichtbarer Chunks.
    pub fn upload_frame<'a>(
        &mut self,
        queue: &wgpu::Queue,
        visible_chunks: impl Iterator<Item = (&'a DirectionalMesh, glam::Vec3)>,
    ) {
        for faces in &mut self.face_scratch {
            faces.clear();
        }
        for origins in &mut self.origin_scratch {
            origins.clear();
        }

        for (mesh, origin) in visible_chunks {
            let origin_data = [origin.x, origin.y, origin.z, 0.0f32];
            for dir in 0..6 {
                let faces = &mesh.faces[dir];
                let remaining = MAX_COMBINED_FACES.saturating_sub(self.face_scratch[dir].len());
                let count = faces.len().min(remaining);

                self.face_scratch[dir].extend_from_slice(&faces[..count]);
                self.origin_scratch[dir].extend(std::iter::repeat_n(origin_data, count));
            }
        }

        for (dir, arena) in self.directions.iter().enumerate() {
            let faces = &self.face_scratch[dir];
            let origins = &self.origin_scratch[dir];

            if !faces.is_empty() {
                queue.write_buffer(&arena.faces_buffer, 0, bytemuck::cast_slice(faces));
                queue.write_buffer(&arena.origins_buffer, 0, bytemuck::cast_slice(origins));
            }

            let args = DrawIndirectArgs {
                vertex_count: 6,
                instance_count: faces.len() as u32,
                first_vertex: 0,
                first_instance: 0,
            };
            queue.write_buffer(&arena.indirect_buffer, 0, args.as_bytes());
        }
    }

    pub fn render<'pass>(&'pass self, render_pass: &mut wgpu::RenderPass<'pass>) {
        for arena in &self.directions {
            render_pass.set_bind_group(0, &arena.bind_group, &[]);
            render_pass.draw_indirect(&arena.indirect_buffer, 0);
        }
    }
}

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use vx_core::ChunkPos;
use vx_mesh::{ChunkMesh, PackedQuad};
use wgpu::util::DeviceExt;

use crate::{CameraUniform, TERRAIN_SHADER};

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const VERTICES_PER_QUAD: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

pub struct ChunkMeshRef<'a> {
    pub pos: ChunkPos,
    pub mesh: &'a ChunkMesh,
}

struct UploadedGeometry {
    bind_group: wgpu::BindGroup,
    indirect: wgpu::Buffer,
    draw_count: u32,
    quad_count: u64,
}

pub struct TerrainRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    geometry: Option<UploadedGeometry>,
}

impl TerrainRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain_bind_group_layout"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_uniform"),
            size: size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline: create_pipeline(device, &bind_group_layout, color_format),
            bind_group_layout,
            camera_buffer,
            geometry: None,
        }
    }

    pub fn upload_static<'a>(
        &mut self,
        device: &wgpu::Device,
        chunks: impl IntoIterator<Item = ChunkMeshRef<'a>>,
    ) {
        let mut quads: Vec<PackedQuad> = Vec::new();
        let mut origins: Vec<[i32; 4]> = Vec::new();
        let mut draws: Vec<DrawIndirectArgs> = Vec::new();
        for chunk in chunks {
            if chunk.mesh.quads.is_empty() {
                continue;
            }
            let slot = origins.len() as u32;
            let base = quads.len() as u32;
            origins.push(chunk.pos.min_block().extend(0).to_array());
            quads.extend_from_slice(&chunk.mesh.quads);
            for range in &chunk.mesh.direction_ranges {
                push_direction_draw(&mut draws, base, range, slot);
            }
        }
        if draws.is_empty() {
            self.geometry = None;
            return;
        }
        let quad_buffer = storage_buffer(device, "terrain_quads", bytemuck::cast_slice(&quads));
        let origin_buffer = storage_buffer(device, "chunk_origins", bytemuck::cast_slice(&origins));
        let indirect = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain_indirect"),
            contents: bytemuck::cast_slice(&draws),
            usage: wgpu::BufferUsages::INDIRECT,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: quad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: origin_buffer.as_entire_binding(),
                },
            ],
        });
        self.geometry = Some(UploadedGeometry {
            bind_group,
            indirect,
            draw_count: draws.len() as u32,
            quad_count: quads.len() as u64,
        });
    }

    pub fn quad_count(&self) -> u64 {
        self.geometry
            .as_ref()
            .map_or(0, |geometry| geometry.quad_count)
    }

    pub fn draw_count(&self) -> u32 {
        self.geometry
            .as_ref()
            .map_or(0, |geometry| geometry.draw_count)
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, camera: &CameraUniform) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(geometry) = &self.geometry else {
            return;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &geometry.bind_group, &[]);
        pass.multi_draw_indirect(&geometry.indirect, 0, geometry.draw_count);
    }
}

fn push_direction_draw(
    draws: &mut Vec<DrawIndirectArgs>,
    base: u32,
    range: &Range<u32>,
    slot: u32,
) {
    if range.is_empty() {
        return;
    }
    draws.push(DrawIndirectArgs {
        vertex_count: range.len() as u32 * VERTICES_PER_QUAD,
        instance_count: 1,
        first_vertex: (base + range.start) * VERTICES_PER_QUAD,
        first_instance: slot,
    });
}

fn storage_buffer(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn layout_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    color_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain_shader"),
        source: wgpu::ShaderSource::Wgsl(TERRAIN_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("terrain_pipeline_layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("terrain_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn terrain_shader_parses_and_validates() {
        let module = naga::front::wgsl::parse_str(crate::TERRAIN_SHADER).expect("WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("WGSL validates");
    }
}

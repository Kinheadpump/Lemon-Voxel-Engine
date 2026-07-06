use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::game::math::cascades::{Cascade, MAX_SHADOW_CASCADES};

use super::pipeline::{DEPTH_FORMAT, DIRECTION_VECTORS};

/// Ein Eintrag pro Richtungs-Arena, wie ihn `ChunkRenderer` fuer den Schatten-Pass bereitstellt -
/// dieselbe persistente Geometrie (Faces/Chunk-Origins), die auch der Opaque-Pass zeichnet. Der
/// Schatten-Pass zeichnet aktuell dieselbe kamera-sichtbare Menge wie der Opaque-Pass (keine
/// separate Licht-Frustum-Kullung pro Kaskade) - eine bewusste Vereinfachung fuer dieses
/// Fundament, siehe Kommentar an `ChunkRenderer::shadow_draw_data`.
pub struct ShadowDrawData<'a> {
    pub bind_group: &'a wgpu::BindGroup,
    pub indirect_buffer: &'a wgpu::Buffer,
    pub draw_count: u32,
}

pub struct ShadowPass {
    pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    direction_buffers: [wgpu::Buffer; 6],
    layer_views: [wgpu::TextureView; MAX_SHADOW_CASCADES],
    pub sampling_view: wgpu::TextureView,
    pub comparison_sampler: wgpu::Sampler,
}

impl ShadowPass {
    pub fn new(device: &wgpu::Device, resolution: u32, depth_bias: f32, depth_bias_slope_scale: f32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: std::mem::size_of::<glam::Mat4>() as u32,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Kein Culling: die Richtungs-Meshes enthalten nur nach aussen zeigende Faces, aber
                // aus Sicht der Sonne (statt der Kamera) waere Backface-Culling hier falsch gepolt.
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: depth_bias.round() as i32,
                    slope_scale: depth_bias_slope_scale,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        let direction_buffers = std::array::from_fn(|dir| {
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("shadow_direction_uniform_buffer"),
                contents: bytemuck::bytes_of(&DIRECTION_VECTORS[dir]),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        });

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_cascade_depth_array"),
            size: wgpu::Extent3d {
                width: resolution.max(1),
                height: resolution.max(1),
                depth_or_array_layers: MAX_SHADOW_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let layer_views = std::array::from_fn(|layer| {
            depth_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("shadow_cascade_layer_view"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });

        let sampling_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow_cascade_sampling_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_comparison_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        Self { pipeline, bind_group_layout, direction_buffers, layer_views, sampling_view, comparison_sampler }
    }

    pub fn direction_buffer(&self, dir: usize) -> &wgpu::Buffer {
        &self.direction_buffers[dir]
    }

    /// Zeichnet alle `cascade_count` Kaskaden in ihre jeweilige Depth-Array-Ebene. Die
    /// Licht-View-Projection wird pro Kaskade ueber Immediate Data (statt einer eigenen
    /// Bind-Group/eines Uniform-Buffers pro Kaskade) gesetzt - spart 6*4 Bind-Group-Wechsel pro
    /// Frame gegen eine einzige `set_immediates`-Aufruf pro Kaskade/Richtung.
    pub fn render_cascades(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cascades: &[Cascade; MAX_SHADOW_CASCADES],
        cascade_count: u32,
        draw_data: &[ShadowDrawData; 6],
    ) {
        for (cascade_index, cascade) in cascades.iter().enumerate().take(cascade_count as usize) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_cascade_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.layer_views[cascade_index],
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_immediates(0, bytemuck::bytes_of(&cascade.view_proj.to_cols_array_2d()));

            for direction in draw_data {
                if direction.draw_count == 0 {
                    continue;
                }
                pass.set_bind_group(0, direction.bind_group, &[]);
                pass.multi_draw_indirect(direction.indirect_buffer, 0, direction.draw_count);
            }
        }
    }
}

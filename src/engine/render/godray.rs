use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::game::math::cascades::{Cascade, MAX_SHADOW_CASCADES};
use crate::game::world::godrays::GodrayInstanceData;

use super::shadow::ShadowPass;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GodrayUniformData {
    view_proj: [[f32; 4]; 4],
    cascade_view_proj: [[[f32; 4]; 4]; MAX_SHADOW_CASCADES],
    cascade_split_far: [f32; 4],
    camera_pos: [f32; 4],
    camera_forward: [f32; 4],
    sun_direction_to_sun: [f32; 4],
    sun_color_intensity: [f32; 4],
    /// x = Kaskaden-Anzahl, y = Ray-Anzahl (aktive Instanzen im Compute-Dispatch), z =
    /// Temporal-Blend-Faktor, w = ungenutzt.
    params: [f32; 4],
}

/// Instanced-Billboard-Godrays: ein Compute-Pass bewertet pro Kandidaten-Ray die
/// Schatten/Licht-Kante an seiner Spitze (siehe `godray_compute.wgsl`) und schreibt die Intensity
/// direkt in dasselbe SSBO zurueck (In-Place-Temporal-Blend, keine Ping-Pong-Buffer noetig). Der
/// Render-Pass zeichnet daraus additiv gefaerbte Quads.
pub struct GodrayPass {
    ray_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    capacity: u32,

    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,

    render_pipeline: wgpu::RenderPipeline,
    render_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group: Option<wgpu::BindGroup>,
}

fn instance_buffer_size(capacity: u32) -> u64 {
    capacity as u64 * std::mem::size_of::<GodrayInstanceData>() as u64
}

/// Zwei Render-Shader-Varianten noetig - siehe identisches Muster in `skybox.rs`: die Depth-Textur
/// des Opaque-Passes ist bei aktivem MSAA multisampled, bei deaktiviertem MSAA nicht.
fn render_shader_source(multisampled: bool) -> String {
    let depth_texture_type = if multisampled { "texture_depth_multisampled_2d" } else { "texture_depth_2d" };
    include_str!("godray_render.wgsl").replace("{DEPTH_TEXTURE_TYPE}", depth_texture_type)
}

impl GodrayPass {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        multisampled: bool,
        capacity: u32,
        shadow_pass: &ShadowPass,
    ) -> Self {
        let ray_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("godray_instance_buffer"),
            size: instance_buffer_size(capacity),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("godray_uniform_buffer"),
            contents: bytemuck::bytes_of(&GodrayUniformData {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                cascade_view_proj: [glam::Mat4::IDENTITY.to_cols_array_2d(); MAX_SHADOW_CASCADES],
                cascade_split_far: [f32::MAX; 4],
                camera_pos: [0.0; 4],
                camera_forward: [0.0, 0.0, 1.0, 0.0],
                sun_direction_to_sun: [0.0, 1.0, 0.0, 0.0],
                sun_color_intensity: [1.0, 1.0, 1.0, 1.0],
                params: [0.0, 0.0, 0.12, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (compute_pipeline, compute_bind_group_layout) = Self::create_compute_pipeline(device);
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("godray_compute_bind_group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ray_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shadow_pass.sampling_view),
                },
            ],
        });

        let (render_pipeline, render_bind_group_layout) =
            Self::create_render_pipeline(device, surface_format, multisampled);

        Self {
            ray_buffer,
            uniform_buffer,
            capacity,
            compute_pipeline,
            compute_bind_group,
            render_pipeline,
            render_bind_group_layout,
            render_bind_group: None,
        }
    }

    fn create_compute_pipeline(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("godray_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("godray_compute.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("godray_compute_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("godray_compute_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("godray_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    fn create_render_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        multisampled: bool,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("godray_render_shader"),
            source: wgpu::ShaderSource::Wgsl(render_shader_source(multisampled).into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("godray_render_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("godray_render_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Additive Blend: Lichtschaechte sollen Helligkeit HINZUFUEGEN statt den Hintergrund zu
        // ersetzen/abzudunkeln - src.rgb*src.a wird auf das bereits komponierte Bild addiert.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("godray_render_pipeline"),
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
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            // Kein Hardware-Depth-Attachment (s. Kommentar in godray_render.wgsl) - der manuelle
            // Tiefentest laeuft per Textur-Sample im Fragment-Shader.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    pub fn upload_instances(&self, queue: &wgpu::Queue, instances: &[GodrayInstanceData]) {
        queue.write_buffer(&self.ray_buffer, 0, bytemuck::cast_slice(instances));
    }

    /// Muss nach jeder Aenderung der Haupt-Depth-View (Init, Resize) aufgerufen werden - analog zu
    /// `SkyboxPass::rebuild_bind_group`. Die Compute-Bind-Group (Shadow-Map) ist unabhaengig von der
    /// Fenstergroesse und wird nur einmal in `new` gebaut.
    pub fn rebuild_render_bind_group(&mut self, device: &wgpu::Device, depth_view: &wgpu::TextureView) {
        self.render_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("godray_render_bind_group"),
            layout: &self.render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.ray_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
            ],
        }));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        cascades: &[Cascade; MAX_SHADOW_CASCADES],
        cascade_count: u32,
        camera_pos: glam::Vec3,
        camera_forward: glam::Vec3,
        sun_direction_to_sun: glam::Vec3,
        sun_color: glam::Vec3,
        sun_intensity: f32,
        temporal_blend: f32,
    ) {
        let data = GodrayUniformData {
            view_proj: view_proj.to_cols_array_2d(),
            cascade_view_proj: std::array::from_fn(|i| cascades[i].view_proj.to_cols_array_2d()),
            cascade_split_far: std::array::from_fn(|i| cascades[i].split_far),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
            camera_forward: [camera_forward.x, camera_forward.y, camera_forward.z, 0.0],
            sun_direction_to_sun: [sun_direction_to_sun.x, sun_direction_to_sun.y, sun_direction_to_sun.z, 0.0],
            sun_color_intensity: [sun_color.x, sun_color.y, sun_color.z, sun_intensity],
            params: [cascade_count as f32, self.capacity as f32, temporal_blend, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&data));
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("godray_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, &self.compute_bind_group, &[]);
        pass.dispatch_workgroups(self.capacity.div_ceil(64), 1, 1);
    }

    pub fn render<'pass>(&'pass self, render_pass: &mut wgpu::RenderPass<'pass>) {
        let Some(bind_group) = &self.render_bind_group else {
            return;
        };

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..6, 0..self.capacity);
    }
}

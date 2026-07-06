use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyboxParams {
    inverse_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    direction_to_sun: [f32; 4],
    zenith_day: [f32; 4],
    horizon_day: [f32; 4],
    night: [f32; 4],
}

/// Prozeduraler Himmels-Gradient, gerendert NACH dem Opaque-Pass (siehe `GpuContext::render`):
/// deckt per Tiefen-Discard nur die Pixel ab, die noch auf dem Reverse-Z-Clear-Wert (0.0 =
/// "unendlich fern") stehen, also keine Geometrie hinter sich haben.
pub struct SkyboxPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    zenith_day: [f32; 4],
    horizon_day: [f32; 4],
    night: [f32; 4],
}

/// Zwei Shader-Varianten noetig: die Depth-Textur des Opaque-Passes ist bei aktivem MSAA
/// multisampled, bei deaktiviertem MSAA nicht - WGPU erlaubt keine Laufzeit-Umschaltung des
/// Bindingtyps, daher wird die Textur-Deklaration bei Pipeline-Erstellung textuell eingesetzt.
fn shader_source(multisampled: bool) -> String {
    let depth_texture_type = if multisampled { "texture_depth_multisampled_2d" } else { "texture_depth_2d" };
    include_str!("skybox.wgsl").replace("{DEPTH_TEXTURE_TYPE}", depth_texture_type)
}

impl SkyboxPass {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        multisampled: bool,
        zenith_day: [f32; 3],
        horizon_day: [f32; 3],
        night: [f32; 3],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skybox_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source(multisampled).into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skybox_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skybox_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skybox_pipeline"),
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skybox_uniform_buffer"),
            contents: bytemuck::bytes_of(&SkyboxParams {
                inverse_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [0.0; 4],
                direction_to_sun: [0.0, 1.0, 0.0, 0.0],
                zenith_day: [zenith_day[0], zenith_day[1], zenith_day[2], 0.0],
                horizon_day: [horizon_day[0], horizon_day[1], horizon_day[2], 0.0],
                night: [night[0], night[1], night[2], 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            bind_group: None,
            zenith_day: [zenith_day[0], zenith_day[1], zenith_day[2], 0.0],
            horizon_day: [horizon_day[0], horizon_day[1], horizon_day[2], 0.0],
            night: [night[0], night[1], night[2], 0.0],
        }
    }

    /// Muss nach jeder Aenderung der Depth-View (Init, Resize) aufgerufen werden.
    pub fn rebuild_bind_group(&mut self, device: &wgpu::Device, depth_view: &wgpu::TextureView) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skybox_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 1, resource: self.uniform_buffer.as_entire_binding() },
            ],
        }));
    }

    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_proj: glam::Mat4, camera_pos: glam::Vec3, direction_to_sun: glam::Vec3) {
        let params = SkyboxParams {
            inverse_view_proj: view_proj.inverse().to_cols_array_2d(),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
            direction_to_sun: [direction_to_sun.x, direction_to_sun.y, direction_to_sun.z, 0.0],
            zenith_day: self.zenith_day,
            horizon_day: self.horizon_day,
            night: self.night,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));
    }

    pub fn render<'pass>(&'pass self, render_pass: &mut wgpu::RenderPass<'pass>) {
        let Some(bind_group) = &self.bind_group else {
            return;
        };

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

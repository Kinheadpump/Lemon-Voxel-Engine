use wgpu::util::DeviceExt;

const KERNEL_SIZE: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoParams {
    inverse_projection: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    screen_size_radius_strength: [f32; 4],
    enabled: [u32; 4],
    kernel: [[f32; 4]; KERNEL_SIZE],
}

fn generate_kernel() -> [[f32; 4]; KERNEL_SIZE] {
    let mut state: u32 = 0x9E3779B9;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f64 / u32::MAX as f64) as f32
    };

    std::array::from_fn(|i| {
        let x = next_f32() * 2.0 - 1.0;
        let y = next_f32() * 2.0 - 1.0;
        let z = next_f32();
        let sample = glam::Vec3::new(x, y, z).normalize();

        let linear = (i as f32 + 1.0) / KERNEL_SIZE as f32;
        let scale = 0.1 + linear * linear * 0.9;
        let sample = sample * scale;

        [sample.x, sample.y, sample.z, 0.0]
    })
}

pub struct SsaoPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    kernel: [[f32; 4]; KERNEL_SIZE],
}

impl SsaoPass {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ssao.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("ssao_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao_pipeline"),
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao_color_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao_uniform_buffer"),
            contents: bytemuck::bytes_of(&SsaoParams {
                inverse_projection: glam::Mat4::IDENTITY.to_cols_array_2d(),
                projection: glam::Mat4::IDENTITY.to_cols_array_2d(),
                screen_size_radius_strength: [1.0, 1.0, 1.0, 1.0],
                enabled: [0, 0, 0, 0],
                kernel: generate_kernel(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            bind_group: None,
            kernel: generate_kernel(),
        }
    }

    /// Muss nach jeder Aenderung von Depth- oder Color-View (Init, Resize) aufgerufen werden.
    pub fn rebuild_bind_group(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        color_view: &wgpu::TextureView,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(color_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.uniform_buffer.as_entire_binding() },
            ],
        }));
    }

    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        projection: glam::Mat4,
        screen_width: f32,
        screen_height: f32,
        radius: f32,
        strength: f32,
        enabled: bool,
    ) {
        let params = SsaoParams {
            inverse_projection: projection.inverse().to_cols_array_2d(),
            projection: projection.to_cols_array_2d(),
            screen_size_radius_strength: [screen_width, screen_height, radius, strength],
            enabled: [enabled as u32, 0, 0, 0],
            kernel: self.kernel,
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

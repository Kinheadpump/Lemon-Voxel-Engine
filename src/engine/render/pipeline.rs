#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniformData {
    pub view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DirectionUniformData {
    pub normal: [f32; 4],
    pub u_axis: [f32; 4],
    pub v_axis: [f32; 4],
}

pub const DIRECTION_VECTORS: [DirectionUniformData; 6] = [
    DirectionUniformData { normal: [-1.0, 0.0, 0.0, 0.0], u_axis: [0.0, 0.0, 1.0, 0.0], v_axis: [0.0, 1.0, 0.0, 0.0] },
    DirectionUniformData { normal: [1.0, 0.0, 0.0, 0.0], u_axis: [0.0, 1.0, 0.0, 0.0], v_axis: [0.0, 0.0, 1.0, 0.0] },
    DirectionUniformData { normal: [0.0, -1.0, 0.0, 0.0], u_axis: [1.0, 0.0, 0.0, 0.0], v_axis: [0.0, 0.0, 1.0, 0.0] },
    DirectionUniformData { normal: [0.0, 1.0, 0.0, 0.0], u_axis: [0.0, 0.0, 1.0, 0.0], v_axis: [1.0, 0.0, 0.0, 0.0] },
    DirectionUniformData { normal: [0.0, 0.0, -1.0, 0.0], u_axis: [0.0, 1.0, 0.0, 0.0], v_axis: [1.0, 0.0, 0.0, 0.0] },
    DirectionUniformData { normal: [0.0, 0.0, 1.0, 0.0], u_axis: [1.0, 0.0, 0.0, 0.0], v_axis: [0.0, 1.0, 0.0, 0.0] },
];

pub struct ChunkPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

pub fn create(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> ChunkPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chunk_face_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chunk_face_bind_group_layout"),
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
                    ty: wgpu::BufferBindingType::Uniform,
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
        label: Some("chunk_face_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("chunk_face_pipeline"),
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
            cull_mode: Some(wgpu::Face::Back),
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

    ChunkPipeline { pipeline, bind_group_layout }
}

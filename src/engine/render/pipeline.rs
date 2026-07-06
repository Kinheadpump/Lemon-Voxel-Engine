#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniformData {
    pub view_proj: [[f32; 4]; 4],
    /// x = Debug-Render-Modus (0 = texturiert, 1 = Wireframe/Mesh-Ansicht).
    pub debug_mode: [u32; 4],
    pub camera_pos: [f32; 4],
    /// Fuer die Kaskaden-Auswahl im Fragment-Shader: `dot(camera_forward, world_pos - camera_pos)`
    /// liefert dieselbe Tiefenmetrik, die die CPU-Seite (`compute_cascades`) fuer die
    /// Split-Distanzen nutzt - unabhaengig von der Reverse-Z-NDC-Tiefe der Hauptkamera.
    pub camera_forward: [f32; 4],
}

/// Wird einmal pro Frame aktualisiert; enthaelt alles, was der Fragment-Shader fuer
/// direktionale Beleuchtung + Cascaded-Shadow-Sampling braucht.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightingUniformData {
    pub cascade_view_proj: [[[f32; 4]; 4]; crate::game::math::cascades::MAX_SHADOW_CASCADES],
    /// Kamera-Distanz, bis zu der die jeweilige Kaskade zustaendig ist (siehe `Cascade::split_far`).
    pub cascade_split_far: [f32; 4],
    /// Richtung von einer Oberflaeche ZUR Sonne (fuer `dot(normal, sun_direction)`).
    pub sun_direction: [f32; 4],
    pub sun_color_intensity: [f32; 4],
    /// x = ambient, y = aktive Kaskaden-Anzahl, z = Shadow-Map-Aufloesung (fuer PCF-Texelgroesse).
    pub ambient_count_resolution: [f32; 4],
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

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const BLOCK_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub struct ChunkPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub block_texture_view: wgpu::TextureView,
    pub block_texture_sampler: wgpu::Sampler,
}

fn create_block_texture_array(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::TextureView, wgpu::Sampler) {
    use super::textures::{TEXTURE_LAYER_COUNT, TEXTURE_SIZE, generate_texture_atlas};

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("block_texture_array"),
        size: wgpu::Extent3d {
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
            depth_or_array_layers: TEXTURE_LAYER_COUNT,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: BLOCK_TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &generate_texture_atlas(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(TEXTURE_SIZE * 4),
            rows_per_image: Some(TEXTURE_SIZE),
        },
        wgpu::Extent3d {
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
            depth_or_array_layers: TEXTURE_LAYER_COUNT,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("block_texture_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    (view, sampler)
}

pub fn create(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    surface_format: wgpu::TextureFormat,
    sample_count: u32,
) -> ChunkPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chunk_face_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let (block_texture_view, block_texture_sampler) = create_block_texture_array(device, queue);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chunk_face_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
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
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
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
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState { count: sample_count, mask: !0, alpha_to_coverage_enabled: false },
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

    ChunkPipeline { pipeline, bind_group_layout, block_texture_view, block_texture_sampler }
}

pub fn create_depth_view(device: &wgpu::Device, width: u32, height: u32, sample_count: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Multisampled Color-Ziel fuer den Opaque-Pass (Hardware-MSAA). Wird per `resolve_target` in ein
/// normales, sampelbares Ziel aufgeloest, bevor der Post-Processing-Pass (SSAO) darauf zugreift.
pub fn create_msaa_color_view(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("msaa_color_texture"),
        size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Aufgeloestes (nicht multisampled) Opaque-Farbziel, sampelbar fuer den Post-Processing-Pass.
pub fn create_resolve_color_view(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resolved_color_texture"),
        size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

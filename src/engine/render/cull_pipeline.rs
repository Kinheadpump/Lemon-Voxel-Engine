#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullUniformData {
    pub view_proj: [[f32; 4]; 4],
    /// xy = HZB-Mip0-Aufloesung in Texeln, zw ungenutzt.
    pub screen_size: [f32; 4],
    /// x = max_draws_per_direction (Element-Stride des kombinierten Indirect-Buffers pro Richtung),
    /// y = Anzahl Pool-Slots (= Dispatch-Thread-Grenze), z = HZB-Mip-Anzahl, w = Element-Stride des
    /// kombinierten ChunkData-Buffers pro Richtung (auf `min_storage_buffer_offset_alignment`
    /// aufgerundet, s. `renderer.rs` - deshalb ein EIGENER Stride statt `x`).
    pub counts: [u32; 4],
    /// xyz = aktuelle Kamera-Position, w ungenutzt. Der HZB stammt aus dem VORHERIGEN Frame, der
    /// Occlusion-Test projiziert Chunk-AABBs aber mit der Kamera-Matrix DIESES Frames dagegen (s.
    /// Kommentar an `occlusion_culled` in cull.wgsl) - bei schneller Kamerarotation oder wenn die
    /// Kamera nah an/in Geometrie steckt (z.B. in einer Hoehle), kann der Bildschirm letztes Frame
    /// fast vollstaendig mit naher Tiefe gefuellt gewesen sein, was diesen Frame faelschlich ALLES
    /// als verdeckt einstuft. `camera_pos` erlaubt dem Shader, den Occlusion-Test fuer kamera-nahe
    /// Chunks zu ueberspringen (nur noch Frustum-Test, immer exakt) statt ihm blind zu vertrauen.
    pub camera_pos: [f32; 4],
}

/// Ein Eintrag pro Chunk-Pool-Slot - Index in `chunk_meta_buffer` IST der `pool_slot` aus
/// `ChunkManager`, sodass Alloc/Free direkt per Index statt per Kompaktierung schreiben.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkMetaGpu {
    /// w = 1.0 aktiv / 0.0 unbelegt-oder-Luft-Chunk (ueberspringt Frustum/Occlusion-Test komplett).
    pub aabb_min: [f32; 4],
    /// w = ungenutztes Padding (vec4-Alignment).
    pub aabb_max: [f32; 4],
    /// Pro Richtung (x = Face-Buffer-Offset, y = Face-Count) - identisch zu `ChunkGpuHandle::slots`.
    pub slots: [[u32; 2]; 6],
}

impl ChunkMetaGpu {
    pub const INACTIVE: Self = Self { aabb_min: [0.0; 4], aabb_max: [0.0; 4], slots: [[0, 0]; 6] };
}

pub fn create(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chunk_cull_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("cull.wgsl").into()),
    });

    let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chunk_cull_bind_group_layout"),
        entries: &[
            storage_entry(0, true),
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
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, false),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chunk_cull_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chunk_cull_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bind_group_layout)
}

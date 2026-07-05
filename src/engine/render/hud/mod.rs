mod font;

const PIXEL_SCALE: f32 = 2.0;
const CHAR_ADVANCE: f32 = (font::GLYPH_WIDTH as f32 + 1.0) * PIXEL_SCALE;
const LINE_ADVANCE: f32 = (font::GLYPH_HEIGHT as f32 + 3.0) * PIXEL_SCALE;
const MARGIN: f32 = 8.0;
const MAX_CHARS: usize = 4096;
const VERTICES_PER_CHAR: usize = 6;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HudVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

pub struct HudRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    visible: bool,
    scratch: Vec<HudVertex>,
}

impl HudRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat, visible_by_default: bool) -> Self {
        let (atlas_width, atlas_height) = font::atlas_size();
        let atlas_data = font::generate_font_atlas();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hud_font_atlas"),
            size: wgpu::Extent3d { width: atlas_width, height: atlas_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
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
            &atlas_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_width * 4),
                rows_per_image: Some(atlas_height),
            },
            wgpu::Extent3d { width: atlas_width, height: atlas_height, depth_or_array_layers: 1 },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hud_font_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hud_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hud_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<HudVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(vertex_layout)],
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud_vertex_buffer"),
            size: (MAX_CHARS * VERTICES_PER_CHAR * std::mem::size_of::<HudVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group,
            vertex_buffer,
            vertex_count: 0,
            visible: visible_by_default,
            scratch: Vec::with_capacity(MAX_CHARS * VERTICES_PER_CHAR),
        }
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    pub fn update_text(
        &mut self,
        queue: &wgpu::Queue,
        screen_width: f32,
        screen_height: f32,
        lines: &[String],
    ) {
        self.scratch.clear();

        for (line_index, line) in lines.iter().enumerate() {
            let base_y = MARGIN + line_index as f32 * LINE_ADVANCE;

            for (char_index, ch) in line.chars().enumerate() {
                if self.scratch.len() / VERTICES_PER_CHAR >= MAX_CHARS {
                    break;
                }
                if ch == ' ' {
                    continue;
                }

                let base_x = MARGIN + char_index as f32 * CHAR_ADVANCE;
                push_glyph_quad(&mut self.scratch, base_x, base_y, screen_width, screen_height, ch);
            }
        }

        self.vertex_count = self.scratch.len() as u32;
        if !self.scratch.is_empty() {
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.scratch));
        }
    }

    pub fn render<'pass>(&'pass self, render_pass: &mut wgpu::RenderPass<'pass>) {
        if !self.visible || self.vertex_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_count, 0..1);
    }
}

fn push_glyph_quad(
    out: &mut Vec<HudVertex>,
    base_x: f32,
    base_y: f32,
    screen_width: f32,
    screen_height: f32,
    ch: char,
) {
    let (atlas_width, atlas_height) = font::atlas_size();
    let (cell_col, cell_row) = font::glyph_cell(ch);

    let glyph_w = font::GLYPH_WIDTH as f32 * PIXEL_SCALE;
    let glyph_h = font::GLYPH_HEIGHT as f32 * PIXEL_SCALE;

    let to_ndc = |px: f32, py: f32| -> [f32; 2] {
        [(px / screen_width) * 2.0 - 1.0, 1.0 - (py / screen_height) * 2.0]
    };

    let u0 = (cell_col as usize * font::CELL_WIDTH) as f32 / atlas_width as f32;
    let v0 = (cell_row as usize * font::CELL_HEIGHT) as f32 / atlas_height as f32;
    let u1 = (cell_col as usize * font::CELL_WIDTH + font::GLYPH_WIDTH) as f32 / atlas_width as f32;
    let v1 = (cell_row as usize * font::CELL_HEIGHT + font::GLYPH_HEIGHT) as f32 / atlas_height as f32;

    let top_left = to_ndc(base_x, base_y);
    let top_right = to_ndc(base_x + glyph_w, base_y);
    let bottom_left = to_ndc(base_x, base_y + glyph_h);
    let bottom_right = to_ndc(base_x + glyph_w, base_y + glyph_h);

    let v_top_left = HudVertex { position: top_left, uv: [u0, v0] };
    let v_top_right = HudVertex { position: top_right, uv: [u1, v0] };
    let v_bottom_left = HudVertex { position: bottom_left, uv: [u0, v1] };
    let v_bottom_right = HudVertex { position: bottom_right, uv: [u1, v1] };

    out.push(v_top_left);
    out.push(v_top_right);
    out.push(v_bottom_right);
    out.push(v_top_left);
    out.push(v_bottom_right);
    out.push(v_bottom_left);
}

/// Hierarchical-Z-Buffer: Mip-Pyramide des Haupt-Depth-Buffers fuer GPU-Occlusion-Culling (siehe
/// `cull.wgsl`). Wird VOR dem Cull-Compute-Pass aus dem noch nicht geleerten Depth-Buffer des
/// VORHERIGEN Frames aufgebaut (ein Frame Latenz, Standardtechnik bei GPU-Driven Rendering).
fn mip_count_for(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height).max(1);
    (u32::BITS - max_dim.leading_zeros()).clamp(1, 10)
}

fn level_dims(width: u32, height: u32, level: u32) -> (u32, u32) {
    (1.max(width >> level), 1.max(height >> level))
}

/// Zwei Shader-Varianten noetig - identisches Muster zu `godray.rs`/`skybox.rs`: die Haupt-Depth-
/// Textur ist bei aktivem MSAA multisampled, sonst nicht. Reverse-Z: Minimum ueber alle Samples
/// ist der konservative (am weitesten entfernte) Wert - Mitteln waere fuer die Occlusion-Schranke
/// falsch.
fn copy_shader_source(multisampled: bool, sample_count: u32) -> String {
    let depth_texture_type = if multisampled { "texture_depth_multisampled_2d" } else { "texture_depth_2d" };
    let sample_loop = if multisampled {
        (0..sample_count).map(|s| format!("d = min(d, textureLoad(depth_tex, coord, {s}));")).collect::<Vec<_>>().join("\n")
    } else {
        "d = min(d, textureLoad(depth_tex, coord, 0));".to_string()
    };
    include_str!("hzb_copy.wgsl")
        .replace("{DEPTH_TEXTURE_TYPE}", depth_texture_type)
        .replace("{SAMPLE_LOOP}", &sample_loop)
}

fn create_texture_and_views(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, Vec<wgpu::TextureView>, wgpu::TextureView, u32) {
    let mip_count = mip_count_for(width, height);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hzb_texture"),
        size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        mip_level_count: mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    let mip_views = (0..mip_count)
        .map(|level| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("hzb_mip_view"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        })
        .collect();

    let sampled_view = texture
        .create_view(&wgpu::TextureViewDescriptor { label: Some("hzb_sampled_view"), ..Default::default() });

    (texture, mip_views, sampled_view, mip_count)
}

fn build_bind_groups(
    device: &wgpu::Device,
    copy_layout: &wgpu::BindGroupLayout,
    downsample_layout: &wgpu::BindGroupLayout,
    mip_views: &[wgpu::TextureView],
    depth_view: &wgpu::TextureView,
) -> (wgpu::BindGroup, Vec<wgpu::BindGroup>) {
    let copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hzb_copy_bind_group"),
        layout: copy_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&mip_views[0]) },
        ],
    });

    let downsample_bind_groups = (1..mip_views.len() as u32)
        .map(|level| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hzb_downsample_bind_group"),
                layout: downsample_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&mip_views[(level - 1) as usize]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&mip_views[level as usize]),
                    },
                ],
            })
        })
        .collect();

    (copy_bind_group, downsample_bind_groups)
}

pub struct HzbPass {
    mip_views: Vec<wgpu::TextureView>,
    sampled_view: wgpu::TextureView,
    mip_count: u32,
    width: u32,
    height: u32,

    copy_pipeline: wgpu::ComputePipeline,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    copy_bind_group: wgpu::BindGroup,

    downsample_pipeline: wgpu::ComputePipeline,
    downsample_bind_group_layout: wgpu::BindGroupLayout,
    downsample_bind_groups: Vec<wgpu::BindGroup>,
}

impl HzbPass {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth_view: &wgpu::TextureView,
        multisampled: bool,
        sample_count: u32,
    ) -> Self {
        let (copy_pipeline, copy_bind_group_layout) = Self::create_copy_pipeline(device, multisampled, sample_count);
        let (downsample_pipeline, downsample_bind_group_layout) = Self::create_downsample_pipeline(device);
        let (_texture, mip_views, sampled_view, mip_count) = create_texture_and_views(device, width, height);
        let (copy_bind_group, downsample_bind_groups) =
            build_bind_groups(device, &copy_bind_group_layout, &downsample_bind_group_layout, &mip_views, depth_view);

        Self {
            mip_views,
            sampled_view,
            mip_count,
            width: width.max(1),
            height: height.max(1),
            copy_pipeline,
            copy_bind_group_layout,
            copy_bind_group,
            downsample_pipeline,
            downsample_bind_group_layout,
            downsample_bind_groups,
        }
    }

    /// Muss bei jeder Fenstergroessen-Aenderung aufgerufen werden - die Mip-Pyramide skaliert mit
    /// der Bildschirmaufloesung.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, depth_view: &wgpu::TextureView) {
        let (_texture, mip_views, sampled_view, mip_count) = create_texture_and_views(device, width, height);
        let (copy_bind_group, downsample_bind_groups) = build_bind_groups(
            device,
            &self.copy_bind_group_layout,
            &self.downsample_bind_group_layout,
            &mip_views,
            depth_view,
        );

        self.mip_views = mip_views;
        self.sampled_view = sampled_view;
        self.mip_count = mip_count;
        self.width = width.max(1);
        self.height = height.max(1);
        self.copy_bind_group = copy_bind_group;
        self.downsample_bind_groups = downsample_bind_groups;
    }

    fn create_copy_pipeline(
        device: &wgpu::Device,
        multisampled: bool,
        sample_count: u32,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hzb_copy_shader"),
            source: wgpu::ShaderSource::Wgsl(copy_shader_source(multisampled, sample_count).into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hzb_copy_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hzb_copy_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hzb_copy_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    fn create_downsample_pipeline(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hzb_downsample_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("hzb_downsample.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hzb_downsample_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hzb_downsample_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hzb_downsample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    pub fn generate(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("hzb_generate_pass"), timestamp_writes: None });

        pass.set_pipeline(&self.copy_pipeline);
        pass.set_bind_group(0, &self.copy_bind_group, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);

        pass.set_pipeline(&self.downsample_pipeline);
        for level in 1..self.mip_count {
            let (w, h) = level_dims(self.width, self.height, level);
            pass.set_bind_group(0, &self.downsample_bind_groups[(level - 1) as usize], &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
    }

    pub fn sampled_view(&self) -> &wgpu::TextureView {
        &self.sampled_view
    }

    pub fn mip_count(&self) -> u32 {
        self.mip_count
    }

    pub fn mip0_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

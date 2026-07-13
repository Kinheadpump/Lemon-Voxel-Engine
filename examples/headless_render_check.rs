// Visuelle Regressions-Pruefung ohne Fenster: rendert einen Huegel-Chunk senkrecht von oben durch
// die echte Engine-Pipeline (MSAA + SSAO), liest die Pixel zurueck und prueft, dass die
// Gras-Oberflaeche tatsaechlich gruen erscheint. Faengt invertiertes Culling / Tiefen-Regressionen
// fruehzeitig ab.

use voxel_engine::engine::config::EngineConfig;
use voxel_engine::engine::core::mesher::mesh_chunk;
use voxel_engine::engine::render::blur::SsaoBlurPass;
use voxel_engine::engine::render::hzb::HzbPass;
use voxel_engine::engine::render::pipeline::{
    self, create_ao_view, create_depth_view, create_msaa_color_view, create_resolve_color_view,
};
use voxel_engine::engine::render::renderer::ChunkRenderer;
use voxel_engine::engine::render::shadow::ShadowPass;
use voxel_engine::engine::render::ssao::SsaoPass;
use voxel_engine::game::world::blocks;
use voxel_engine::game::world::chunk::{CHUNK_SIZE, Chunk};
use voxel_engine::game::world::generator::TerrainGenerator;

const SIZE: u32 = 256;
const SAMPLES: u32 = 4;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Sucht deterministisch (Spirale um den Ursprung) den ersten Oberflaechen-Chunk, dessen Spalten
/// mehrheitlich Gras exponieren - fest kodierte Koordinaten brachen bei jedem Terrain-Umbau.
fn find_grassy_chunk(generator: &TerrainGenerator) -> (i32, i32, i32, i32) {
    for radius in 0i32..64 {
        for chunk_z in -radius..=radius {
            for chunk_x in -radius..=radius {
                if chunk_x.abs() != radius && chunk_z.abs() != radius {
                    continue;
                }
                let mut heights = [0i32; (CHUNK_SIZE * CHUNK_SIZE) as usize];
                let mut min_height = i32::MAX;
                let mut max_height = i32::MIN;
                for local_z in 0..CHUNK_SIZE {
                    for local_x in 0..CHUNK_SIZE {
                        let h = generator
                            .height_at(chunk_x * CHUNK_SIZE + local_x, chunk_z * CHUNK_SIZE + local_z);
                        heights[(local_z * CHUNK_SIZE + local_x) as usize] = h;
                        min_height = min_height.min(h);
                        max_height = max_height.max(h);
                    }
                }
                // Flach (Kamera-Distanz zur Oberflaeche bleibt im FOV-Budget, s. `eye`), klar ueber
                // Strand-Hoehe und alle Oberflaechen in EINEM Chunk-Y.
                if min_height < 4 || max_height - min_height > 8 {
                    continue;
                }
                let chunk_y = min_height.div_euclid(CHUNK_SIZE);
                if max_height >= (chunk_y + 1) * CHUNK_SIZE {
                    continue;
                }

                let mut chunk = Chunk::empty();
                generator.generate_chunk(chunk_x, chunk_y, chunk_z, &mut chunk);
                let mut grass_surfaces = 0;
                for local_z in 0..CHUNK_SIZE {
                    for local_x in 0..CHUNK_SIZE {
                        let local_y =
                            heights[(local_z * CHUNK_SIZE + local_x) as usize] - chunk_y * CHUNK_SIZE;
                        if chunk.get_block(local_x, local_y, local_z) == blocks::GRASS {
                            grass_surfaces += 1;
                        }
                    }
                }
                if grass_surfaces * 100 >= (CHUNK_SIZE * CHUNK_SIZE) * 85 {
                    return (chunk_x, chunk_y, chunk_z, max_height);
                }
            }
        }
    }
    panic!("kein grasreicher Chunk in 64 Chunks Umkreis gefunden");
}

fn main() {
    let green_ratio = pollster::block_on(render_top_down_green_ratio());
    println!("Gras-Anteil (Top-Down): {:.1}%", green_ratio * 100.0);
    assert!(green_ratio > 0.5, "Gras-Oberflaeche von oben nicht sichtbar - Culling/Tiefe defekt!");
    println!("OK: Gras-Oberflaeche wird korrekt gerendert.");
}

async fn render_top_down_green_ratio() -> f32 {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        })
        .await
        .expect("kein adapter");
    let required_limits =
        wgpu::Limits { max_immediate_size: std::mem::size_of::<glam::Mat4>() as u32, ..wgpu::Limits::default() };

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("headless"),
            required_features: wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::SHADER_DRAW_INDEX
                | wgpu::Features::IMMEDIATES
                | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
            required_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("kein device");

    let config = EngineConfig::default();
    let generator = TerrainGenerator::new(&config);
    let (chunk_x, chunk_y, chunk_z, surface_height) = find_grassy_chunk(&generator);
    println!("Test-Chunk: ({chunk_x}, {chunk_y}, {chunk_z}), Oberflaeche bei y={surface_height}");

    // Abstand so, dass die 32-Block-Chunk-Flaeche das 60-Grad-FOV fuellt (halbe Sichtweite bei
    // Distanz d ist d*tan(30) - bei d=26 sind das ~15 Bloecke).
    let eye = glam::Vec3::new(
        (chunk_x * CHUNK_SIZE + 16) as f32,
        (surface_height + 26) as f32,
        (chunk_z * CHUNK_SIZE + 16) as f32,
    );
    let projection =
        glam::camera::rh::proj::directx::perspective_infinite_reverse(60f32.to_radians(), 1.0, 0.1);
    let view = glam::camera::rh::view::look_to_mat4(
        eye,
        glam::Vec3::new(0.0, -1.0, 0.0),
        glam::Vec3::new(0.0, 0.0, -1.0),
    );
    let view_proj = projection * view;

    let shadow_pass = ShadowPass::new(
        &device,
        config.player.shadow_map_resolution,
        config.dev.shadow_depth_bias,
        config.dev.shadow_depth_bias_slope_scale,
    );
    let chunk_pipeline = pipeline::create(&device, &queue, FORMAT, SAMPLES);
    let mut renderer = ChunkRenderer::new(&device, &chunk_pipeline, view_proj, &config, &shadow_pass);

    let mut chunk = Chunk::empty();
    generator.generate_chunk(chunk_x, chunk_y, chunk_z, &mut chunk);
    let mesh = mesh_chunk(&chunk, chunk_x, chunk_y, chunk_z, [None; 6], |wx, wy, wz| generator.is_solid(wx, wy, wz));
    renderer.update_camera(&queue, view_proj, eye, glam::Vec3::new(0.0, -1.0, 0.0));
    let handle = renderer.alloc_chunk(&queue, &mesh);
    let aabb_min = glam::Vec3::new(
        (chunk_x * CHUNK_SIZE) as f32,
        (chunk_y * CHUNK_SIZE) as f32,
        (chunk_z * CHUNK_SIZE) as f32,
    );
    let aabb_max = aabb_min + glam::Vec3::splat(CHUNK_SIZE as f32);
    renderer.update_chunk_meta(&queue, 0, aabb_min, aabb_max, &handle);

    let msaa_color = create_msaa_color_view(&device, SIZE, SIZE, SAMPLES, FORMAT);
    let depth = create_depth_view(&device, SIZE, SIZE, SAMPLES);
    let hzb = HzbPass::new(&device, SIZE, SIZE, &depth, SAMPLES > 1, SAMPLES);
    renderer.rebuild_cull_bind_group(&device, &hzb);
    let resolve = create_resolve_color_view(&device, SIZE, SIZE, FORMAT);
    let ao = create_ao_view(&device, SIZE, SIZE);
    let mut ssao = SsaoPass::new(&device);
    ssao.rebuild_bind_group(&device, &depth);
    ssao.update_uniforms(&queue, projection, SIZE as f32, SIZE as f32, 2.0, 1.4, true);
    let mut blur = SsaoBlurPass::new(&device, FORMAT);
    blur.rebuild_bind_group(&device, &ao, &depth, &resolve);
    blur.update_uniforms(&queue, SIZE as f32, SIZE as f32, 0.0008);

    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output"),
        size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    hzb.generate(&mut encoder);
    renderer.dispatch_cull(&mut encoder, &queue, view_proj, eye, &hzb);
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("chunk"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &msaa_color,
                resolve_target: Some(&resolve),
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 }), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&chunk_pipeline.pipeline);
        renderer.render(&mut pass);
    }
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssao_raw_ao"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ao,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        ssao.render(&mut pass);
    }
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssao_blur"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        blur.render(&mut pass);
    }

    let padded = (SIZE * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &output, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(SIZE) } },
        wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
    );
    queue.submit(std::iter::once(encoder.finish()));

    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let data = readback.slice(..).get_mapped_range().expect("map");
    let mut green = 0u32;
    for y in 0..SIZE {
        let row = &data[(y * padded) as usize..];
        for x in 0..SIZE {
            let p = (x * 4) as usize;
            let (r, g, b) = (row[p] as i32, row[p + 1] as i32, row[p + 2] as i32);
            if g > r + 20 && g > b + 20 {
                green += 1;
            }
        }
    }
    green as f32 / (SIZE * SIZE) as f32
}

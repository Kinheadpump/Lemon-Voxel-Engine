// Visuelle Regressions-Pruefung ohne Fenster: rendert Chunk (0,0) senkrecht von oben durch die
// echte Engine-Pipeline (MSAA + SSAO), liest die Pixel zurueck und prueft, dass die Gras-Oberflaeche
// tatsaechlich gruen erscheint. Faengt invertiertes Culling / Tiefen-Regressionen fruehzeitig ab.

use voxel_engine::engine::config::EngineConfig;
use voxel_engine::engine::core::mesher::mesh_chunk;
use voxel_engine::engine::render::pipeline::{
    self, create_depth_view, create_msaa_color_view, create_resolve_color_view,
};
use voxel_engine::engine::render::renderer::ChunkRenderer;
use voxel_engine::engine::render::ssao::SsaoPass;
use voxel_engine::game::world::chunk::Chunk;
use voxel_engine::game::world::generator::TerrainGenerator;

const SIZE: u32 = 256;
const SAMPLES: u32 = 4;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

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
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("headless"),
            required_features: wgpu::Features::INDIRECT_FIRST_INSTANCE | wgpu::Features::SHADER_DRAW_INDEX,
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("kein device");

    let eye = glam::Vec3::new(16.0, 40.0, 16.0);
    let projection =
        glam::camera::rh::proj::directx::perspective_infinite_reverse(60f32.to_radians(), 1.0, 0.1);
    let view = glam::camera::rh::view::look_to_mat4(
        eye,
        glam::Vec3::new(0.0, -1.0, 0.0),
        glam::Vec3::new(0.0, 0.0, -1.0),
    );
    let view_proj = projection * view;

    let config = EngineConfig::default();
    let chunk_pipeline = pipeline::create(&device, &queue, FORMAT, SAMPLES);
    let mut renderer = ChunkRenderer::new(&device, &chunk_pipeline, view_proj, &config);

    let generator = TerrainGenerator::new(&config);
    let mut chunk = Chunk::empty();
    generator.generate_chunk(0, 0, 0, &mut chunk);
    let mesh = mesh_chunk(&chunk, 0, 0, 0, |wx, wy, wz| generator.is_solid(wx, wy, wz));
    renderer.update_camera(&queue, view_proj);
    let handle = renderer.alloc_chunk(&queue, &mesh, glam::Vec3::ZERO);
    renderer.set_visible(&queue, &[handle]);

    let msaa_color = create_msaa_color_view(&device, SIZE, SIZE, SAMPLES, FORMAT);
    let depth = create_depth_view(&device, SIZE, SIZE, SAMPLES);
    let resolve = create_resolve_color_view(&device, SIZE, SIZE, FORMAT);
    let mut ssao = SsaoPass::new(&device, FORMAT);
    ssao.rebuild_bind_group(&device, &depth, &resolve);
    ssao.update_uniforms(&queue, projection, SIZE as f32, SIZE as f32, 2.0, 1.4, true);

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
            label: Some("ssao"),
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
        ssao.render(&mut pass);
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

pub mod game;
pub mod engine;

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use engine::core::mesher::mesh_chunk;
use engine::render::pipeline;
use game::world::chunk::Chunk;

fn build_test_chunk() -> Chunk {
    let mut chunk = Chunk::empty();

    for z in 0..32 {
        for x in 0..32 {
            chunk.set_block(x, 0, z, 1);
        }
    }
    chunk.set_block(16, 1, 16, 2);

    chunk
}

fn run_mesher_smoke_test() {
    let mesh = mesh_chunk(&build_test_chunk());

    const DIR_NAMES: [&str; 6] = ["-X", "+X", "-Y", "+Y", "-Z", "+Z"];
    for (dir, faces) in mesh.faces.iter().enumerate() {
        log::info!("Mesher-Test Richtung {}: {} Faces", DIR_NAMES[dir], faces.len());
    }
}

struct DirectionDrawData {
    bind_group: wgpu::BindGroup,
    face_count: u32,
}

struct ChunkDrawData {
    directions: [Option<DirectionDrawData>; 6],
}

fn build_chunk_draw_data(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view_proj: glam::Mat4,
) -> ChunkDrawData {
    use wgpu::util::DeviceExt;

    let mesh = mesh_chunk(&build_test_chunk());

    let camera_data = pipeline::CameraUniformData { view_proj: view_proj.to_cols_array_2d() };
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera_uniform_buffer"),
        contents: bytemuck::bytes_of(&camera_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let mut directions: [Option<DirectionDrawData>; 6] = std::array::from_fn(|_| None);

    for (dir, faces) in mesh.faces.iter().enumerate() {
        if faces.is_empty() {
            continue;
        }

        let direction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("direction_uniform_buffer"),
            contents: bytemuck::bytes_of(&pipeline::DIRECTION_VECTORS[dir]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("face_storage_buffer"),
            contents: bytemuck::cast_slice(faces),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk_face_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: direction_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: storage_buffer.as_entire_binding() },
            ],
        });

        directions[dir] = Some(DirectionDrawData { bind_group, face_count: faces.len() as u32 });
    }

    ChunkDrawData { directions }
}

struct GpuContext {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    chunk_pipeline: pipeline::ChunkPipeline,
    chunk_draw_data: ChunkDrawData,
}

impl GpuContext {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("Surface-Erstellung fehlgeschlagen");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .expect("Kein kompatibler GPU-Adapter gefunden");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("primary_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Device-Erstellung fehlgeschlagen");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let chunk_pipeline = pipeline::create(&device, config.format);

        let aspect = config.width as f32 / config.height as f32;
        let projection =
            glam::camera::rh::proj::directx::perspective(60f32.to_radians(), aspect, 0.1, 200.0);
        let view = glam::camera::rh::view::look_at_mat4(
            glam::Vec3::new(48.0, 40.0, 48.0),
            glam::Vec3::new(16.0, 8.0, 16.0),
            glam::Vec3::Y,
        );
        let chunk_draw_data =
            build_chunk_draw_data(&device, &chunk_pipeline.bind_group_layout, projection * view);

        Self { surface, device, queue, config, window, chunk_pipeline, chunk_draw_data }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                log::error!("Surface-Validierungsfehler");
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.chunk_pipeline.pipeline);
            for direction in self.chunk_draw_data.directions.iter().flatten() {
                render_pass.set_bind_group(0, &direction.bind_group, &[]);
                render_pass.draw(0..6, 0..direction.face_count);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);
    }
}

#[derive(Default)]
struct App {
    gpu: Option<GpuContext>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        let window_attributes = Window::default_attributes().with_title("Voxel Engine");
        let window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .expect("Fenster-Erstellung fehlgeschlagen"),
        );

        self.gpu = Some(pollster::block_on(GpuContext::new(window)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                gpu.render();
                gpu.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_module("wgpu_hal::vulkan::instance", log::LevelFilter::Off)
        .filter_module("wgpu_core", log::LevelFilter::Error)
        .init();

    run_mesher_smoke_test();

    let event_loop = EventLoop::new().expect("Event-Loop-Erstellung fehlgeschlagen");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event-Loop abgestürzt");
}

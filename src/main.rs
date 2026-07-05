pub mod game;
pub mod engine;

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Window, WindowId};

use engine::render::pipeline;
use engine::render::renderer::ChunkRenderer;
use game::input::{InputState, MoveCommand};
use game::math::camera::Camera;
use game::world::manager::ChunkManager;

const MOVE_SPEED_UNITS_PER_SEC: f32 = 12.0;
const MOUSE_SENSITIVITY_RADIANS_PER_PIXEL: f32 = 0.0025;

struct GpuContext {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    chunk_pipeline: pipeline::ChunkPipeline,
    renderer: ChunkRenderer,
    depth_view: wgpu::TextureView,
}

impl GpuContext {
    async fn new(window: Arc<Window>, initial_view_proj: glam::Mat4) -> Self {
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
        let renderer = ChunkRenderer::new(&device, &chunk_pipeline, initial_view_proj);
        let depth_view = pipeline::create_depth_view(&device, config.width, config.height);

        Self { surface, device, queue, config, window, chunk_pipeline, renderer, depth_view }
    }

    fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height as f32
    }

    fn update_camera(&self, view_proj: glam::Mat4) {
        self.renderer.update_camera(&self.queue, view_proj);
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = pipeline::create_depth_view(&self.device, width, height);
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.chunk_pipeline.pipeline);
            self.renderer.render(&mut render_pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);
    }
}

struct App {
    gpu: Option<GpuContext>,
    camera: Camera,
    input: InputState,
    chunk_manager: ChunkManager,
    last_frame: Instant,
    last_stats_log: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            gpu: None,
            camera: Camera::new(glam::Vec3::new(16.0, 40.0, 16.0), 0.0, -0.6),
            input: InputState::default(),
            chunk_manager: ChunkManager::new(),
            last_frame: Instant::now(),
            last_stats_log: Instant::now(),
        }
    }
}

impl App {
    fn apply_movement(&mut self, dt: f32) {
        let forward = self.camera.forward();
        let right = self.camera.right();
        let mut motion = glam::Vec3::ZERO;

        for command in self.input.active_commands() {
            match command {
                MoveCommand::Forward => motion += forward,
                MoveCommand::Backward => motion -= forward,
                MoveCommand::StrafeLeft => motion -= right,
                MoveCommand::StrafeRight => motion += right,
                MoveCommand::Ascend => motion += glam::Vec3::Y,
                MoveCommand::Descend => motion -= glam::Vec3::Y,
            }
        }

        if motion != glam::Vec3::ZERO {
            self.camera.position += motion.normalize() * MOVE_SPEED_UNITS_PER_SEC * dt;
        }

        let (dx, dy) = self.input.take_mouse_delta();
        self.camera.rotate(
            dx * MOUSE_SENSITIVITY_RADIANS_PER_PIXEL,
            -dy * MOUSE_SENSITIVITY_RADIANS_PER_PIXEL,
        );
    }
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

        window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
            .ok();
        window.set_cursor_visible(false);

        let aspect = window.inner_size().width.max(1) as f32 / window.inner_size().height.max(1) as f32;
        let initial_view_proj = self.camera.view_projection(aspect);

        self.gpu = Some(pollster::block_on(GpuContext::new(window, initial_view_proj)));
        self.last_frame = Instant::now();
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input.handle_mouse_delta(delta.0 as f32, delta.1 as f32);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if code == winit::keyboard::KeyCode::Escape
                        && event.state == ElementState::Pressed
                    {
                        event_loop.exit();
                    }
                    self.input.handle_key(code, event.state == ElementState::Pressed);
                }
            }
            WindowEvent::RedrawRequested => {
                let aspect = gpu.aspect();

                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                self.apply_movement(dt);

                let gpu = self.gpu.as_mut().expect("GPU-Kontext verschwunden");
                self.chunk_manager.update(self.camera.position, &gpu.queue, &gpu.renderer);

                let view_proj = self.camera.view_projection(aspect);
                gpu.update_camera(view_proj);

                gpu.render();

                if now.duration_since(self.last_stats_log).as_secs_f32() >= 1.0 {
                    self.last_stats_log = now;
                    log::info!(
                        "Aktive Chunks: {} | Position: ({:.1}, {:.1}, {:.1})",
                        self.chunk_manager.loaded_chunk_count(),
                        self.camera.position.x,
                        self.camera.position.y,
                        self.camera.position.z
                    );
                }

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

    let event_loop = EventLoop::new().expect("Event-Loop-Erstellung fehlgeschlagen");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event-Loop abgestürzt");
}

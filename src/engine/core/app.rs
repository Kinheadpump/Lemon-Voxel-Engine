use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::engine::config::EngineConfig;
use crate::engine::render::context::GpuContext;
use crate::game::input::{InputState, MoveCommand};
use crate::game::math::camera::Camera;
use crate::game::math::frustum::Frustum;
use crate::game::world::manager::ChunkManager;

pub struct App {
    config: EngineConfig,
    gpu: Option<GpuContext>,
    camera: Camera,
    input: InputState,
    chunk_manager: ChunkManager,
    last_frame: Instant,
    last_stats_log: Instant,
}

impl App {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            config,
            gpu: None,
            camera: Camera::new(
                glam::Vec3::new(16.0, 40.0, 16.0),
                0.0,
                -0.6,
                config.fov_y_radians,
            ),
            input: InputState::default(),
            chunk_manager: ChunkManager::new(config.render_distance_chunks),
            last_frame: Instant::now(),
            last_stats_log: Instant::now(),
        }
    }

    pub fn run(config: EngineConfig) {
        let event_loop = EventLoop::new().expect("Event-Loop-Erstellung fehlgeschlagen");
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut app = App::new(config);
        event_loop.run_app(&mut app).expect("Event-Loop abgestürzt");
    }

    fn apply_movement(&mut self, dt: f32) {
        let forward = self.camera.forward_flat();
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
            self.camera.position += motion.normalize() * self.config.movement_speed * dt;
        }

        let (dx, dy) = self.input.take_mouse_delta();
        self.camera.rotate(
            dx * self.config.mouse_sensitivity,
            -dy * self.config.mouse_sensitivity,
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

        self.gpu = Some(pollster::block_on(GpuContext::new(window, initial_view_proj, &self.config)));
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

                let view_proj = self.camera.view_projection(aspect);
                let frustum = Frustum::from_view_projection(view_proj);

                self.chunk_manager.update(self.camera.position, &frustum);

                let gpu = self.gpu.as_mut().expect("GPU-Kontext verschwunden");
                gpu.update_camera(view_proj);
                gpu.upload_frame(self.chunk_manager.visible_chunks());

                gpu.render();

                if now.duration_since(self.last_stats_log).as_secs_f32() >= 1.0 {
                    self.last_stats_log = now;
                    log::info!(
                        "Aktive Chunks: {} | Sichtbare Chunks: {} | Position: ({:.1}, {:.1}, {:.1})",
                        self.chunk_manager.loaded_chunk_count(),
                        self.chunk_manager.visible_chunk_count(),
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

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::engine::config::EngineConfig;
use crate::engine::render::context::GpuContext;
use crate::engine::render::textures::TEXTURE_LAYER_STONE;
use crate::game::input::{InputState, MoveCommand};
use crate::game::math::camera::Camera;
use crate::game::math::cascades::compute_cascades;
use crate::game::math::sun::Sun;
use crate::game::physics::PlayerPhysics;
use crate::game::world::godrays::GodrayField;
use crate::game::world::manager::{ChunkManager, INTERACTION_REACH};

/// Block-Typ, der beim Platzieren verwendet wird - "einfache" Interaktion ohne Inventar/Auswahl.
const PLACE_BLOCK_ID: u16 = TEXTURE_LAYER_STONE as u16;

pub struct App {
    config: EngineConfig,
    gpu: Option<GpuContext>,
    camera: Camera,
    sun: Sun,
    input: InputState,
    physics: PlayerPhysics,
    chunk_manager: ChunkManager,
    godray_field: GodrayField,
    last_frame: Instant,
    last_stats_log: Instant,
    fps_ema: f32,
    hud_text: String,
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
            sun: Sun::new(config.sun_initial_time_of_day),
            input: InputState::default(),
            physics: PlayerPhysics::new(config.start_flying, &config),
            chunk_manager: ChunkManager::new(&config),
            godray_field: GodrayField::new(&config),
            last_frame: Instant::now(),
            last_stats_log: Instant::now(),
            fps_ema: 0.0,
            hud_text: String::with_capacity(256),
        }
    }

    pub fn run(config: EngineConfig) {
        let event_loop = EventLoop::new().expect("Event-Loop-Erstellung fehlgeschlagen");
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut app = App::new(config);
        event_loop.run_app(&mut app).expect("Event-Loop abgestürzt");
    }

    fn apply_movement(&mut self, dt: f32) {
        let (dx, dy) = self.input.take_mouse_delta();
        self.camera.rotate(dx * self.config.mouse_sensitivity, -dy * self.config.mouse_sensitivity);

        let forward = self.camera.forward_flat();
        let right = self.camera.right();

        let speed = if self.input.is_sprinting() {
            self.config.movement_speed * self.config.sprint_multiplier
        } else {
            self.config.movement_speed
        };

        let commands = self.input.active_commands();

        if self.physics.flying {
            let mut motion = glam::Vec3::ZERO;
            for &command in commands {
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
                self.camera.position += motion.normalize() * speed * dt;
            }
            return;
        }

        let mut horizontal = glam::Vec3::ZERO;
        for &command in commands {
            match command {
                MoveCommand::Forward => horizontal += forward,
                MoveCommand::Backward => horizontal -= forward,
                MoveCommand::StrafeLeft => horizontal -= right,
                MoveCommand::StrafeRight => horizontal += right,
                MoveCommand::Ascend | MoveCommand::Descend => {}
            }
        }
        if horizontal != glam::Vec3::ZERO {
            horizontal = horizontal.normalize() * speed;
        }

        // Kollisionsabfrage respektiert geladene/editierte Chunk-Daten (nicht nur das prozedurale
        // Terrain), damit abgebaute/platzierte Bloecke sofort physikalisch wirksam sind.
        let chunk_manager = &self.chunk_manager;
        let is_solid = |x: i32, y: i32, z: i32| chunk_manager.is_solid_at(x, y, z);
        self.physics.advance(
            dt,
            &is_solid,
            &mut self.camera.position,
            horizontal,
            self.input.is_jump_or_ascend_held(),
            self.config.gravity,
            self.config.jump_speed,
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
            WindowEvent::MouseInput { state, button, .. } => {
                self.input.handle_mouse_button(button, state == ElementState::Pressed);
            }
            WindowEvent::RedrawRequested => {
                let aspect = gpu.aspect();

                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Toggles muessen VOR Bewegung/Kamera-Update konsumiert werden, damit z.B. ein
                // Fly-Toggle noch im selben Frame die Bewegungsart beeinflusst und ein
                // Wireframe-Toggle noch im selben Frame ins Kamera-Uniform einfliesst.
                if self.input.take_hud_toggle_requested() {
                    self.gpu.as_mut().expect("GPU-Kontext verschwunden").toggle_hud();
                }
                if self.input.take_fly_toggle_requested() {
                    self.physics.toggle_flying();
                }
                if self.input.take_wireframe_toggle_requested() {
                    self.gpu.as_mut().expect("GPU-Kontext verschwunden").renderer.toggle_wireframe();
                }

                self.apply_movement(dt);
                self.sun.advance(dt, self.config.sun_cycle_seconds);

                let instant_fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
                self.fps_ema =
                    if self.fps_ema <= 0.0 { instant_fps } else { self.fps_ema * 0.9 + instant_fps * 0.1 };

                let view_proj = self.camera.view_projection(aspect);

                let gpu = self.gpu.as_mut().expect("GPU-Kontext verschwunden");
                gpu.update_camera(view_proj, self.camera.position, self.camera.forward());
                gpu.update_ssao(self.camera.projection_matrix(aspect));

                let direction_to_sun = self.sun.direction_to_sun();
                let cascades = compute_cascades(
                    &self.camera,
                    aspect,
                    self.sun.light_direction(),
                    self.config.shadow_cascade_count,
                    self.config.shadow_max_distance,
                    self.config.shadow_split_lambda,
                    self.config.shadow_map_resolution,
                );
                // Warmes Licht knapp ueber dem Horizont, neutrales Weiss im Zenit - rein
                // aesthetische Interpolation, keine physikalische Simulation.
                let sun_height = direction_to_sun.y.max(0.0);
                let sun_color = glam::Vec3::new(1.0, 0.85 + 0.15 * sun_height, 0.7 + 0.3 * sun_height);
                gpu.update_lighting(cascades, direction_to_sun, sun_color, self.config.sun_intensity, self.config.ambient_light);

                if let Some(instances) =
                    self.godray_field.update(self.camera.position, self.chunk_manager.generator())
                {
                    gpu.upload_godrays(instances);
                }

                {
                    let (queue, renderer) = gpu.queue_and_renderer();

                    if self.input.take_mine_requested() {
                        if let Some(hit) =
                            self.chunk_manager.raycast(self.camera.position, self.camera.forward(), INTERACTION_REACH)
                        {
                            self.chunk_manager.set_block(hit.block.x, hit.block.y, hit.block.z, 0, queue, renderer);
                        }
                    }
                    if self.input.take_place_requested() {
                        if let Some(hit) =
                            self.chunk_manager.raycast(self.camera.position, self.camera.forward(), INTERACTION_REACH)
                        {
                            let target = hit.block + hit.normal;
                            if !self.physics.occupies_block(self.camera.position, target) {
                                self.chunk_manager.set_block(
                                    target.x,
                                    target.y,
                                    target.z,
                                    PLACE_BLOCK_ID,
                                    queue,
                                    renderer,
                                );
                            }
                        }
                    }

                    self.chunk_manager.update(
                        self.camera.position,
                        &cascades,
                        self.config.shadow_cascade_count,
                        queue,
                        renderer,
                    );
                }

                let mode_text = if self.physics.flying {
                    "FLYING"
                } else if self.physics.grounded {
                    "WALKING"
                } else {
                    "FALLING"
                };

                self.hud_text.clear();
                let _ = write!(self.hud_text, "FPS: {:.0}\nFRAME: {:.2}MS / GPU: ", self.fps_ema, dt * 1000.0);
                match gpu.last_gpu_time_ms() {
                    Some(ms) => {
                        let _ = write!(self.hud_text, "{ms:.2}MS");
                    }
                    None => self.hud_text.push_str("N/A"),
                }
                self.hud_text.push_str("\nVRAM: ");
                match gpu.vram_usage_mb() {
                    Some(mb) => {
                        let _ = write!(self.hud_text, "{mb:.1}MB");
                    }
                    None => self.hud_text.push_str("N/A"),
                }
                let _ = write!(
                    self.hud_text,
                    "\nCHUNKS: {}\nDRAW CALLS (GPU-CULLED): {}\nPOS: {:.1} / {:.1} / {:.1}\nMODE: {mode_text} (F=TOGGLE, F4=WIREFRAME)",
                    self.chunk_manager.loaded_chunk_count(),
                    gpu.renderer.draw_call_count(),
                    self.camera.position.x,
                    self.camera.position.y,
                    self.camera.position.z
                );
                gpu.update_hud_text(&self.hud_text);

                gpu.render();

                if now.duration_since(self.last_stats_log).as_secs_f32() >= 1.0 {
                    self.last_stats_log = now;
                    // Nur einmal pro Sekunde - eine String-Allokation hier ist unkritisch.
                    let vram_text = match gpu.vram_usage_mb() {
                        Some(mb) => format!("{mb:.1}MB"),
                        None => "N/A".to_string(),
                    };
                    log::info!(
                        "FPS: {:.0} | Frame: {:.2}ms | VRAM: {} | Modus: {} | Aktive Chunks: {} | GPU-Draws: {} | Position: ({:.1}, {:.1}, {:.1})",
                        self.fps_ema,
                        dt * 1000.0,
                        vram_text,
                        mode_text,
                        self.chunk_manager.loaded_chunk_count(),
                        gpu.renderer.draw_call_count(),
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

use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use crossbeam_channel::Receiver;
use glam::Vec3;
use vx_render::{
    ChunkMeshRef, FlyCamera, GpuContext, TerrainRenderer, begin_terrain_pass, create_depth_view,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::{
    options::Options,
    region::{self, MeshedRegion},
};

const MOVE_SPEED: f32 = 40.0;
const SPRINT_MULTIPLIER: f32 = 4.0;
const MOUSE_SENSITIVITY: f32 = 0.002;

pub fn run(options: Options) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let (sender, receiver) = crossbeam_channel::bounded(1);
    let view_radius = options.view_radius;
    rayon::spawn(move || {
        let _ = sender.send(region::build_region(view_radius));
    });
    let mut app = App {
        options,
        region_receiver: receiver,
        state: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    options: Options,
    region_receiver: Receiver<MeshedRegion>,
    state: Option<RenderState>,
}

struct RenderState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    gpu: GpuContext,
    terrain: TerrainRenderer,
    camera: FlyCamera,
    held_keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    last_frame: Instant,
    frame_stats: FrameStats,
}

#[derive(Default)]
struct FrameStats {
    frames: u32,
    accumulated: Duration,
    worst: Duration,
}

impl FrameStats {
    fn record(&mut self, frame_time: Duration, terrain: &TerrainRenderer) {
        self.frames += 1;
        self.accumulated += frame_time;
        self.worst = self.worst.max(frame_time);
        if self.accumulated >= Duration::from_secs(1) {
            let average_ms = self.accumulated.as_secs_f64() * 1e3 / f64::from(self.frames);
            log::info!(
                "fps {:.0} | avg {average_ms:.2} ms | worst {:.2} ms | {} draws | {} quads",
                1e3 / average_ms,
                self.worst.as_secs_f64() * 1e3,
                terrain.draw_count(),
                terrain.quad_count(),
            );
            *self = Self::default();
        }
    }
}

impl RenderState {
    fn new(event_loop: &ActiveEventLoop, options: &Options) -> Result<Self> {
        let window =
            Arc::new(event_loop.create_window(Window::default_attributes().with_title("vx"))?);
        let instance = GpuContext::create_instance();
        let surface = instance.create_surface(window.clone())?;
        let gpu = pollster::block_on(GpuContext::new(instance, Some(&surface)))?;
        log::info!("{}", gpu.describe());
        let size = window.inner_size();
        let mut surface_config = surface
            .get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| anyhow::anyhow!("surface unsupported by adapter"))?;
        surface_config.present_mode = if options.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        surface.configure(&gpu.device, &surface_config);
        let depth_view =
            create_depth_view(&gpu.device, surface_config.width, surface_config.height);
        let terrain = TerrainRenderer::new(&gpu.device, surface_config.format);
        Ok(Self {
            window,
            surface,
            surface_config,
            depth_view,
            gpu,
            terrain,
            camera: FlyCamera::looking_at(Vec3::new(0.0, 120.0, 0.0), Vec3::new(60.0, 80.0, 60.0)),
            held_keys: HashSet::new(),
            cursor_grabbed: false,
            last_frame: Instant::now(),
            frame_stats: FrameStats::default(),
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface
            .configure(&self.gpu.device, &self.surface_config);
        self.depth_view = create_depth_view(
            &self.gpu.device,
            self.surface_config.width,
            self.surface_config.height,
        );
    }

    fn set_cursor_grab(&mut self, grabbed: bool) {
        let mode = if grabbed {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
        let applied = self.window.set_cursor_grab(mode).or_else(|_| {
            self.window.set_cursor_grab(if grabbed {
                CursorGrabMode::Confined
            } else {
                mode
            })
        });
        self.cursor_grabbed = grabbed && applied.is_ok();
        self.window.set_cursor_visible(!self.cursor_grabbed);
    }

    fn advance_camera(&mut self, dt: f32) {
        let axis = |positive: KeyCode, negative: KeyCode| {
            f32::from(u8::from(self.held_keys.contains(&positive)))
                - f32::from(u8::from(self.held_keys.contains(&negative)))
        };
        let forward = self.camera.forward();
        let flat_forward = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
        let direction = flat_forward * axis(KeyCode::KeyW, KeyCode::KeyS)
            + self.camera.flat_right() * axis(KeyCode::KeyD, KeyCode::KeyA)
            + Vec3::Y * axis(KeyCode::Space, KeyCode::ControlLeft);
        let speed = if self.held_keys.contains(&KeyCode::ShiftLeft) {
            MOVE_SPEED * SPRINT_MULTIPLIER
        } else {
            MOVE_SPEED
        };
        self.camera.position += direction.normalize_or_zero() * speed * dt;
    }

    fn render(&mut self) {
        let now = Instant::now();
        let frame_time = now - self.last_frame;
        self.last_frame = now;
        self.advance_camera(frame_time.as_secs_f32());

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface
                    .configure(&self.gpu.device, &self.surface_config);
                return;
            }
            _ => return,
        };
        let aspect = self.surface_config.width as f32 / self.surface_config.height as f32;
        self.terrain
            .update_camera(&self.gpu.queue, &self.camera.uniform(aspect));
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = begin_terrain_pass(&mut encoder, &view, &self.depth_view);
            self.terrain.draw(&mut pass);
        }
        self.gpu.queue.submit([encoder.finish()]);
        self.gpu.queue.present(frame);
        self.frame_stats.record(frame_time, &self.terrain);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match RenderState::new(event_loop, &self.options) {
            Ok(state) => self.state = Some(state),
            Err(error) => {
                log::error!("renderer init failed: {error:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                state.set_cursor_grab(true);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                match event.state {
                    ElementState::Pressed if code == KeyCode::Escape => {
                        state.set_cursor_grab(false);
                    }
                    ElementState::Pressed => {
                        state.held_keys.insert(code);
                    }
                    ElementState::Released => {
                        state.held_keys.remove(&code);
                    }
                }
            }
            WindowEvent::RedrawRequested => state.render(),
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event
            && state.cursor_grabbed
        {
            state.camera.rotate(
                dx as f32 * MOUSE_SENSITIVITY,
                -dy as f32 * MOUSE_SENSITIVITY,
            );
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        let Some(state) = &mut self.state else { return };
        if let Ok(region) = self.region_receiver.try_recv() {
            region.log_summary();
            state.terrain.upload_static(
                &state.gpu.device,
                region
                    .meshes
                    .iter()
                    .map(|(pos, mesh)| ChunkMeshRef { pos: *pos, mesh }),
            );
        }
        state.window.request_redraw();
    }
}

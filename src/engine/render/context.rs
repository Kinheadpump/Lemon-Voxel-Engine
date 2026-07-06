use std::sync::Arc;

use winit::window::Window;

use crate::engine::config::EngineConfig;

use super::gpu_timer::GpuTimer;
use super::hud::HudRenderer;
use super::pipeline;
use super::renderer::ChunkRenderer;
use super::ssao::SsaoPass;

pub struct GpuContext {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub window: Arc<Window>,
    chunk_pipeline: pipeline::ChunkPipeline,
    pub renderer: ChunkRenderer,
    msaa_color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    resolve_color_view: wgpu::TextureView,
    clear_color: wgpu::Color,
    hud: HudRenderer,
    gpu_timer: Option<GpuTimer>,
    ssao: SsaoPass,
    msaa_samples: u32,
    ssao_enabled: bool,
    ssao_radius: f32,
    ssao_strength: f32,
}

impl GpuContext {
    pub async fn new(window: Arc<Window>, initial_view_proj: glam::Mat4, engine_config: &EngineConfig) -> Self {
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

        let adapter_features = adapter.features();
        let mut required_features = wgpu::Features::INDIRECT_FIRST_INSTANCE;
        if adapter_features.contains(super::gpu_timer::REQUIRED_FEATURES) {
            required_features |= super::gpu_timer::REQUIRED_FEATURES;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("primary_device"),
                required_features,
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

        let msaa_samples = engine_config.msaa_samples.max(1);

        let chunk_pipeline = pipeline::create(&device, &queue, config.format, msaa_samples);
        let renderer = ChunkRenderer::new(&device, &chunk_pipeline, initial_view_proj);
        let msaa_color_view =
            pipeline::create_msaa_color_view(&device, config.width, config.height, msaa_samples, config.format);
        let depth_view = pipeline::create_depth_view(&device, config.width, config.height, msaa_samples);
        let resolve_color_view = pipeline::create_resolve_color_view(&device, config.width, config.height, config.format);

        let mut ssao = SsaoPass::new(&device, config.format);
        ssao.rebuild_bind_group(&device, &depth_view, &resolve_color_view);

        let hud = HudRenderer::new(&device, &queue, config.format, engine_config.hud_visible_default);
        let gpu_timer = GpuTimer::try_new(&device, &queue);

        if gpu_timer.is_none() {
            log::warn!("GPU-Timestamp-Queries auf dieser Hardware nicht unterstuetzt - GPU-Render-Time im HUD bleibt leer");
        }

        Self {
            surface,
            device,
            queue,
            config,
            window,
            chunk_pipeline,
            renderer,
            msaa_color_view,
            depth_view,
            resolve_color_view,
            clear_color: engine_config.clear_color,
            hud,
            gpu_timer,
            ssao,
            msaa_samples,
            ssao_enabled: engine_config.ssao_enabled,
            ssao_radius: engine_config.ssao_radius,
            ssao_strength: engine_config.ssao_strength,
        }
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height as f32
    }

    pub fn update_camera(&self, view_proj: glam::Mat4) {
        self.renderer.update_camera(&self.queue, view_proj);
    }

    pub fn update_ssao(&self, projection: glam::Mat4) {
        self.ssao.update_uniforms(
            &self.queue,
            projection,
            self.config.width as f32,
            self.config.height as f32,
            self.ssao_radius,
            self.ssao_strength,
            self.ssao_enabled,
        );
    }

    pub fn toggle_hud(&mut self) {
        self.hud.toggle();
    }

    pub fn update_hud_text(&mut self, lines: &[String]) {
        let width = self.config.width as f32;
        let height = self.config.height as f32;
        self.hud.update_text(&self.queue, width, height, lines);
    }

    pub fn last_gpu_time_ms(&self) -> Option<f32> {
        self.gpu_timer.as_ref().and_then(GpuTimer::last_gpu_time_ms)
    }

    pub fn vram_usage_mb(&self) -> Option<f32> {
        self.device
            .generate_allocator_report()
            .map(|report| report.total_allocated_bytes as f32 / (1024.0 * 1024.0))
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.msaa_color_view =
            pipeline::create_msaa_color_view(&self.device, width, height, self.msaa_samples, self.config.format);
        self.depth_view = pipeline::create_depth_view(&self.device, width, height, self.msaa_samples);
        self.resolve_color_view = pipeline::create_resolve_color_view(&self.device, width, height, self.config.format);
        self.ssao.rebuild_bind_group(&self.device, &self.depth_view, &self.resolve_color_view);
    }

    pub fn render(&mut self) {
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

        let swapchain_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_encoder"),
            });

        {
            let timestamp_writes = self.gpu_timer.as_ref().map(GpuTimer::timestamp_writes);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("chunk_opaque_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_color_view,
                    resolve_target: Some(&self.resolve_color_view),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
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
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.chunk_pipeline.pipeline);
            self.renderer.render(&mut render_pass);
        }

        if let Some(gpu_timer) = self.gpu_timer.as_mut() {
            gpu_timer.resolve(&mut encoder);
        }

        {
            let mut ssao_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_post_process_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swapchain_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(self.clear_color), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            self.ssao.render(&mut ssao_pass);
        }

        {
            let mut hud_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hud_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swapchain_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            self.hud.render(&mut hud_pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        if let Some(gpu_timer) = self.gpu_timer.as_mut() {
            gpu_timer.after_submit(&self.device);
        }

        self.queue.present(frame);
    }
}

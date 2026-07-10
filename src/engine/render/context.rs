use std::sync::Arc;

use winit::window::Window;

use crate::engine::config::EngineConfig;
use crate::game::math::cascades::{Cascade, MAX_SHADOW_CASCADES};

use super::blur::SsaoBlurPass;
use super::godray::GodrayPass;
use super::gpu_timer::GpuTimer;
use super::hud::HudRenderer;
use super::hzb::HzbPass;
use super::pipeline;
use super::renderer::ChunkRenderer;
use super::shadow::ShadowPass;
use super::skybox::SkyboxPass;
use super::ssao::SsaoPass;
use crate::game::world::godrays::GodrayInstanceData;

pub struct GpuContext {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub window: Arc<Window>,
    chunk_pipeline: pipeline::ChunkPipeline,
    pub renderer: ChunkRenderer,
    hzb: HzbPass,
    msaa_color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    resolve_color_view: wgpu::TextureView,
    ao_view: wgpu::TextureView,
    clear_color: wgpu::Color,
    hud: HudRenderer,
    gpu_timer: Option<GpuTimer>,
    ssao: SsaoPass,
    blur: SsaoBlurPass,
    ssao_blur_depth_threshold: f32,
    shadow_pass: ShadowPass,
    skybox: SkyboxPass,
    godray: GodrayPass,
    // Godrays vorerst deaktiviert (s. Kommentar in `App::window_event`) - Feld bleibt fuer die
    // spaetere Reaktivierung erhalten.
    #[allow(dead_code)]
    godray_temporal_blend: f32,
    /// Vom letzten `update_lighting`-Aufruf gemerkt, damit `render()` dieselben Kaskaden-Matrizen
    /// zum Befuellen der Shadow-Map-Ebenen nutzt, die auch im Lighting-Uniform des Opaque-Passes
    /// stehen (sonst liefen Shadow-Rendering und Sampling auseinander).
    cascades: [Cascade; MAX_SHADOW_CASCADES],
    cascade_count: u32,
    shadow_map_resolution: u32,
    last_view_proj: glam::Mat4,
    last_camera_pos: glam::Vec3,
    last_camera_forward: glam::Vec3,
    msaa_samples: u32,
    /// SSAO liest die Tiefe ueber eine `texture_depth_multisampled_2d`-Bindung; bei MSAA=1 ist die
    /// Depth-Textur nicht multisampled, sodass die SSAO-Bindgroup gar nicht erst gebaut werden darf.
    msaa_active: bool,
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
        // IMMEDIATES traegt pro Schatten-Kaskade/Richtung nur die 64-Byte Licht-View-Projection -
        // spart gegenueber Uniform-Buffern + Bind-Group-Wechseln pro Kaskade unnoetigen Overhead.
        let mut required_features = wgpu::Features::INDIRECT_FIRST_INSTANCE
            | wgpu::Features::SHADER_DRAW_INDEX
            | wgpu::Features::IMMEDIATES
            | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        if adapter_features.contains(super::gpu_timer::REQUIRED_FEATURES) {
            required_features |= super::gpu_timer::REQUIRED_FEATURES;
        }

        let required_limits = wgpu::Limits {
            max_immediate_size: std::mem::size_of::<glam::Mat4>() as u32,
            ..wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("primary_device"),
                required_features,
                required_limits,
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
        let msaa_active = msaa_samples > 1;

        let shadow_pass = ShadowPass::new(
            &device,
            engine_config.shadow_map_resolution,
            engine_config.shadow_depth_bias,
            engine_config.shadow_depth_bias_slope_scale,
        );
        let chunk_pipeline = pipeline::create(&device, &queue, config.format, msaa_samples);
        let mut renderer = ChunkRenderer::new(&device, &chunk_pipeline, initial_view_proj, engine_config, &shadow_pass);
        let msaa_color_view =
            pipeline::create_msaa_color_view(&device, config.width, config.height, msaa_samples, config.format);
        let depth_view = pipeline::create_depth_view(&device, config.width, config.height, msaa_samples);
        let resolve_color_view = pipeline::create_resolve_color_view(&device, config.width, config.height, config.format);
        let ao_view = pipeline::create_ao_view(&device, config.width, config.height);

        let hzb = HzbPass::new(&device, config.width, config.height, &depth_view, msaa_active, msaa_samples);
        renderer.rebuild_cull_bind_group(&device, &hzb);

        let mut ssao = SsaoPass::new(&device);
        let mut blur = SsaoBlurPass::new(&device, config.format);
        if msaa_active {
            ssao.rebuild_bind_group(&device, &depth_view);
            blur.rebuild_bind_group(&device, &ao_view, &depth_view, &resolve_color_view);
        } else if engine_config.ssao_enabled {
            log::warn!("SSAO deaktiviert: benoetigt Multisampled Depth (msaa_samples > 1)");
        }

        let mut skybox = SkyboxPass::new(
            &device,
            config.format,
            msaa_active,
            engine_config.sky_zenith_day_color,
            engine_config.sky_horizon_day_color,
            engine_config.sky_night_color,
        );
        skybox.rebuild_bind_group(&device, &depth_view);

        let mut godray =
            GodrayPass::new(&device, config.format, msaa_active, engine_config.godray_count, &shadow_pass);
        godray.rebuild_render_bind_group(&device, &depth_view);

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
            hzb,
            msaa_color_view,
            depth_view,
            resolve_color_view,
            ao_view,
            clear_color: engine_config.clear_color,
            hud,
            gpu_timer,
            ssao,
            blur,
            ssao_blur_depth_threshold: engine_config.ssao_blur_depth_threshold,
            shadow_pass,
            skybox,
            godray,
            godray_temporal_blend: engine_config.godray_temporal_blend,
            cascades: [Cascade::default(); MAX_SHADOW_CASCADES],
            cascade_count: engine_config.shadow_cascade_count,
            shadow_map_resolution: engine_config.shadow_map_resolution,
            last_view_proj: initial_view_proj,
            last_camera_pos: glam::Vec3::ZERO,
            last_camera_forward: glam::Vec3::Z,
            msaa_samples,
            msaa_active,
            ssao_enabled: engine_config.ssao_enabled,
            ssao_radius: engine_config.ssao_radius,
            ssao_strength: engine_config.ssao_strength,
        }
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Split-Borrow fuer den Chunk-Update-Pfad: Queue (shared) und Renderer (mutable) gleichzeitig,
    /// ohne dass der Aufrufer die Queue clonen muss, um den Borrow-Konflikt aufzuloesen.
    pub fn queue_and_renderer(&mut self) -> (&wgpu::Queue, &mut ChunkRenderer) {
        (&self.queue, &mut self.renderer)
    }

    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height as f32
    }

    pub fn update_camera(&mut self, view_proj: glam::Mat4, camera_pos: glam::Vec3, camera_forward: glam::Vec3) {
        self.renderer.update_camera(&self.queue, view_proj, camera_pos, camera_forward);
        self.last_view_proj = view_proj;
        self.last_camera_pos = camera_pos;
        self.last_camera_forward = camera_forward;
    }

    /// Aktualisiert Sonnen-/Kaskaden-Uniforms (Opaque-Pass + Skybox) UND merkt sich die
    /// Kaskaden-Matrizen fuer den Shadow-Pass in `render()` - beide muessen exakt dieselben
    /// Matrizen verwenden wie das Lighting-Uniform, sonst laufen Rendering und Sampling auseinander.
    pub fn update_lighting(
        &mut self,
        cascades: [Cascade; MAX_SHADOW_CASCADES],
        direction_to_sun: glam::Vec3,
        sun_color: glam::Vec3,
        sun_intensity: f32,
        ambient: f32,
    ) {
        self.renderer.update_lighting(
            &self.queue,
            &cascades,
            self.cascade_count,
            self.shadow_map_resolution,
            direction_to_sun,
            sun_color,
            sun_intensity,
            ambient,
        );
        self.skybox.update_uniforms(&self.queue, self.last_view_proj, self.last_camera_pos, direction_to_sun);
        // Godrays vorerst deaktiviert (s. Kommentar in `App::window_event`) - Uniform-Update fuer
        // einen Pass, der ohnehin nicht mehr dispatcht/gerendert wird, waere reine Verschwendung.
        // self.godray.update_uniforms(
        //     &self.queue,
        //     self.last_view_proj,
        //     &cascades,
        //     self.cascade_count,
        //     self.last_camera_pos,
        //     self.last_camera_forward,
        //     direction_to_sun,
        //     sun_color,
        //     sun_intensity,
        //     self.godray_temporal_blend,
        // );
        self.cascades = cascades;
    }

    /// Volle Neu-Platzierung der Godray-Kandidaten (siehe `GodrayField::update` - nur bei
    /// ausreichender Kamerabewegung aufgerufen, kein Pro-Frame-Upload).
    pub fn upload_godrays(&self, instances: &[GodrayInstanceData]) {
        self.godray.upload_instances(&self.queue, instances);
    }

    pub fn update_ssao(&self, projection: glam::Mat4) {
        self.ssao.update_uniforms(
            &self.queue,
            projection,
            self.config.width as f32,
            self.config.height as f32,
            self.ssao_radius,
            self.ssao_strength,
            self.ssao_enabled && self.msaa_active,
        );
        self.blur.update_uniforms(
            &self.queue,
            self.config.width as f32,
            self.config.height as f32,
            self.ssao_blur_depth_threshold,
        );
    }

    pub fn toggle_hud(&mut self) {
        self.hud.toggle();
    }

    pub fn update_hud_text(&mut self, text: &str) {
        let width = self.config.width as f32;
        let height = self.config.height as f32;
        self.hud.update_text(&self.queue, width, height, text);
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
        self.ao_view = pipeline::create_ao_view(&self.device, width, height);
        if self.msaa_active {
            self.ssao.rebuild_bind_group(&self.device, &self.depth_view);
            self.blur.rebuild_bind_group(&self.device, &self.ao_view, &self.depth_view, &self.resolve_color_view);
        }
        self.skybox.rebuild_bind_group(&self.device, &self.depth_view);
        self.godray.rebuild_render_bind_group(&self.device, &self.depth_view);
        self.hzb.resize(&self.device, width, height, &self.depth_view);
        self.renderer.rebuild_cull_bind_group(&self.device, &self.hzb);
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

        // HZB-Aufbau aus dem noch nicht geleerten Depth-Buffer des VORHERIGEN Frames, dann
        // GPU-Driven Frustum+Occlusion-Culling fuer DIESES Frame (s. `hzb.rs`/`cull.wgsl`) - beide
        // MUESSEN vor dem Opaque-Pass laufen, der `depth_view` gleich ueberschreibt.
        self.hzb.generate(&mut encoder);
        self.renderer.dispatch_cull(&mut encoder, &self.queue, self.last_view_proj, self.last_camera_pos, &self.hzb);
        self.renderer.record_stats_copy(&mut encoder);

        self.shadow_pass.render_cascades(
            &mut encoder,
            &self.cascades,
            self.cascade_count,
            &self.renderer.shadow_draw_data(),
        );

        // Godrays vorerst deaktiviert (s. Kommentar in `App::window_event`).
        // self.godray.dispatch(&mut encoder);

        {
            let timestamp_writes = self.gpu_timer.as_ref().map(GpuTimer::timestamp_writes);

            // Ohne MSAA (sample_count == 1) darf kein `resolve_target` gesetzt werden - wgpu
            // verlangt dafuer eine tatsaechlich multisampled Quelle. Es wird dann direkt in die
            // Swapchain gerendert und der SSAO-Post-Process-Pass komplett uebersprungen.
            let (color_view, resolve_target) = if self.msaa_active {
                (&self.msaa_color_view, Some(&self.resolve_color_view))
            } else {
                (&swapchain_view, None)
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("chunk_opaque_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
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
            // Skybox NACH dem Opaque-Pass: fuellt per Tiefen-Discard nur die Pixel, die noch auf
            // dem Reverse-Z-Clear-Wert stehen. Ziel ist bei aktivem MSAA das schon aufgeloeste
            // `resolve_color_view` (die SSAO-Passe direkt danach liest dieselbe View als Input),
            // sonst direkt die Swapchain.
            let skybox_target = if self.msaa_active { &self.resolve_color_view } else { &swapchain_view };
            let mut skybox_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("skybox_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: skybox_target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            self.skybox.render(&mut skybox_pass);
        }

        if self.msaa_active {
            // Schreibt NUR den rohen (verrauschten) Occlusion-Faktor in `ao_view` - noch keine
            // Farbe. Der Blur-Pass danach glaettet das kantenerhaltend und multipliziert es erst
            // dann mit dem Bild; ohne diese Trennung war das Rauschen direkt sichtbar und, weil rein
            // bildschirmkoordinaten-basiert, bei Kamerabewegung wie ein bildschirmfixiertes
            // statisches Rauschen wahrnehmbar ("deep-fried"-Look).
            let mut ssao_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_raw_ao_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ao_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            self.ssao.render(&mut ssao_pass);
        }

        if self.msaa_active {
            let mut blur_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_blur_pass"),
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

            self.blur.render(&mut blur_pass);
        }

        // Godrays vorerst deaktiviert (s. Kommentar in `App::window_event`) - Platzierung/
        // Kantenerkennung liefert noch nicht das gewuenschte Ergebnis. Ganzer Pass uebersprungen
        // statt nur `self.godray.render(...)` auszukommentieren, damit auch keine leere
        // Render-Pass-Erstellung anfaellt.
        // {
        //     let mut godray_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        //         label: Some("godray_pass"),
        //         color_attachments: &[Some(wgpu::RenderPassColorAttachment {
        //             view: &swapchain_view,
        //             resolve_target: None,
        //             depth_slice: None,
        //             ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        //         })],
        //         depth_stencil_attachment: None,
        //         timestamp_writes: None,
        //         occlusion_query_set: None,
        //         multiview_mask: None,
        //     });
        //
        //     self.godray.render(&mut godray_pass);
        // }

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
        self.renderer.after_submit(&self.device);

        self.queue.present(frame);
    }
}

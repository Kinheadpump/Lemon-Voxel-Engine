use std::sync::{Arc, Mutex};

const READBACK_SLOTS: usize = 2;

pub const REQUIRED_FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY
    .union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    .union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);

struct ReadbackSlot {
    buffer: wgpu::Buffer,
    ready: Arc<Mutex<bool>>,
    busy: bool,
}

/// Misst die reine GPU-Ausfuehrungszeit des Chunk-Render-Passes ueber Timestamp-Queries.
/// Nicht auf jeder Hardware verfuegbar (z.B. manche Tile-Based-GPUs) - degradiert dann sauber
/// auf `None` statt abzustuerzen. Lesevorgang ist asynchron und blockiert den CPU-Thread nie
/// (zwei alternierende Readback-Buffer, non-blocking Poll).
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    slots: [ReadbackSlot; READBACK_SLOTS],
    frame_parity: usize,
    timestamp_period_ns: f32,
    last_gpu_time_ms: Option<f32>,
}

impl GpuTimer {
    pub fn try_new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        let supported = device.features().contains(REQUIRED_FEATURES);
        if !supported {
            return None;
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame_timestamp_query_set"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });

        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_timestamp_resolve_buffer"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let slots = std::array::from_fn(|_| ReadbackSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame_timestamp_readback_buffer"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            ready: Arc::new(Mutex::new(false)),
            busy: false,
        });

        Some(Self {
            query_set,
            resolve_buffer,
            slots,
            frame_parity: 0,
            timestamp_period_ns: queue.get_timestamp_period(),
            last_gpu_time_ms: None,
        })
    }

    pub fn timestamp_writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    /// Muss nach der Render-Pass, aber vor `queue.submit`, aufgerufen werden.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve_buffer, 0);

        let slot = &mut self.slots[self.frame_parity];
        if !slot.busy {
            encoder.copy_buffer_to_buffer(&self.resolve_buffer, 0, &slot.buffer, 0, 16);
        }
    }

    /// Muss nach `queue.submit` aufgerufen werden, um den Readback fuer diesen Frame anzustossen
    /// und das Ergebnis eines frueheren Frames abzuholen (nicht-blockierend).
    pub fn after_submit(&mut self, device: &wgpu::Device) {
        let period = self.timestamp_period_ns;
        let slot = &mut self.slots[self.frame_parity];

        if !slot.busy {
            slot.busy = true;
            let ready = Arc::clone(&slot.ready);
            slot.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    *ready.lock().unwrap() = true;
                }
            });
        }

        device.poll(wgpu::PollType::Poll).ok();

        for slot in &mut self.slots {
            let is_ready = *slot.ready.lock().unwrap();
            if !is_ready {
                continue;
            }

            let raw: [u64; 2] = {
                let view = slot.buffer.slice(..).get_mapped_range().expect("Buffer nicht gemappt");
                let mut timestamps = [0u64; 2];
                timestamps.copy_from_slice(bytemuck::cast_slice(&view));
                timestamps
            };
            slot.buffer.unmap();
            *slot.ready.lock().unwrap() = false;
            slot.busy = false;

            let elapsed_ns = raw[1].saturating_sub(raw[0]) as f32 * period;
            self.last_gpu_time_ms = Some(elapsed_ns / 1_000_000.0);
        }

        self.frame_parity = (self.frame_parity + 1) % READBACK_SLOTS;
    }

    pub fn last_gpu_time_ms(&self) -> Option<f32> {
        self.last_gpu_time_ms
    }
}

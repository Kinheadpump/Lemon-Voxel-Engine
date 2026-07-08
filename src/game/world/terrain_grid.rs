use super::chunk::CHUNK_SIZE;

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

const HEIGHT_GRID_AXIS_CAP: usize = CHUNK_SIZE as usize + 1;
const HEIGHT_GRID_CAP: usize = HEIGHT_GRID_AXIS_CAP * HEIGHT_GRID_AXIS_CAP;

/// Sparses 2D-Hoehenraster: Rauschen wird nur an `stride`-Abstaenden ausgewertet, dazwischen wird
/// bilinear interpoliert - bei stride=4 ueber 90% weniger Noise-Aufrufe als volle Aufloesung. Bei
/// den hier verwendeten Frequenzen (Gelaendemerkmale ueber viele Chunks) liegt der
/// Interpolationsfehler unterhalb der Voxelaufloesung.
pub struct HeightGrid {
    values: [f32; HEIGHT_GRID_CAP],
    axis_len: i32,
    stride: i32,
}

impl HeightGrid {
    /// `sample` wird mit chunk-lokalen Koordinaten (0..=CHUNK_SIZE, an `stride`-Vielfachen)
    /// aufgerufen; der Aufrufer rechnet sie in Weltkoordinaten um.
    pub fn fill(stride: i32, mut sample: impl FnMut(i32, i32) -> f32) -> Self {
        let stride = stride.clamp(1, CHUNK_SIZE);
        let stride = if CHUNK_SIZE % stride == 0 { stride } else { 1 };
        let axis_len = CHUNK_SIZE / stride + 1;

        let mut values = [0f32; HEIGHT_GRID_CAP];
        for gz in 0..axis_len {
            for gx in 0..axis_len {
                values[(gz * axis_len + gx) as usize] = sample(gx * stride, gz * stride);
            }
        }
        Self { values, axis_len, stride }
    }

    #[inline(always)]
    fn at(&self, gx: i32, gz: i32) -> f32 {
        self.values[(gz * self.axis_len + gx) as usize]
    }

    /// `local_x`/`local_z` muessen in `0..CHUNK_SIZE` liegen.
    pub fn sample(&self, local_x: i32, local_z: i32) -> f32 {
        let gx = local_x as f32 / self.stride as f32;
        let gz = local_z as f32 / self.stride as f32;
        let gx0 = gx.floor() as i32;
        let gz0 = gz.floor() as i32;
        let tx = gx.fract();
        let tz = gz.fract();

        let h00 = self.at(gx0, gz0);
        let h10 = self.at(gx0 + 1, gz0);
        let h01 = self.at(gx0, gz0 + 1);
        let h11 = self.at(gx0 + 1, gz0 + 1);
        lerp(lerp(h00, h10, tx), lerp(h01, h11, tx), tz)
    }
}

const CAVE_GRID_AXIS_CAP: usize = CHUNK_SIZE as usize / 2 + 1;
const CAVE_GRID_CAP: usize = CAVE_GRID_AXIS_CAP * CAVE_GRID_AXIS_CAP * CAVE_GRID_AXIS_CAP;

/// Sparses 3D-Dichteraster fuer Hoehlenrauschen, analog zu [`HeightGrid`] aber trilinear und mit
/// Mindest-Stride 2 (Stack-Bedarf waechst kubisch: stride=1 waere 33^3*4B = 143 KiB pro Chunk).
pub struct CaveGrid {
    values: [f32; CAVE_GRID_CAP],
    axis_len: i32,
    stride: i32,
}

impl CaveGrid {
    /// `sample` wird mit chunk-lokalem X/Z (0..=CHUNK_SIZE) und Welt-Y aufgerufen; `origin_y` ist
    /// die Welt-Y-Koordinate von `local_y == 0` in diesem Chunk.
    pub fn fill(stride: i32, origin_y: i32, mut sample: impl FnMut(i32, i32, i32) -> f32) -> Self {
        let stride = stride.clamp(2, CHUNK_SIZE / 2);
        let stride = if CHUNK_SIZE % stride == 0 { stride } else { 2 };
        let axis_len = CHUNK_SIZE / stride + 1;

        let mut values = [0f32; CAVE_GRID_CAP];
        for gz in 0..axis_len {
            for gy in 0..axis_len {
                for gx in 0..axis_len {
                    let index = ((gz * axis_len + gy) * axis_len + gx) as usize;
                    values[index] = sample(gx * stride, origin_y + gy * stride, gz * stride);
                }
            }
        }
        Self { values, axis_len, stride }
    }

    #[inline(always)]
    fn at(&self, gx: i32, gy: i32, gz: i32) -> f32 {
        self.values[((gz * self.axis_len + gy) * self.axis_len + gx) as usize]
    }

    /// `local_x`/`local_y`/`local_z` muessen in `0..CHUNK_SIZE` liegen.
    pub fn sample(&self, local_x: i32, local_y: i32, local_z: i32) -> f32 {
        let gx = local_x as f32 / self.stride as f32;
        let gy = local_y as f32 / self.stride as f32;
        let gz = local_z as f32 / self.stride as f32;
        let (gx0, tx) = (gx.floor() as i32, gx.fract());
        let (gy0, ty) = (gy.floor() as i32, gy.fract());
        let (gz0, tz) = (gz.floor() as i32, gz.fract());

        let c00 = lerp(self.at(gx0, gy0, gz0), self.at(gx0 + 1, gy0, gz0), tx);
        let c10 = lerp(self.at(gx0, gy0 + 1, gz0), self.at(gx0 + 1, gy0 + 1, gz0), tx);
        let c01 = lerp(self.at(gx0, gy0, gz0 + 1), self.at(gx0 + 1, gy0, gz0 + 1), tx);
        let c11 = lerp(self.at(gx0, gy0 + 1, gz0 + 1), self.at(gx0 + 1, gy0 + 1, gz0 + 1), tx);
        lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz)
    }
}

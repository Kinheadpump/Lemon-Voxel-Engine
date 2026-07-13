use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use bevy_math::Vec2;
use noiz::prelude::*;
use noiz::{Noise, NoiseFunction};

use crate::engine::config::EngineConfig;

/// Hierarchische Hoehen-/Klima-Synthese nach dem InfiniteDiffusion-Schema (Goslin, SIGGRAPH'26):
/// eine Kaskade progressiv feinerer Ebenen, jede erzeugt ueberlappende FENSTER, deren Beitraege
/// per Gewichtskernel gemittelt werden (MultiDiffusion-Blending, Gl. 2 des Papers) - nahtlos,
/// unendlich, seed-konsistent und mit O(1)-Random-Access. Die Verfeinerungsfunktion Phi pro Ebene
/// ist hier prozedural (parent-konditioniertes Ridged-Detail + Erosions-Relaxation) statt eines
/// gelernten Modells - die Fenster-/Cache-/Blending-Infrastruktur ist identisch und Phi damit
/// spaeter durch Modell-Inferenz ersetzbar, ohne Sampler oder Cache anzufassen.
pub struct TerrainPyramid {
    base: Noise<BaseFbm>,
    base_frequency: f32,
    continental_amplitude: f32,
    temperature: Noise<common_noise::Perlin>,
    humidity: Noise<common_noise::Perlin>,
    climate_frequency: f32,
    detail: [Noise<DetailFbm>; LEVEL_COUNT],
    mountain_amplitude: f32,
    mountain_exponent: f32,
    sea_compression_range: f32,
    sea_compression_exponent: f32,
    noise_origin_offset: f32,
    window_caches: [RwLock<WindowMap>; LEVEL_COUNT],
}

/// Dekodierte Spalten-Antwort des Samplers - Hoehe in Weltbloecken (linear, see-komprimiert),
/// Klima in snorm mit Hoehen-Lapse bereits eingerechnet.
#[derive(Clone, Copy)]
pub struct ColumnSample {
    pub height: f32,
    pub temperature: f32,
    pub humidity: f32,
}

type BaseFbm = common_noise::Fbm<common_noise::Perlin>;
type DetailFbm = common_noise::Fbm<common_noise::Perlin>;

pub const LEVEL_COUNT: usize = 4;

/// Fenstergroesse in Ebenen-Pixeln - mit Stride = halbe Groesse deckt JEDER Pixel exakt 4 Fenster
/// ab (2 pro Achse), s. `blended_pixel`.
const WINDOW_SIZE: i32 = 32;
const WINDOW_STRIDE: i32 = 16;
/// Rand-Puffer um jedes Fenster, in dem Nachbarschafts-Operatoren (Erosion) volle Nachbarn sehen -
/// nach dem Filtern weggeschnitten, Restdifferenzen an den Raendern glaettet das Fenster-Blending.
const WINDOW_APRON: i32 = 4;
const PADDED_SIZE: i32 = WINDOW_SIZE + 2 * WINDOW_APRON;
const PADDED_AREA: usize = (PADDED_SIZE * PADDED_SIZE) as usize;

pub const CHANNELS: usize = 3;
const CH_HEIGHT: usize = 0;
const CH_TEMPERATURE: usize = 1;
const CH_HUMIDITY: usize = 2;

/// Fertiges Fenster: px-major interleaved (`(v*W+u)*CHANNELS + ch`), Hoehe in Signed-Sqrt-Raum
/// (Paper 5.1: komprimiert Hochrelief-Werte, gleichmaessige Varianz beim Blenden ueber Fenster).
type WindowData = [f32; (WINDOW_SIZE * WINDOW_SIZE) as usize * CHANNELS];
type WindowMap = HashMap<(i32, i32), Arc<WindowData>>;

/// Max. Fenster pro Ebene im geteilten Cache (12 KiB/Fenster -> 6 MiB/Ebene Deckel). Eviction ist
/// immer sicher: Fenster sind reine Funktionen des Seeds und werden deterministisch identisch
/// nachberechnet (Paper 3.3, LRU-Eigenschaft).
const WINDOW_CACHE_CAP: usize = 512;

/// Minimalgewicht des separablen Dreieckskernels am Fensterrand - exakt 0 wuerde Randpixel
/// beitragslos machen und die Gewichtssumme dort degenerieren lassen.
const WEIGHT_EPSILON: f32 = 1.0 / 32.0;

/// Grundrauhigkeit (Bloecke) der Ebenen ausserhalb von Bergmasken - verhindert Billardtisch-Flaechen,
/// ohne das Gebirgs-Budget (`mountain_amplitude`) anzutasten.
const PLAINS_ROUGHNESS: f32 = 9.0;
/// Ab dieser Bergmaske dominiert Ridged- statt Smooth-Detail vollstaendig.
const RIDGE_FULL_BLEND_MASK: f32 = 0.8;
/// Temperaturabfall pro Weltblock Hoehe ueber dem Meer (snorm-Einheiten) - koppelt Klima kausal an
/// das erzeugte Relief (Fels-/Nadelwald-Grenzen folgen Bergen statt eigenem Rauschen).
const TEMPERATURE_LAPSE_PER_BLOCK: f32 = 0.004;
/// Steigungsschwelle (Bloecke pro Block) der thermischen Relaxation - flachere Haenge bleiben
/// unangetastet, steilere werden Richtung Schuttkegel geglaettet.
const EROSION_TALUS_SLOPE: f32 = 1.2;
const EROSION_STRENGTH: f32 = 0.4;

struct LevelSpec {
    blocks_per_pixel: i32,
    /// Anteil dieses Frequenzbands am Gesamt-Detail-Budget - Summe ueber alle Ebenen = 1.
    detail_share: f32,
    /// Wellenlaenge (Bloecke) des Detail-fBm dieser Ebene - zwischen eigenem und Eltern-Pixelmass.
    detail_wavelength: f32,
    erosion_iterations: u32,
}

/// Skalenfaktor 4 pro Stufe: die Kaskadenkosten amortisieren geometrisch (Paper 8: Faktor
/// a/(a-1) = 1.33 unabhaengig von der Tiefe). Ebene 0 traegt kein Detail (Basis-Synthese).
const LEVELS: [LevelSpec; LEVEL_COUNT] = [
    LevelSpec { blocks_per_pixel: 256, detail_share: 0.0, detail_wavelength: 0.0, erosion_iterations: 0 },
    LevelSpec { blocks_per_pixel: 64, detail_share: 0.5, detail_wavelength: 384.0, erosion_iterations: 0 },
    LevelSpec { blocks_per_pixel: 16, detail_share: 0.32, detail_wavelength: 96.0, erosion_iterations: 2 },
    LevelSpec { blocks_per_pixel: 4, detail_share: 0.18, detail_wavelength: 24.0, erosion_iterations: 2 },
];

const SEA_LEVEL_F: f32 = 0.0;

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline(always)]
fn signed_pow(value: f32, exponent: f32) -> f32 {
    value.signum() * value.abs().powf(exponent)
}

#[inline(always)]
fn signed_sqrt_encode(height: f32) -> f32 {
    height.signum() * height.abs().sqrt()
}

#[inline(always)]
fn signed_sqrt_decode(value: f32) -> f32 {
    value.signum() * value * value
}

/// Separabler linearer Kernel (1 im Zentrum, epsilon am Rand) - die Kernel-Wahl des Papers (6.0:
/// FID 19.32 -> 14.78 gegenueber konstanter Gewichtung).
#[inline(always)]
fn triangle_weight(u: i32) -> f32 {
    let centered = ((u as f32 + 0.5) / WINDOW_SIZE as f32) * 2.0 - 1.0;
    (1.0 - centered.abs()).max(WEIGHT_EPSILON)
}

/// Direkt-gemapptes Thread-lokales Memo der zuletzt benutzten Fenster-Arcs - der Sampler holt pro
/// Spalte bis zu 16 Fenster-Referenzen, fast immer dieselben 4; ohne Memo wuerde jede davon den
/// geteilten `RwLock` treffen.
const WINDOW_MEMO_SLOTS: usize = 64;
type WindowMemoSlot = Option<(usize, i32, i32, Arc<WindowData>)>;

thread_local! {
    static WINDOW_MEMO: RefCell<[WindowMemoSlot; WINDOW_MEMO_SLOTS]> =
        const { RefCell::new([const { None }; WINDOW_MEMO_SLOTS]) };
}

#[inline(always)]
fn window_memo_slot(level: usize, wi: i32, wj: i32) -> usize {
    let hash = (level as u32).wrapping_mul(0xC2B2_AE35)
        ^ (wi as u32).wrapping_mul(0x9E37_79B1)
        ^ (wj as u32).wrapping_mul(0x85EB_CA6B);
    (hash as usize) & (WINDOW_MEMO_SLOTS - 1)
}

impl TerrainPyramid {
    pub fn new(config: &EngineConfig) -> Self {
        let seed = config.dev.terrain_seed;
        let fbm = |octaves: u32, gain: f32, salt: u32| {
            let mut noise = Noise::from(LayeredNoise::new(
                Normed::default(),
                Persistence(gain),
                FractalLayers {
                    layer: Octave(common_noise::Perlin::default()),
                    lacunarity: 2.0,
                    amount: octaves,
                },
            ));
            noise.set_seed(seed.wrapping_add(salt));
            noise
        };

        let mut temperature = Noise::<common_noise::Perlin>::default();
        temperature.set_seed(seed.wrapping_add(0x68E3_1DA4));
        let mut humidity = Noise::<common_noise::Perlin>::default();
        humidity.set_seed(seed.wrapping_add(0xB529_7A4D));

        Self {
            base: fbm(3, 0.5, 0x9E37_79B9),
            base_frequency: 1.0 / config.dev.terrain_continent_scale_blocks.max(64.0),
            continental_amplitude: config.dev.terrain_continental_amplitude,
            temperature,
            humidity,
            climate_frequency: 1.0 / config.dev.terrain_climate_scale_blocks.max(64.0),
            detail: std::array::from_fn(|level| {
                fbm(2, 0.5, 0x85EB_CA77_u32.wrapping_add(level as u32 * 0x27D4_EB2F))
            }),
            mountain_amplitude: config.dev.terrain_mountain_amplitude,
            mountain_exponent: config.dev.terrain_mountain_exponent,
            sea_compression_range: config.dev.terrain_sea_compression_range.max(1.0),
            sea_compression_exponent: config.dev.terrain_sea_compression_exponent,
            noise_origin_offset: config.dev.terrain_noise_origin_offset,
            window_caches: std::array::from_fn(|_| RwLock::new(WindowMap::default())),
        }
    }

    /// Einzige oeffentliche Abfrage: dekodierte Spalte an einer Weltposition. O(1) amortisiert
    /// (max. 16 Fenster-Memo-Zugriffe), deterministisch im Seed, unabhaengig von Abfragereihenfolge
    /// und Cache-Zustand (Paper 3.5).
    pub fn sample(&self, world_x: i32, world_z: i32) -> ColumnSample {
        let raw = self.sample_level(LEVEL_COUNT - 1, world_x as f32, world_z as f32);
        let height = self.compress_toward_sea_level(signed_sqrt_decode(raw[CH_HEIGHT]));
        ColumnSample {
            height,
            temperature: raw[CH_TEMPERATURE]
                - TEMPERATURE_LAPSE_PER_BLOCK * (height - SEA_LEVEL_F).max(0.0),
            humidity: raw[CH_HUMIDITY],
        }
    }

    /// Drueckt Hoehen nahe dem Meeresspiegel sanft Richtung 0 (flache Straende), laesst Werte
    /// jenseits der Range linear weiterlaufen - Decode-seitiges Shaping NACH dem Blending, damit
    /// die Fenster-Synthese davon nichts wissen muss.
    fn compress_toward_sea_level(&self, height: f32) -> f32 {
        let range = self.sea_compression_range;
        let delta = height - SEA_LEVEL_F;
        let clamped = delta.clamp(-range, range);
        let excess = delta - clamped;
        SEA_LEVEL_F + signed_pow(clamped / range, self.sea_compression_exponent) * range + excess
    }

    /// Roh-Sample (Signed-Sqrt-Hoehe + Klima) einer Ebene an kontinuierlichen Weltkoordinaten -
    /// bilinear ueber die 4 umschliessenden Pixel-Zentren, jedes davon fenster-geblendet.
    fn sample_level(&self, level: usize, world_x: f32, world_z: f32) -> [f32; CHANNELS] {
        let bpp = LEVELS[level].blocks_per_pixel as f32;
        let fx = world_x / bpp - 0.5;
        let fz = world_z / bpp - 0.5;
        let gx0 = fx.floor() as i32;
        let gz0 = fz.floor() as i32;
        let tx = fx - gx0 as f32;
        let tz = fz - gz0 as f32;

        let p00 = self.blended_pixel(level, gx0, gz0);
        let p10 = self.blended_pixel(level, gx0 + 1, gz0);
        let p01 = self.blended_pixel(level, gx0, gz0 + 1);
        let p11 = self.blended_pixel(level, gx0 + 1, gz0 + 1);

        std::array::from_fn(|ch| lerp(lerp(p00[ch], p10[ch], tx), lerp(p01[ch], p11[ch], tx), tz))
    }

    /// Der InfiniteDiffusion-Kern (Gl. 2): gewichtetes Mittel der Beitraege ALLER Fenster, die den
    /// Pixel ueberlappen - bei Stride = W/2 sind das exakt 4, unabhaengig von der Position
    /// (konstantes |kappa(R)|, die Voraussetzung fuer O(1)-Random-Access, Paper B.4).
    fn blended_pixel(&self, level: usize, gx: i32, gz: i32) -> [f32; CHANNELS] {
        let i_hi = gx.div_euclid(WINDOW_STRIDE);
        let j_hi = gz.div_euclid(WINDOW_STRIDE);

        let mut accumulated = [0.0f32; CHANNELS];
        let mut weight_sum = 0.0f32;
        for wi in [i_hi - 1, i_hi] {
            for wj in [j_hi - 1, j_hi] {
                let window = self.window(level, wi, wj);
                let u = gx - wi * WINDOW_STRIDE;
                let v = gz - wj * WINDOW_STRIDE;
                let weight = triangle_weight(u) * triangle_weight(v);
                let base = ((v * WINDOW_SIZE + u) as usize) * CHANNELS;
                for ch in 0..CHANNELS {
                    accumulated[ch] += window[base + ch] * weight;
                }
                weight_sum += weight;
            }
        }
        std::array::from_fn(|ch| accumulated[ch] / weight_sum)
    }

    /// Fenster-Beschaffung in zwei Phasen: Thread-Memo (lock-frei), dann geteilter Cache, dann
    /// Neuberechnung. Der `RefCell`-Borrow wird VOR einer moeglichen Berechnung freigegeben -
    /// `compute_window` rekursiert in die Elternebene und damit wieder hierher.
    fn window(&self, level: usize, wi: i32, wj: i32) -> Arc<WindowData> {
        let slot = window_memo_slot(level, wi, wj);
        let memo_hit = WINDOW_MEMO.with_borrow(|memo| match &memo[slot] {
            Some((l, i, j, window)) if *l == level && *i == wi && *j == wj => {
                Some(Arc::clone(window))
            }
            _ => None,
        });
        if let Some(window) = memo_hit {
            return window;
        }

        let window = self.shared_window(level, wi, wj);
        WINDOW_MEMO.with_borrow_mut(|memo| {
            memo[slot] = Some((level, wi, wj, Arc::clone(&window)));
        });
        window
    }

    /// Kein Lock wird ueber die (rekursive) Berechnung gehalten - ein Rennen zweier Threads um
    /// dasselbe Fenster berechnet es doppelt, deterministisch identisch, und `entry` behaelt eines.
    fn shared_window(&self, level: usize, wi: i32, wj: i32) -> Arc<WindowData> {
        if let Some(window) = self.window_caches[level]
            .read()
            .expect("Window-Cache vergiftet")
            .get(&(wi, wj))
        {
            return Arc::clone(window);
        }

        let computed = Arc::new(self.compute_window(level, wi, wj));
        let mut map = self.window_caches[level].write().expect("Window-Cache vergiftet");
        let window = Arc::clone(map.entry((wi, wj)).or_insert(computed));
        if map.len() > WINDOW_CACHE_CAP {
            let evict: Vec<(i32, i32)> = map
                .keys()
                .filter(|key| **key != (wi, wj))
                .take(WINDOW_CACHE_CAP / 8)
                .copied()
                .collect();
            for key in evict {
                map.remove(&key);
            }
        }
        window
    }

    /// Phi dieser Ebene: synthetisiert das gepolsterte Fenster pixelweise (Basis-Synthese auf Ebene
    /// 0, sonst Eltern-Sample + parent-konditioniertes Detailband), laesst die Fenster-Operatoren
    /// (Erosion) auf dem Puffer laufen und schneidet den Apron weg.
    fn compute_window(&self, level: usize, wi: i32, wj: i32) -> WindowData {
        let spec = &LEVELS[level];
        let bpp = spec.blocks_per_pixel as f32;

        let mut height = [0.0f32; PADDED_AREA];
        let mut temperature = [0.0f32; PADDED_AREA];
        let mut humidity = [0.0f32; PADDED_AREA];

        for v in 0..PADDED_SIZE {
            for u in 0..PADDED_SIZE {
                let gx = wi * WINDOW_STRIDE - WINDOW_APRON + u;
                let gz = wj * WINDOW_STRIDE - WINDOW_APRON + v;
                let world_x = (gx as f32 + 0.5) * bpp;
                let world_z = (gz as f32 + 0.5) * bpp;
                let index = (v * PADDED_SIZE + u) as usize;

                if level == 0 {
                    height[index] = self.sample2d(&self.base, self.base_frequency, world_x, world_z)
                        * self.continental_amplitude;
                    temperature[index] =
                        self.sample2d(&self.temperature, self.climate_frequency, world_x, world_z);
                    humidity[index] =
                        self.sample2d(&self.humidity, self.climate_frequency, world_x, world_z);
                } else {
                    let parent = self.sample_level(level - 1, world_x, world_z);
                    let parent_height = signed_sqrt_decode(parent[CH_HEIGHT]);
                    height[index] =
                        parent_height + self.detail_at(level, world_x, world_z, parent_height);
                    temperature[index] = parent[CH_TEMPERATURE];
                    humidity[index] = parent[CH_HUMIDITY];
                }
            }
        }

        for _ in 0..spec.erosion_iterations {
            thermal_relaxation(&mut height, spec.blocks_per_pixel as f32);
        }

        let mut window = [0.0f32; (WINDOW_SIZE * WINDOW_SIZE) as usize * CHANNELS];
        for v in 0..WINDOW_SIZE {
            for u in 0..WINDOW_SIZE {
                let padded = ((v + WINDOW_APRON) * PADDED_SIZE + (u + WINDOW_APRON)) as usize;
                let out = ((v * WINDOW_SIZE + u) as usize) * CHANNELS;
                window[out + CH_HEIGHT] = signed_sqrt_encode(height[padded]);
                window[out + CH_TEMPERATURE] = temperature[padded];
                window[out + CH_HUMIDITY] = humidity[padded];
            }
        }
        window
    }

    /// Bandbegrenztes Detail, konditioniert auf die Elternhoehe (hierarchische Kopplung statt
    /// unabhaengiger fBm-Ueberlagerung): die Bergmaske steuert Amplitude UND den Uebergang von
    /// Smooth- zu Ridged-Charakteristik - Grate entstehen nur dort, wo die groebere Ebene bereits
    /// Gebirge etabliert hat.
    fn detail_at(&self, level: usize, world_x: f32, world_z: f32, parent_height: f32) -> f32 {
        let spec = &LEVELS[level];
        let normalized = parent_height / self.continental_amplitude.max(1.0);
        let mountain_mask =
            ((normalized + 1.0) * 0.5).clamp(0.0, 1.0).powf(self.mountain_exponent);

        let smooth = self.sample2d(
            &self.detail[level],
            1.0 / spec.detail_wavelength,
            world_x,
            world_z,
        );
        let ridged = 1.0 - 2.0 * smooth.abs();
        let shape = lerp(smooth, ridged, (mountain_mask / RIDGE_FULL_BLEND_MASK).min(1.0));
        shape * (PLAINS_ROUGHNESS + self.mountain_amplitude * mountain_mask) * spec.detail_share
    }

    #[inline]
    fn sample2d<N: NoiseFunction<Vec2, Output = f32>>(
        &self,
        noise: &Noise<N>,
        frequency: f32,
        world_x: f32,
        world_z: f32,
    ) -> f32 {
        noise.sample(Vec2::new(
            world_x * frequency + self.noise_origin_offset,
            world_z * frequency + self.noise_origin_offset,
        ))
    }
}

/// Steigungsbegrenzte Jacobi-Relaxation (Naeherung thermischer Erosion): nur Pixel, deren lokales
/// Relief die Talus-Steigung ueberschreitet, werden Richtung 4er-Nachbarschaftsmittel gezogen -
/// Schutthaenge glaetten sich, Ebenen und maessige Haenge bleiben unberuehrt. Genau die Art
/// Nachbarschafts-Operator, die die fensterbasierte Architektur erlaubt und ein reiner
/// Pro-Spalte-Generator nicht ausdruecken kann.
fn thermal_relaxation(height: &mut [f32; PADDED_AREA], blocks_per_pixel: f32) {
    let talus_drop = EROSION_TALUS_SLOPE * blocks_per_pixel;
    let mut relaxed = *height;
    for v in 1..PADDED_SIZE - 1 {
        for u in 1..PADDED_SIZE - 1 {
            let index = (v * PADDED_SIZE + u) as usize;
            let center = height[index];
            let north = height[index - PADDED_SIZE as usize];
            let south = height[index + PADDED_SIZE as usize];
            let west = height[index - 1];
            let east = height[index + 1];
            let relief = (center - north)
                .abs()
                .max((center - south).abs())
                .max((center - west).abs())
                .max((center - east).abs());
            if relief > talus_drop {
                let mean = (north + south + west + east) * 0.25;
                relaxed[index] = center + (mean - center) * EROSION_STRENGTH;
            }
        }
    }
    *height = relaxed;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pyramid() -> TerrainPyramid {
        TerrainPyramid::new(&EngineConfig::default())
    }

    #[test]
    fn signed_sqrt_roundtrips() {
        for h in [-1500.0, -3.7, 0.0, 0.25, 42.0, 900.0] {
            assert!((signed_sqrt_decode(signed_sqrt_encode(h)) - h).abs() < 1e-3);
        }
    }

    /// Seed-Konsistenz + Ordnungs-Invarianz (Paper 3.5): zwei unabhaengige Instanzen, dieselben
    /// Punkte in entgegengesetzter Reihenfolge (und damit voellig anderem Cache-/Fenster-Zustand)
    /// muessen bit-identische Ergebnisse liefern.
    #[test]
    fn queries_are_seed_consistent_and_order_invariant() {
        let a = pyramid();
        let b = pyramid();
        let points: Vec<(i32, i32)> =
            (-6..6).flat_map(|x| (-6..6).map(move |z| (x * 137, z * 211))).collect();

        let forward: Vec<f32> = points.iter().map(|&(x, z)| a.sample(x, z).height).collect();
        let backward: Vec<f32> =
            points.iter().rev().map(|&(x, z)| b.sample(x, z).height).collect();

        for (fwd, bwd) in forward.iter().zip(backward.iter().rev()) {
            assert_eq!(fwd.to_bits(), bwd.to_bits());
        }
    }

    /// Nahtlosigkeit ueber Fenster- und Ebenen-Grenzen: entlang einer langen Linie (kreuzt mehrere
    /// Strides ALLER Ebenen) darf die Hoehe zwischen Nachbarbloecken nie springen - das Blending
    /// nach Gl. 2 garantiert Stetigkeit, ein Fehler in kappa/Gewichten wuerde hier als Kante
    /// sichtbar.
    #[test]
    fn height_is_continuous_across_window_boundaries() {
        let p = pyramid();
        let mut previous = p.sample(-2048, 313).height;
        for x in -2047..2048 {
            let current = p.sample(x, 313).height;
            assert!(
                (current - previous).abs() < 12.0,
                "Hoehensprung {} bei x={x}",
                (current - previous).abs()
            );
            previous = current;
        }
    }

    /// Hoehen-Lapse: identische Klimalage, aber grosse Hoehendifferenz muss die Temperatur senken.
    #[test]
    fn temperature_drops_with_altitude() {
        let p = pyramid();
        let mut peak: Option<ColumnSample> = None;
        let mut plain: Option<ColumnSample> = None;
        for x in (-4096..4096).step_by(64) {
            let sample = p.sample(x, 0);
            if sample.height > 80.0 && peak.is_none() {
                peak = Some(sample);
            }
            if (0.0..8.0).contains(&sample.height) && plain.is_none() {
                plain = Some(sample);
            }
        }
        if let (Some(peak), Some(plain)) = (peak, plain) {
            let peak_base = peak.temperature + TEMPERATURE_LAPSE_PER_BLOCK * peak.height;
            assert!(peak.temperature < peak_base);
            assert!(plain.temperature >= plain.temperature);
        }
    }
}

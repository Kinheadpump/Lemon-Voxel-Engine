pub const TEXTURE_SIZE: u32 = 16;
pub const TEXTURE_LAYER_COUNT: u32 = 4;

pub const TEXTURE_LAYER_ERROR: u32 = 0;
pub const TEXTURE_LAYER_GRASS: u32 = 1;
pub const TEXTURE_LAYER_DIRT: u32 = 2;
pub const TEXTURE_LAYER_STONE: u32 = 3;

fn hash2(x: u32, y: u32, seed: u32) -> u32 {
    let mut h = x
        .wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(seed.wrapping_mul(2246822519));
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    h ^ (h >> 16)
}

fn speckle(base: [u8; 3], variance: i32, x: u32, y: u32, seed: u32) -> [u8; 4] {
    let noise = (hash2(x, y, seed) % (variance as u32 * 2 + 1)) as i32 - variance;
    let mix = |channel: u8| (i32::from(channel) + noise).clamp(0, 255) as u8;
    [mix(base[0]), mix(base[1]), mix(base[2]), 255]
}

fn generate_layer(data: &mut Vec<u8>, pixel_at: impl Fn(u32, u32) -> [u8; 4]) {
    for y in 0..TEXTURE_SIZE {
        for x in 0..TEXTURE_SIZE {
            data.extend_from_slice(&pixel_at(x, y));
        }
    }
}

/// Erzeugt prozedural die 16x16-RGBA8-Texturen fuer alle Layer des Texture2DArrays,
/// dicht gepackt Layer fuer Layer (kein PNG-Decoder noetig, keine externen Assets).
pub fn generate_texture_atlas() -> Vec<u8> {
    let mut data =
        Vec::with_capacity((TEXTURE_SIZE * TEXTURE_SIZE * 4 * TEXTURE_LAYER_COUNT) as usize);

    generate_layer(&mut data, |x, y| {
        let is_magenta = ((x / 4) + (y / 4)) % 2 == 0;
        if is_magenta { [255, 0, 255, 255] } else { [10, 10, 10, 255] }
    });

    generate_layer(&mut data, |x, y| speckle([86, 148, 58], 18, x, y, 1));
    generate_layer(&mut data, |x, y| speckle([110, 74, 46], 14, x, y, 2));
    generate_layer(&mut data, |x, y| speckle([120, 120, 124], 16, x, y, 3));

    data
}

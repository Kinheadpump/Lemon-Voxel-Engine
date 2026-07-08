use crate::engine::render::textures::{
    TEXTURE_LAYER_DIRT, TEXTURE_LAYER_GRASS, TEXTURE_LAYER_SAND, TEXTURE_LAYER_STONE,
};

pub const AIR: u16 = 0;
pub const GRASS: u16 = TEXTURE_LAYER_GRASS as u16;
pub const DIRT: u16 = TEXTURE_LAYER_DIRT as u16;
pub const STONE: u16 = TEXTURE_LAYER_STONE as u16;
pub const SAND: u16 = TEXTURE_LAYER_SAND as u16;

/// Bestimmt die Block-ID einer Oberflaechen-Saeule aus Tiefe unter der Oberflaeche, lokaler
/// Hangneigung (max. Hoehenunterschied zu den 4 Nachbar-Saeulen) und Strandzugehoerigkeit. Die
/// Erd-/Gras-Schicht wird mit steigender Neigung graduell duenner statt hart zwischen Erde und Fels
/// umzuschalten - an steilen Klippen (`slope >= dirt_depth`) bleibt so blanker Fels stehen.
pub fn surface_block(depth_from_surface: i32, slope: i32, dirt_depth: i32, is_beach: bool) -> u16 {
    let eroded_dirt_depth = (dirt_depth - slope).max(0);

    if is_beach && depth_from_surface < dirt_depth {
        return SAND;
    }
    if depth_from_surface == 0 && eroded_dirt_depth > 0 {
        GRASS
    } else if depth_from_surface < eroded_dirt_depth {
        DIRT
    } else {
        STONE
    }
}

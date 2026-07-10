use crate::engine::render::textures::{
    TEXTURE_LAYER_DIRT, TEXTURE_LAYER_GRASS, TEXTURE_LAYER_LEAVES, TEXTURE_LAYER_LOG, TEXTURE_LAYER_SAND,
    TEXTURE_LAYER_STONE, TEXTURE_LAYER_WATER,
};

pub const AIR: u16 = 0;
pub const GRASS: u16 = TEXTURE_LAYER_GRASS as u16;
pub const DIRT: u16 = TEXTURE_LAYER_DIRT as u16;
pub const STONE: u16 = TEXTURE_LAYER_STONE as u16;
pub const SAND: u16 = TEXTURE_LAYER_SAND as u16;
pub const WATER: u16 = TEXTURE_LAYER_WATER as u16;
pub const LOG: u16 = TEXTURE_LAYER_LOG as u16;
pub const LEAVES: u16 = TEXTURE_LAYER_LEAVES as u16;

/// Oberflaechen-Kontext einer Spalte - einmal pro Spalte bestimmt (Biom-Rauschen, Hoehenband),
/// dann fuer alle Voxel der Spalte wiederverwendet.
#[derive(Clone, Copy)]
pub struct ColumnSurface {
    /// Spaltenhoehe liegt im Strand-Hoehenband um den Wasserspiegel.
    pub is_beach: bool,
    /// Spaltenoberflaeche liegt UNTER dem Wasserspiegel (Ozean-/Seeboden).
    pub is_underwater: bool,
    /// Wuesten-Biom: heiss UND trocken (striktes 2D-Temperatur/Feuchtigkeits-Mapping, s.
    /// `TerrainGenerator::column_surface`) - Sand statt Gras/Erde, unabhaengig vom Hoehenband.
    pub is_desert: bool,
    /// Hochgebirge: Spaltenhoehe ueber der Fels-Grenze - nackter Stein statt Gras/Erde, damit hohe
    /// Gipfel nicht komplett begruent wirken.
    pub is_rock: bool,
    /// Rohes Temperatur-Sample (snorm, ungefaehr -1..1) - bereits fuer den Wuesten-Check berechnet,
    /// hier zusaetzlich exponiert, damit `flora.rs` daraus die Baumart waehlen kann (z.B. Tanne in
    /// kalten Regionen), ohne eine zweite Rauschprobe an derselben Position zu verschwenden.
    pub temperature: f32,
}

/// Bestimmt die Block-ID einer Oberflaechen-Saeule aus Tiefe unter der Oberflaeche, lokaler
/// Hangneigung (max. Hoehenunterschied zu den 4 Nachbar-Saeulen) und dem Spalten-Kontext. Die
/// Deckschicht wird mit steigender Neigung graduell duenner statt hart umzuschalten - an steilen
/// Klippen (`slope >= dirt_depth`) bleibt blanker Fels stehen. Prioritaet der Deckschicht-Regeln:
/// Hochgebirge (Fels) > Unterwasser (nie Gras - Ozeanboden ist Sand am Ufer, sonst Erde) >
/// Strand/Wueste (Sand) > Gras/Erde.
pub fn surface_block(depth_from_surface: i32, slope: i32, dirt_depth: i32, surface: ColumnSurface) -> u16 {
    if surface.is_rock {
        return STONE;
    }

    let eroded_dirt_depth = (dirt_depth - slope).max(0);
    let in_top_layer = depth_from_surface < dirt_depth;

    if surface.is_underwater && in_top_layer {
        return if surface.is_beach { SAND } else { DIRT };
    }
    if (surface.is_beach || surface.is_desert) && in_top_layer {
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

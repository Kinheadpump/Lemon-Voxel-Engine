use super::*;

/// Sicherheitsmarge (Weltbloecke) fuer den Luft-Chunk-Fruehausstieg in `generate_chunk`: eine
/// Baumkrone aus einer NACHBAR-Spalte (innerhalb des Baum-Suchradius, aber ausserhalb der eigenen
/// 1024 Saeulen dieses Chunks) kann in einen Chunk hineinragen, dessen eigene lokale Saeulen alle
/// niedriger liegen - besonders an Klippenkanten, wo die Regional-Kontrast-Karte auf wenigen
/// Bloecken Distanz stark springen kann. Grosszuegig aus der maximalen Regional-Amplitude (Default
/// 22, Range ca. +-44) plus Baum-Reichweite bemessen, nicht aus der exakten Baumhoehe abgeleitet -
/// eine zu knappe Marge waere ein stiller Cross-Chunk-Bug (fehlende Blattvoxel), eine zu grosszuegige
/// kostet nur ein paar zusaetzliche, aber billige Leerlauf-Chunks.
pub(super) const TREE_HEIGHT_SAFETY_MARGIN: i32 = 48;

/// Deterministisches White-Noise-Hash fuer die Baum-Spawn-Entscheidung: reine Bit-Mischung ueber
/// (Gitterzelle, Seed, Salt) - KEIN Gradientenrauschen (`noiz`), weil Spawn-Entscheidungen diskret
/// sind (existiert der Baum in dieser Zelle oder nicht) und keine raeumliche Kontinuitaet zwischen
/// Nachbarzellen brauchen. `salt` unterscheidet die verschiedenen pro Zelle abgeleiteten Werte
/// (Jitter-X, Jitter-Z, Spawn-Wuerfel, Stammhoehe, Kronenradius) - dieselbe Zellkoordinate liefert
/// pro Salt einen unabhaengigen, aber ueber Aufrufe hinweg REPRODUZIERBAREN Wert.
#[inline(always)]
fn tree_hash(seed: u32, cell_x: i32, cell_z: i32, salt: u32) -> u32 {
    let mut h = (cell_x as u32).wrapping_mul(0x9E37_79B1)
        ^ (cell_z as u32).wrapping_mul(0x85EB_CA6B)
        ^ seed.wrapping_mul(0xC2B2_AE35)
        ^ salt.wrapping_mul(0x27D4_EB2F);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x2971_25AC);
    h ^ (h >> 15)
}

/// Bildet einen Hash auf `[0, 1)` ab (obere 24 Bit fuer gleichmaessige Streuung).
#[inline(always)]
fn hash_unit(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Deterministisch berechnete Struktur EINES Baum-Spawns - s. `TerrainGenerator::tree_candidate`.
#[derive(Clone, Copy)]
pub(super) struct TreeSpawn {
    pub(super) world_x: i32,
    pub(super) world_z: i32,
    /// Topmost solider Block der Spawn-Saeule (== `height_at`) - Stamm beginnt bei `ground_y + 1`.
    pub(super) ground_y: i32,
    pub(super) trunk_height: i32,
    pub(super) crown_radius: i32,
}

/// Obergrenze gleichzeitig im Suchradius gefundener Baum-Kandidaten - bei realistischen
/// Spawn-Chancen (< 20%) liegen es fast immer 0-2, hier grosszuegig bemessen fuer extreme
/// Test-Configs (`terrain_tree_spawn_chance = 1.0`). Ueberzaehlige Kandidaten (praktisch nie im
/// echten Spiel) werden schlicht ignoriert statt zu reallozieren.
pub(super) const MAX_NEARBY_TREES: usize = 16;
const EMPTY_TREE_SPAWN: TreeSpawn = TreeSpawn { world_x: 0, world_z: 0, ground_y: 0, trunk_height: 0, crown_radius: 0 };

impl TerrainGenerator {
    /// Sammelt alle Baum-Kandidaten, die eine Position bei `(world_x, world_z)` ueberhaupt
    /// erreichen KOENNTEN (unabhaengig von Y) - der Suchradius haengt nur von X/Z ab, nicht von Y.
    /// Getrennt von `tree_occupies` extrahiert, damit Aufrufer mit einer FESTEN X- oder Z-Achse
    /// (die drei `solid_plane_*`) diese Suche EINMAL pro Spalte statt einmal pro Voxel durchfuehren
    /// koennen - ohne dieses Batching wiederholte eine 32-Bloecke-hohe Rand-Ebenen-Spalte dieselbe
    /// Gitterzellen-Suche bis zu 32x, s. Kommentar an `solid_plane_x`.
    #[cold]
    pub(super) fn nearby_tree_candidates(&self, world_x: i32, world_z: i32) -> ([TreeSpawn; MAX_NEARBY_TREES], usize) {
        let search_radius = self.tree_grid_size + self.tree_crown_radius_max;
        let cell_min_x = (world_x - search_radius).div_euclid(self.tree_grid_size);
        let cell_max_x = (world_x + search_radius).div_euclid(self.tree_grid_size);
        let cell_min_z = (world_z - search_radius).div_euclid(self.tree_grid_size);
        let cell_max_z = (world_z + search_radius).div_euclid(self.tree_grid_size);

        let mut trees = [EMPTY_TREE_SPAWN; MAX_NEARBY_TREES];
        let mut count = 0;
        'search: for cell_z in cell_min_z..=cell_max_z {
            for cell_x in cell_min_x..=cell_max_x {
                if let Some(tree) = self.tree_candidate(cell_x, cell_z) {
                    trees[count] = tree;
                    count += 1;
                    if count >= MAX_NEARBY_TREES {
                        break 'search;
                    }
                }
            }
        }
        (trees, count)
    }

    /// Prueft eine bereits gesammelte Kandidatenliste (s. `nearby_tree_candidates`) gegen EINEN
    /// Punkt - exakt dieselbe Trunk-Saeule/Kronen-Kugel-Geometrie wie `place_flora`, MUSS mit ihr
    /// uebereinstimmen, sonst genau die Bulk/Fallback-Divergenz, die `is_carved` schon einmal
    /// hatte (s. dortiger Kommentar).
    #[cold]
    pub(super) fn tree_occupies_among(trees: &[TreeSpawn], world_x: i32, world_y: i32, world_z: i32) -> bool {
        for tree in trees {
            let trunk_top = tree.ground_y + tree.trunk_height;

            if world_x == tree.world_x && world_z == tree.world_z && world_y > tree.ground_y && world_y <= trunk_top {
                return true;
            }

            let dx = world_x - tree.world_x;
            let dy = world_y - trunk_top;
            let dz = world_z - tree.world_z;
            if dx * dx + dy * dy + dz * dz <= tree.crown_radius * tree.crown_radius {
                return true;
            }
        }
        false
    }

    /// Einzelpunkt-Fallback fuer Baum-Belegung ("steht an dieser Weltposition Stamm oder Krone
    /// irgendeines Baumes?"). Genutzt von `is_solid`/`is_physically_solid`, die (anders als der
    /// Bulk-Pfad `place_flora`) kein lokales Chunk-Array haben, aus dem sie einfach lesen koennten.
    #[cold]
    pub(super) fn tree_occupies(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        let (trees, count) = self.nearby_tree_candidates(world_x, world_z);
        Self::tree_occupies_among(&trees[..count], world_x, world_y, world_z)
    }

    /// Deterministisch berechneter Baum-Spawn-Kandidat fuer EINE Gitterzelle des
    /// `tree_grid_size`-Rasters, oder `None`, wenn die Zelle keinen Baum traegt (Spawn-Wuerfel
    /// verfehlt ODER Untergrund ist Wasser/Strand/Wueste/Fels). Rein aus (`tree_seed`,
    /// `cell_x`, `cell_z`) berechnet - IDENTISCHES Ergebnis unabhaengig davon, welcher Chunk gerade
    /// generiert wird ("Pure Function"-Prinzip, keine Chunk-Nachbar-Reads noetig).
    #[cold]
    pub(super) fn tree_candidate(&self, cell_x: i32, cell_z: i32) -> Option<TreeSpawn> {
        if hash_unit(tree_hash(self.tree_seed, cell_x, cell_z, 2)) >= self.tree_spawn_chance {
            return None;
        }

        let jitter_x = (hash_unit(tree_hash(self.tree_seed, cell_x, cell_z, 0)) * self.tree_grid_size as f32) as i32;
        let jitter_z = (hash_unit(tree_hash(self.tree_seed, cell_x, cell_z, 1)) * self.tree_grid_size as f32) as i32;
        let world_x = cell_x * self.tree_grid_size + jitter_x;
        let world_z = cell_z * self.tree_grid_size + jitter_z;

        let ground_y = self.height_at(world_x, world_z);
        let surface = self.column_surface(world_x, world_z, ground_y);
        if surface.is_underwater || surface.is_beach || surface.is_desert || surface.is_rock {
            return None;
        }

        let trunk_span = (self.tree_trunk_height_max - self.tree_trunk_height_min + 1).max(1);
        let trunk_height = self.tree_trunk_height_min
            + (hash_unit(tree_hash(self.tree_seed, cell_x, cell_z, 3)) * trunk_span as f32) as i32;
        let crown_span = (self.tree_crown_radius_max - self.tree_crown_radius_min + 1).max(1);
        let crown_radius = self.tree_crown_radius_min
            + (hash_unit(tree_hash(self.tree_seed, cell_x, cell_z, 4)) * crown_span as f32) as i32;

        Some(TreeSpawn { world_x, world_z, ground_y, trunk_height, crown_radius })
    }

    /// Cross-Chunk-Baumplatzierung: iteriert nicht nur die eigenen 32x32 Saeulen, sondern das
    /// Spawn-Gitter ueber einen um `tree_grid_size + tree_crown_radius_max` VERGROESSERTEN
    /// X/Z-Radius (deckt jede Zelle ab, deren Jitter+Krone theoretisch in diesen Chunk reichen
    /// kann). Fuer jeden gefundenen Kandidaten wird die VOLLE Baumstruktur berechnet, aber nur der
    /// Teil, der tatsaechlich in diesen Chunk faellt, lokal geschrieben (`place_tree_voxel`
    /// klemmt) - jeder betroffene Chunk kommt so unabhaengig zum selben Ergebnis, ohne je Daten
    /// eines Nachbar-Chunks zu lesen oder zu senden.
    #[cold]
    pub(super) fn place_flora(&self, chunk_x: i32, chunk_y: i32, chunk_z: i32, chunk: &mut Chunk) {
        let chunk_origin_x = chunk_x * CHUNK_SIZE;
        let chunk_origin_y = chunk_y * CHUNK_SIZE;
        let chunk_origin_z = chunk_z * CHUNK_SIZE;

        let search_radius = self.tree_grid_size + self.tree_crown_radius_max;
        let cell_min_x = (chunk_origin_x - search_radius).div_euclid(self.tree_grid_size);
        let cell_max_x = (chunk_origin_x + CHUNK_SIZE - 1 + search_radius).div_euclid(self.tree_grid_size);
        let cell_min_z = (chunk_origin_z - search_radius).div_euclid(self.tree_grid_size);
        let cell_max_z = (chunk_origin_z + CHUNK_SIZE - 1 + search_radius).div_euclid(self.tree_grid_size);

        for cell_z in cell_min_z..=cell_max_z {
            for cell_x in cell_min_x..=cell_max_x {
                let Some(tree) = self.tree_candidate(cell_x, cell_z) else { continue };

                let trunk_top = tree.ground_y + tree.trunk_height;
                let footprint_min_y = tree.ground_y + 1;
                let footprint_max_y = trunk_top + tree.crown_radius;
                // Billige AABB-Ablehnung, bevor die Voxel des Baumes einzeln durchlaufen werden -
                // die meisten Kandidaten im vergroesserten Suchradius treffen diesen Chunk gar
                // nicht (weder horizontal noch vertikal).
                if footprint_max_y < chunk_origin_y
                    || footprint_min_y > chunk_origin_y + CHUNK_SIZE - 1
                    || tree.world_x + tree.crown_radius < chunk_origin_x
                    || tree.world_x - tree.crown_radius > chunk_origin_x + CHUNK_SIZE - 1
                    || tree.world_z + tree.crown_radius < chunk_origin_z
                    || tree.world_z - tree.crown_radius > chunk_origin_z + CHUNK_SIZE - 1
                {
                    continue;
                }

                // Stamm: gerader Holzstamm vom Boden bis zur Kronenbasis.
                for world_y in footprint_min_y..=trunk_top {
                    Self::place_tree_voxel(
                        chunk,
                        chunk_origin_x,
                        chunk_origin_y,
                        chunk_origin_z,
                        tree.world_x,
                        world_y,
                        tree.world_z,
                        blocks::LOG,
                    );
                }

                // Krone: einfache Kugel um die Stammspitze (spaeter durch Space Colonization
                // ersetzbar, ohne dass sich am Cross-Chunk-Mechanismus etwas aendert).
                let radius_sq = (tree.crown_radius * tree.crown_radius) as f32;
                for dy in -tree.crown_radius..=tree.crown_radius {
                    for dz in -tree.crown_radius..=tree.crown_radius {
                        for dx in -tree.crown_radius..=tree.crown_radius {
                            if (dx * dx + dy * dy + dz * dz) as f32 > radius_sq {
                                continue;
                            }
                            Self::place_tree_voxel(
                                chunk,
                                chunk_origin_x,
                                chunk_origin_y,
                                chunk_origin_z,
                                tree.world_x + dx,
                                trunk_top + dy,
                                tree.world_z + dz,
                                blocks::LEAVES,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Schreibt EINEN Baum-Voxel nur, wenn er in den aktuellen Chunk faellt (lokales Clamping) UND
    /// die Zielposition dort noch Luft ist - so ueberschreibt ein Baum weder Terrain (z.B. eine
    /// hoehere Nachbarsaeule, die in den Kronenradius hineinragt) noch bereits platzierte Voxel
    /// eines anderen Baumes.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn place_tree_voxel(
        chunk: &mut Chunk,
        chunk_origin_x: i32,
        chunk_origin_y: i32,
        chunk_origin_z: i32,
        world_x: i32,
        world_y: i32,
        world_z: i32,
        block_id: u16,
    ) {
        let local_x = world_x - chunk_origin_x;
        let local_y = world_y - chunk_origin_y;
        let local_z = world_z - chunk_origin_z;
        if !(0..CHUNK_SIZE).contains(&local_x) || !(0..CHUNK_SIZE).contains(&local_y) || !(0..CHUNK_SIZE).contains(&local_z)
        {
            return;
        }
        if chunk.get_block(local_x, local_y, local_z) != 0 {
            return;
        }
        chunk.set_block(local_x, local_y, local_z, block_id);
    }
}

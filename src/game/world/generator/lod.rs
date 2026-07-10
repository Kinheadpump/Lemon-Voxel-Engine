use super::*;

impl TerrainGenerator {
    /// Guenstiger Okklusions-Fallback fuer LOD-Chunks (Mesher-Randabfrage bei `voxel_scale > 1`):
    /// nur Hoehe + Wasser, KEINE Hoehlen/Baeume - s. `generate_lod_chunk`-Kommentar zur Begruendung.
    /// `world_*` sind bereits echte Weltkoordinaten (Aufrufer multipliziert lokale Mesher-Koordinaten
    /// vorher mit `voxel_scale`).
    pub fn is_solid_lod(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        let height = self.height_at(world_x, world_z);
        if Self::is_water_position(height, world_y) {
            return true;
        }
        world_y <= height
    }

    /// LOD-Variante von `generate_chunk`: EIN Voxel deckt `voxel_scale` Weltbloecke pro Achse ab
    /// (Chunk-Koordinate UND lokaler Schritt skalieren gleichermassen). Hoehe wird weiterhin EXAKT
    /// pro Spalte gesampelt (kein Interpolieren zwischen Spalten - dieselbe Falle, die
    /// `height_at`s Git-Historie schon einmal zeigte), aber BEIDE Hoehlensysteme UND Baeume
    /// entfallen komplett: Hoehlen-Innenraeume sind aus der Distanz ohnehin nie sichtbar und mit
    /// Abstand der teuerste Teil der Generierung, Baeume sind bei `voxel_scale > 1` Sub-Voxel-Detail.
    /// Das macht LOD-Chunks nicht nur einfacher, sondern massiv billiger als LOD0 - genau richtig,
    /// weil es von ihnen absolut mehr gibt (siehe Ring-Radien in `EngineConfig`).
    pub fn generate_lod_chunk(&self, chunk_x: i32, chunk_y: i32, chunk_z: i32, voxel_scale: i32, chunk: &mut Chunk) {
        chunk.clear();

        let chunk_origin_x = chunk_x * CHUNK_SIZE * voxel_scale;
        let chunk_origin_y = chunk_y * CHUNK_SIZE * voxel_scale;
        let chunk_origin_z = chunk_z * CHUNK_SIZE * voxel_scale;

        let mut local_height = [0i32; (CHUNK_SIZE * CHUNK_SIZE) as usize];
        let mut chunk_max_height = i32::MIN;
        for local_z in 0..CHUNK_SIZE {
            for local_x in 0..CHUNK_SIZE {
                let world_x = chunk_origin_x + local_x * voxel_scale;
                let world_z = chunk_origin_z + local_z * voxel_scale;
                let h = self.height_at(world_x, world_z);
                local_height[(local_z * CHUNK_SIZE + local_x) as usize] = h;
                chunk_max_height = chunk_max_height.max(h);
            }
        }

        // Chunk liegt vollstaendig ueber Terrainoberflaeche UND Wasserspiegel - reine Luft,
        // `chunk.clear()` oben reicht bereits. Anders als `generate_chunk` keine Baumkronen-
        // Sicherheitsmarge noetig (LOD-Chunks platzieren nie Baeume).
        if chunk_origin_y > chunk_max_height && chunk_origin_y > WATER_LEVEL {
            return;
        }

        let height_lookup = |local_x: i32, local_z: i32| -> i32 {
            if (0..CHUNK_SIZE).contains(&local_x) && (0..CHUNK_SIZE).contains(&local_z) {
                local_height[(local_z * CHUNK_SIZE + local_x) as usize]
            } else {
                self.height_at(chunk_origin_x + local_x * voxel_scale, chunk_origin_z + local_z * voxel_scale)
            }
        };

        let chunk_top_y = chunk_origin_y + CHUNK_SIZE * voxel_scale - voxel_scale;

        for local_z in 0..CHUNK_SIZE {
            for local_x in 0..CHUNK_SIZE {
                let height = local_height[(local_z * CHUNK_SIZE + local_x) as usize];
                let column_has_terrain = chunk_origin_y <= height;
                let column_has_water = height < WATER_LEVEL && chunk_origin_y <= WATER_LEVEL;
                if !column_has_terrain && !column_has_water {
                    continue;
                }

                let world_x = chunk_origin_x + local_x * voxel_scale;
                let world_z = chunk_origin_z + local_z * voxel_scale;

                // Wasserfuellung zuerst - billig (kein Rauschen) und unabhaengig von Hangneigung/
                // Biom, exakt wie im LOD0-Pfad.
                if column_has_water {
                    let water_bottom = (height + voxel_scale).max(chunk_origin_y);
                    let water_top = WATER_LEVEL.min(chunk_top_y);
                    let mut world_y = water_bottom;
                    while world_y <= water_top {
                        let local_y = (world_y - chunk_origin_y) / voxel_scale;
                        chunk.set_block(local_x, local_y, local_z, blocks::WATER);
                        world_y += voxel_scale;
                    }
                }
                if !column_has_terrain {
                    continue;
                }

                let slope = (height - height_lookup(local_x - 1, local_z))
                    .abs()
                    .max((height - height_lookup(local_x + 1, local_z)).abs())
                    .max((height - height_lookup(local_x, local_z - 1)).abs())
                    .max((height - height_lookup(local_x, local_z + 1)).abs());
                let surface = self.column_surface(world_x, world_z, height);

                let mut world_y = chunk_origin_y;
                let mut local_y = 0;
                while local_y < CHUNK_SIZE {
                    if world_y > height {
                        break;
                    }
                    let depth_from_surface = height - world_y;
                    let block_id = blocks::surface_block(depth_from_surface, slope, self.dirt_layer_depth, surface);
                    chunk.set_block(local_x, local_y, local_z, block_id);
                    world_y += voxel_scale;
                    local_y += 1;
                }
            }
        }
    }
}

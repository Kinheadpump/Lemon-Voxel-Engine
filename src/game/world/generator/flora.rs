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

/// Deterministisches White-Noise-Hash fuer die Baum-Spawn-Entscheidung UND fuer das Space-
/// Colonization-Wachstum selbst: reine Bit-Mischung ueber (Zellkoordinate, Seed, Salt) - KEIN
/// Gradientenrauschen (`noiz`), weil sowohl Spawn-Entscheidungen als auch Attraktorpunkte diskret/
/// unabhaengig sind und keine raeumliche Kontinuitaet zwischen Nachbarwerten brauchen. `salt`
/// unterscheidet die verschiedenen pro Zelle abgeleiteten Werte (Jitter, Spawn-Wuerfel, Stammhoehe,
/// Kronenradius, Attraktorpunkte) - dieselbe Zellkoordinate liefert pro Salt einen unabhaengigen,
/// aber ueber Aufrufe hinweg REPRODUZIERBAREN Wert.
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

/// Baumart, aus dem Biom (Temperatur) der Spawn-Saeule abgeleitet - jede Art hat ihre eigene
/// Wachstumshuelle (s. `sample_envelope_point`) und Kronen-/Ast-Proportionen, aber teilt sich
/// denselben Space-Colonization-Algorithmus. Weitere Arten (z.B. Palme in Wuesten) lassen sich
/// spaeter rein durch eine weitere `sample_envelope_point`-Huelle ergaenzen.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TreeSpecies {
    /// Rundliche/lockere Krone (Laubbaum-Charakter) - Attraktoren in einer Kugel ueber der
    /// Stammspitze.
    Round,
    /// Schmale, hoch aufragende Kronen-Huelle (Nadelbaum-Charakter) - Attraktoren in einem Kegel,
    /// der zur Spitze hin auslaeuft.
    Pine,
}

/// Unterhalb dieser (rohen, ungefaehr -1..1) Temperatur waechst `TreeSpecies::Pine` statt
/// `TreeSpecies::Round` - simple Median-Schwelle ohne weitere Kalibrierung, da beide Kronenformen
/// bei jedem Schwellwert plausibel aussehen (kein "falscher" Wert wie bei den Hoehlen-Schwellen,
/// wo eine falsche Zahl das halbe Volumen aushoehlen wuerde).
const PINE_TEMPERATURE_THRESHOLD: f32 = 0.0;

/// Anzahl Gitter-Knoten (Wurzel bei der Stammspitze + Wachstumssegmente) des Space-Colonization-
/// Skeletts EINES Baumes - klein gehalten (nicht "jeder Zweig"), damit `TreeSpawn` klein genug
/// bleibt, um in den pro-Spalte gecachten Kandidatenlisten der `solid_plane_*`-Funktionen zu
/// stehen (bis zu `MAX_NEARBY_TREES` Baeume * 32 Spalten pro Ebene).
const MAX_TREE_NODES: usize = 9;
/// Anzahl deterministisch gesampelter Attraktorpunkte in der artspezifischen Wachstumshuelle -
/// mehr Punkte = dichteres/gleichmaessigeres Astwerk, aber mehr Rechenaufwand pro Baum-Kandidat.
const SCA_ATTRACTOR_COUNT: usize = 20;
/// Obergrenze an Wachstumsiterationen - jede Iteration fuegt hoechstens einen neuen Knoten pro
/// bereits vorhandenem, noch von Attraktoren beeinflusstem Knoten hinzu. Wachstum stoppt vorher
/// automatisch, sobald `MAX_TREE_NODES` erreicht ist oder keine Attraktoren mehr wirken.
const SCA_MAX_ITERATIONS: usize = 8;

/// Deterministisch berechnete Struktur EINES Baum-Spawns inklusive seines vollstaendig
/// vorberechneten Space-Colonization-Skeletts - s. `TerrainGenerator::tree_candidate`.
#[derive(Clone, Copy)]
pub(super) struct TreeSpawn {
    pub(super) world_x: i32,
    pub(super) world_z: i32,
    /// Topmost solider Block der Spawn-Saeule (== `height_at`) - Stamm beginnt bei `ground_y + 1`.
    pub(super) ground_y: i32,
    pub(super) trunk_height: i32,
    pub(super) crown_radius: i32,
    pub(super) species: TreeSpecies,
    /// Skelett-Knoten RELATIV zur Stammspitze (`ground_y + trunk_height`, an Weltposition
    /// `(world_x, world_z)`) - Knoten 0 ist immer die Wurzel (liegt exakt auf der Stammspitze).
    pub(super) nodes: [glam::Vec3; MAX_TREE_NODES],
    /// Elternindex jedes Knotens (Knoten 0 hat keinen Elternteil, Wert dort unbenutzt) - ein
    /// Aststueck ist die Strecke `nodes[i] -> nodes[parents[i]]` fuer `i in 1..node_count`.
    pub(super) parents: [i8; MAX_TREE_NODES],
    pub(super) node_count: u8,
}

/// Obergrenze gleichzeitig im Suchradius gefundener Baum-Kandidaten - bei realistischen
/// Spawn-Chancen (< 20%) liegen es fast immer 0-2, hier grosszuegig bemessen fuer extreme
/// Test-Configs (`terrain_tree_spawn_chance = 1.0`). Ueberzaehlige Kandidaten (praktisch nie im
/// echten Spiel) werden schlicht ignoriert statt zu reallozieren.
pub(super) const MAX_NEARBY_TREES: usize = 10;
const EMPTY_TREE_SPAWN: TreeSpawn = TreeSpawn {
    world_x: 0,
    world_z: 0,
    ground_y: 0,
    trunk_height: 0,
    crown_radius: 0,
    species: TreeSpecies::Round,
    nodes: [glam::Vec3::ZERO; MAX_TREE_NODES],
    parents: [-1; MAX_TREE_NODES],
    node_count: 0,
};

/// Sampelt EINEN deterministischen Punkt in der artspezifischen Wachstumshuelle, relativ zur
/// Stammspitze - `index` unterscheidet die `SCA_ATTRACTOR_COUNT` Punkte, `salt_base` haelt sie
/// unabhaengig von den Spawn-Entscheidungs-Hashes derselben Zelle.
fn sample_envelope_point(species: TreeSpecies, seed: u32, cell_x: i32, cell_z: i32, index: u32, crown_radius: f32) -> glam::Vec3 {
    let u1 = hash_unit(tree_hash(seed, cell_x, cell_z, 20 + index * 3));
    let u2 = hash_unit(tree_hash(seed, cell_x, cell_z, 21 + index * 3));
    let u3 = hash_unit(tree_hash(seed, cell_x, cell_z, 22 + index * 3));

    match species {
        // Gleichverteilung IM VOLUMEN einer Kugel (nicht nur auf der Oberflaeche): Radius mit
        // Kubikwurzel skaliert, Winkel ueblich ueber Kugelkoordinaten. Vertikal leicht gestaucht
        // und nach oben verschoben, damit die Krone sichtbar UEBER der Stammspitze sitzt statt sie
        // zu umschliessen - ein Laubbaum haengt nicht symmetrisch um den Stammansatz herum.
        TreeSpecies::Round => {
            let theta = u1 * std::f32::consts::TAU;
            let phi = (2.0 * u2 - 1.0).clamp(-1.0, 1.0).acos();
            let r = crown_radius * u3.cbrt();
            let x = r * phi.sin() * theta.cos();
            let y = r * phi.cos() * 0.75;
            let z = r * phi.sin() * theta.sin();
            glam::Vec3::new(x, y + crown_radius * 0.5, z)
        }
        // Gleichverteilung in einem Kegel: Hoehe linear gesampelt, Radius an dieser Hoehe linear
        // zur Spitze hin ausgeduennt (Kegel-Silhouette), innerhalb der Scheibe per Wurzel-Sampling
        // gleichverteilt (verhindert Haeufung im Zentrum).
        TreeSpecies::Pine => {
            let cone_height = crown_radius * 2.6;
            let y = cone_height * u1;
            let radius_at_y = (crown_radius * (1.0 - y / cone_height)).max(0.05);
            let theta = u2 * std::f32::consts::TAU;
            let r = radius_at_y * u3.sqrt();
            glam::Vec3::new(r * theta.cos(), y, r * theta.sin())
        }
    }
}

/// Space Colonization Algorithm: waechst deterministisch ein kleines Astskelett aus der
/// Stammspitze (Knoten 0, Ursprung) in Richtung der Attraktorpunkte. Klassischer SCA-Kern (Karwowski
/// & Prusinkiewicz): pro Iteration sucht jeder ueberlebende Attraktor seinen naechsten Knoten
/// INNERHALB des Einflussradius, jeder so getroffene Knoten waechst EINEN Schritt in die gemittelte
/// Richtung seiner Attraktoren, danach sterben alle Attraktoren innerhalb des Toetungsradius um
/// IRGENDEINEN Knoten. Bricht ab, sobald `MAX_TREE_NODES` erreicht ist oder eine Iteration keinen
/// einzigen neuen Knoten mehr erzeugt (alle verbleibenden Attraktoren ausser Reichweite).
fn grow_skeleton(species: TreeSpecies, seed: u32, cell_x: i32, cell_z: i32, crown_radius: f32) -> ([glam::Vec3; MAX_TREE_NODES], [i8; MAX_TREE_NODES], u8) {
    let mut attractors = [glam::Vec3::ZERO; SCA_ATTRACTOR_COUNT];
    let mut attractor_alive = [true; SCA_ATTRACTOR_COUNT];
    for (i, attractor) in attractors.iter_mut().enumerate() {
        *attractor = sample_envelope_point(species, seed, cell_x, cell_z, i as u32, crown_radius);
    }

    let influence_radius = crown_radius * 2.0;
    let kill_radius = crown_radius * 0.4;
    let segment_length = crown_radius * 0.45;

    let mut nodes = [glam::Vec3::ZERO; MAX_TREE_NODES];
    let mut parents = [-1i8; MAX_TREE_NODES];
    let mut node_count: usize = 1;

    for _ in 0..SCA_MAX_ITERATIONS {
        if node_count >= MAX_TREE_NODES {
            break;
        }

        let mut growth_dir = [glam::Vec3::ZERO; MAX_TREE_NODES];
        let mut growth_count = [0u32; MAX_TREE_NODES];

        for (a, &attractor) in attractors.iter().enumerate() {
            if !attractor_alive[a] {
                continue;
            }
            let mut nearest = usize::MAX;
            let mut nearest_dist = influence_radius;
            for (n, &node) in nodes.iter().enumerate().take(node_count) {
                let d = node.distance(attractor);
                if d < nearest_dist {
                    nearest_dist = d;
                    nearest = n;
                }
            }
            if nearest != usize::MAX {
                growth_dir[nearest] += (attractor - nodes[nearest]).normalize_or_zero();
                growth_count[nearest] += 1;
            }
        }

        let mut grew = false;
        let existing_count = node_count;
        for n in 0..existing_count {
            if node_count >= MAX_TREE_NODES || growth_count[n] == 0 {
                continue;
            }
            let dir = (growth_dir[n] / growth_count[n] as f32).normalize_or_zero();
            if dir == glam::Vec3::ZERO {
                continue;
            }
            nodes[node_count] = nodes[n] + dir * segment_length;
            parents[node_count] = n as i8;
            node_count += 1;
            grew = true;
        }

        for (a, &attractor) in attractors.iter().enumerate() {
            if !attractor_alive[a] {
                continue;
            }
            if (0..node_count).any(|n| nodes[n].distance(attractor) < kill_radius) {
                attractor_alive[a] = false;
            }
        }

        if !grew {
            break;
        }
    }

    (nodes, parents, node_count as u8)
}

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
    /// Punkt - exakt dieselbe Trunk-Saeule/Skelett-Geometrie wie `place_flora`, MUSS mit ihr
    /// uebereinstimmen, sonst genau die Bulk/Fallback-Divergenz, die `is_carved` schon einmal
    /// hatte (s. dortiger Kommentar). Aeste werden als Kapsel (Punkt-zu-Strecke-Abstand) geprueft,
    /// Blaetter als kleine Kugel um jeden Skelett-Knoten.
    #[cold]
    pub(super) fn tree_occupies_among(trees: &[TreeSpawn], world_x: i32, world_y: i32, world_z: i32) -> bool {
        for tree in trees {
            let trunk_top = tree.ground_y + tree.trunk_height;

            if world_x == tree.world_x && world_z == tree.world_z && world_y > tree.ground_y && world_y <= trunk_top {
                return true;
            }

            let root = glam::Vec3::new(tree.world_x as f32, trunk_top as f32, tree.world_z as f32);
            let point = glam::Vec3::new(world_x as f32, world_y as f32, world_z as f32) - root;
            let leaf_radius = leaf_cluster_radius(tree.species, tree.crown_radius as f32);
            let branch_radius = BRANCH_RADIUS;

            let node_count = tree.node_count as usize;
            for i in 0..node_count {
                if point.distance(tree.nodes[i]) <= leaf_radius {
                    return true;
                }
                if i == 0 {
                    continue;
                }
                let parent = tree.nodes[tree.parents[i] as usize];
                if point_to_segment_distance(point, parent, tree.nodes[i]) <= branch_radius {
                    return true;
                }
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
    /// generiert wird ("Pure Function"-Prinzip, keine Chunk-Nachbar-Reads noetig). Berechnet dabei
    /// auch gleich das komplette Space-Colonization-Skelett - Baum-Spawns sind selten genug
    /// (Spawn-Chance * Biom-Trefferquote), dass die paar tausend Rechenoperationen pro TATSAECHLICH
    /// gefundenem Baum nicht ins Gewicht fallen, dafuer braucht KEIN Aufrufer (Bulk-Platzierung UND
    /// Einzelpunkt-Fallback) eine eigene Nachberechnungs-/Cache-Logik - eine einzige Quelle der
    /// Wahrheit fuer die Baumform.
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
        let species = if surface.temperature < PINE_TEMPERATURE_THRESHOLD { TreeSpecies::Pine } else { TreeSpecies::Round };

        let trunk_span = (self.tree_trunk_height_max - self.tree_trunk_height_min + 1).max(1);
        let trunk_height = self.tree_trunk_height_min
            + (hash_unit(tree_hash(self.tree_seed, cell_x, cell_z, 3)) * trunk_span as f32) as i32;
        let crown_span = (self.tree_crown_radius_max - self.tree_crown_radius_min + 1).max(1);
        let crown_radius = self.tree_crown_radius_min
            + (hash_unit(tree_hash(self.tree_seed, cell_x, cell_z, 4)) * crown_span as f32) as i32;

        let (nodes, parents, node_count) = grow_skeleton(species, self.tree_seed, cell_x, cell_z, crown_radius as f32);

        Some(TreeSpawn { world_x, world_z, ground_y, trunk_height, crown_radius, species, nodes, parents, node_count })
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
                // Vertikale AABB deckt Stamm UND Skelett-Knoten ab (Knoten liegen relativ zur
                // Stammspitze, nach oben/aussen begrenzt durch die Wachstumshuelle, s.
                // `sample_envelope_point` - `crown_radius * 2.6` ist die hoechste vorkommende
                // Huellenhoehe, bei `TreeSpecies::Pine`).
                let footprint_min_y = tree.ground_y + 1;
                let footprint_max_y = trunk_top + (tree.crown_radius as f32 * 2.6).ceil() as i32;
                let footprint_radius = tree.crown_radius + 1;
                // Billige AABB-Ablehnung, bevor die Voxel des Baumes einzeln durchlaufen werden -
                // die meisten Kandidaten im vergroesserten Suchradius treffen diesen Chunk gar
                // nicht (weder horizontal noch vertikal).
                if footprint_max_y < chunk_origin_y
                    || footprint_min_y > chunk_origin_y + CHUNK_SIZE - 1
                    || tree.world_x + footprint_radius < chunk_origin_x
                    || tree.world_x - footprint_radius > chunk_origin_x + CHUNK_SIZE - 1
                    || tree.world_z + footprint_radius < chunk_origin_z
                    || tree.world_z - footprint_radius > chunk_origin_z + CHUNK_SIZE - 1
                {
                    continue;
                }

                // Stamm: gerader Holzstamm vom Boden bis zur Kronenbasis (Skelett-Wurzel).
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

                let root = glam::Vec3::new(tree.world_x as f32, trunk_top as f32, tree.world_z as f32);
                let node_count = tree.node_count as usize;

                // Aeste: jedes Skelett-Segment (Knoten -> Elternknoten) als Kapsel gerastert -
                // ALLE ganzzahligen Voxel in der Bounding-Box werden gegen EXAKT dieselbe Punkt-zu-
                // Strecke-Distanzformel geprueft wie `tree_occupies_among` (kein "Strecke ablaufen
                // und runden" mehr, das an der Kapsel-Grenze leicht von der exakten Pruefung
                // abweichen konnte - dieselbe Bulk/Fallback-Divergenzklasse wie bei `is_carved`).
                for i in 1..node_count {
                    let from = root + tree.nodes[tree.parents[i] as usize];
                    let to = root + tree.nodes[i];
                    let min = from.min(to) - glam::Vec3::splat(BRANCH_RADIUS);
                    let max = from.max(to) + glam::Vec3::splat(BRANCH_RADIUS);
                    for wy in min.y.floor() as i32..=max.y.ceil() as i32 {
                        for wz in min.z.floor() as i32..=max.z.ceil() as i32 {
                            for wx in min.x.floor() as i32..=max.x.ceil() as i32 {
                                let p = glam::Vec3::new(wx as f32, wy as f32, wz as f32);
                                if point_to_segment_distance(p, from, to) <= BRANCH_RADIUS {
                                    Self::place_tree_voxel(
                                        chunk,
                                        chunk_origin_x,
                                        chunk_origin_y,
                                        chunk_origin_z,
                                        wx,
                                        wy,
                                        wz,
                                        blocks::LOG,
                                    );
                                }
                            }
                        }
                    }
                }

                // Krone: kleine Blattkugel um JEDEN Skelett-Knoten - ergibt bei mehreren Knoten
                // eine zusammenhaengende, aber unregelmaessige (nicht perfekt runde) Silhouette,
                // artspezifisch unterschiedlich dicht (Tanne enger/spitzer als Laubbaum). Wie bei
                // den Aesten: ganzzahlige Voxel direkt gegen die exakte Kugel-Distanz geprueft,
                // nicht ueber einen Integer-Offset VOR dem Runden (s. Kommentar oben).
                let leaf_radius = leaf_cluster_radius(tree.species, tree.crown_radius as f32);
                for i in 0..node_count {
                    let center = root + tree.nodes[i];
                    let min = center - glam::Vec3::splat(leaf_radius);
                    let max = center + glam::Vec3::splat(leaf_radius);
                    for wy in min.y.floor() as i32..=max.y.ceil() as i32 {
                        for wz in min.z.floor() as i32..=max.z.ceil() as i32 {
                            for wx in min.x.floor() as i32..=max.x.ceil() as i32 {
                                let p = glam::Vec3::new(wx as f32, wy as f32, wz as f32);
                                if p.distance(center) <= leaf_radius {
                                    Self::place_tree_voxel(
                                        chunk,
                                        chunk_origin_x,
                                        chunk_origin_y,
                                        chunk_origin_z,
                                        wx,
                                        wy,
                                        wz,
                                        blocks::LEAVES,
                                    );
                                }
                            }
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

/// Radius der Ast-Kapseln - dünn (unter einem Block), damit Aeste als schmale Linien statt als
/// dicke Rohre wirken; der finale Voxel-Look wird dadurch bestimmt, wie nah eine Voxelmitte an der
/// Strecke liegt.
pub(super) const BRANCH_RADIUS: f32 = 0.6;

/// Radius der Blattkugel PRO Skelett-Knoten, artabhaengig - Tanne enger/spitzer (Nadelbaum-Look),
/// Laubbaum grosszuegiger/voller. Skaliert mit `crown_radius`, damit groessere Baeume auch
/// groessere Blattbueschel bekommen. `pub(super)` statt modul-privat, damit der Cross-Chunk-
/// Konsistenztest in `generator.rs` exakt dieselbe Formel wiederverwendet statt sie zu duplizieren.
pub(super) fn leaf_cluster_radius(species: TreeSpecies, crown_radius: f32) -> f32 {
    match species {
        TreeSpecies::Round => (crown_radius * 0.55).max(1.3),
        TreeSpecies::Pine => (crown_radius * 0.4).max(1.0),
    }
}

/// Kuerzester Abstand von `point` zur Strecke `a..b` (Kapsel-Test ohne Radius) - Klassiker:
/// Projektion auf die Strecke, auf `[0, 1]` geklemmt. `pub(super)` aus demselben Grund wie
/// `leaf_cluster_radius`.
pub(super) fn point_to_segment_distance(point: glam::Vec3, a: glam::Vec3, b: glam::Vec3) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-6 {
        return point.distance(a);
    }
    let t = ((point - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    point.distance(a + ab * t)
}

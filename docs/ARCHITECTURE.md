# Voxel Engine Architektur-Spezifikation

Diese Dokumentation beschreibt die fundamentale Datenstruktur und die hochoptimierte Rendering-Pipeline der Voxel-Engine. Das System ist auf minimalen Speicherverbrauch und maximale GPU-Durchsatzraten ausgelegt.

---

## 1. Chunk- und Welt-Geometrie
- **Dimensionen:** Die Spielwelt ist in Chunks der festen Größe $32 \times 32 \times 32$ Voxel unterteilt.
- **Speicher-Layout:** Ein Chunk wird als flaches, eindimensionales Array (`[u16; 32 * 32 * 32]`) repräsentiert, um maximale CPU-Cache-Lokalität bei Iterationen zu garantieren.
- **Welt-Positionierung:** Chunks besitzen keine individuellen, transformierten Vertex-Daten für ihre Weltposition. Die Platzierung im globalen Raum erfolgt ausschließlich über ein globales Koordinaten-Mapping in einem Storage Buffer.

---

## 2. Die Daten- und Meshing-Pipeline

Das Rendering verzichtet komplett auf das Senden einzelner Würfel. Die Pipeline reduziert die Geometrie in vier aufeinanderfolgenden Schritten:

[Rohdaten: 32x32x32 Grid]
│
▼
[Schritt 1: Hidden Face Culling] -> Entfernt alle verdeckten Innenflächen zwischen Voxels
│
▼
[Schritt 2: Directional Splitting] -> Teilt Geometrie in 6 orthogonale Meshes (-X, +X, -Y, +Y, -Z, +Z)
│
▼
[Schritt 3: Greedy Meshing] -> Kombiniert koplanare Flächen gleicher Textur zu größeren Rechtecken
│
▼
[Schritt 4: Instanced Stretches] -> Übergibt komprimierte Vertex-Daten an die GPU

---

### 2.1 Speicher-Architektur & Restriktionen
- **Keine Sparse Voxel Octrees (SVOs):** Die Engine verzichtet explizit auf Octrees. SVOs sparen zwar RAM, zerstören aber die Cache-Lokalität der CPU.
- **Flache Arrays:** Jeder Chunk nutzt zwingend ein lineares Array `[u16; 32768]` (32x32x32). Dies garantiert $O(1)$ Zugriffszeiten und maximale Speicherbandbreite beim Evaluieren des Greedy-Meshing-Algorithmus.

## 3. Die 32-Bit Vertex-Kompression (Zero-Waste)

Die gesamte Information einer Rechteck-Ecke (Face-Instanz) wird lückenlos in einem einzigen vorzeichenlosen 32-Bit-Integer (`u32`) komprimiert. Da die Chunks in 6 Richtungs-Meshes aufgeteilt sind, entfällt die Speicherung der Normalen komplett.

### Bit-Belegungsplan (`u32`)

| Bit-Bereich | Modulo / Maske | Funktion | Beschreibung |
| :--- | :--- | :--- | :--- |
| **00 - 04** | `0x1F` | `x_pos` | Lokale X-Startkoordinate (0 - 31) |
| **05 - 09** | `0x1F` | `y_pos` | Lokale Y-Startkoordinate (0 - 31) |
| **10 - 14** | `0x1F` | `z_pos` | Lokale Z-Startkoordinate (0 - 31) |
| **15 - 21** | `0x7F` | `texture_id` | Textur-Index aus der Palette (0 - 127) |
| **22 - 26** | `0x1F` | `width_stretch`| Horizontale Streckung des Faces (1 - 32 Blöcke) |
| **27 - 31** | `0x1F` | `height_stretch`| Vertikale Streckung des Faces (1 - 32 Blöcke) |

---

## 4. GPU-Rendering & Modern Indirect Pipeline

Die Engine nutzt eine hochmoderne, indirekte Rendering-Pipeline, um den Overhead durch CPU-Draw-Calls auf ein Minimum zu reduzieren.

### Speicher-Management (Unified Buffer)
- Alle generierten Chunk-Meshes werden in einem einzigen, großen, globalen Vertex-Buffer verwaltet.
- Jedem Chunk-Mesh wird ein kontinuierlicher Slice (Offset und Länge) innerhalb dieses Buffers zugewiesen.

### Frustum Culling & Sichtbarkeitsprüfung
- Die Kamera-Frustum-Prüfung (FOV) filtert nicht sichtbare Chunks heraus.
- Die verbleibenden, sichtbaren Zeichenbefehle werden kompakt in einem **Indirect Buffer** gesammelt.

### Die Zeichen-Ausführung (`wgpu`)
- Das finale Rendering wird über einen einzigen Befehl angestoßen: **`draw_indirect`** (das funktionale Äquivalent zu `glMultiDrawArraysIndirect`).
- **Backface Culling:** Flächen, deren Normalen vom Spieler weggedreht sind, werden durch die Separation in 6 Richtungs-Meshes bereits vorab vom Zeichenbefehl ausgeschlossen.

### Storage Buffer Datensatz (SSBO-Äquivalent)
Für jeden aktiven Draw-Call liest die GPU Daten aus einem Bindungs-Kontext (Storage Buffer), welcher folgende Informationen pro Mesh hält:
1. Die globale Weltposition des Chunks (`vec3`).
2. Die Ausrichtung/Normale des spezifischen Richtungs-Meshes zur korrekten Rotations-Matrix-Berechnung des Basis-Quads.

Im WGSL-Shader wird über die eingebaute Variable `@builtin(draw_index)` (entspricht `gl_DrawID`) der exakte Index des aktuellen Meshes ermittelt, um die korrekte Weltposition und Orientierung atomar aus dem Storage Buffer auszulesen.

### Texture Batching (Texture2DArray)
Um den State-Change-Overhead zu eliminieren und das `draw_indirect` nicht zu brechen, werden alle Block-Texturen in einem einzigen **Texture2DArray** auf die GPU geladen. Der `texture_id` Wert aus dem komprimierten 32-Bit-Vertex dient im Fragment-Shader direkt als Layer-Index für den Array-Zugriff.

### 4.1 Rendering-Roadmap & Restriktionen
- **Verbotene Techniken:** Geometry Shaders und Hardware-Tessellation werden aufgrund schlechter Performance auf modernen GPUs strikt untersagt. Wir nutzen ausschließlich Vertex Pulling via SSBOs.
- **Zukünftige Optimierungen:** Implementierung von Frustum & Occlusion Culling. Weit entfernte Chunks erhalten ein algorithmisch gedownsampletes Mesh (Level of Detail).
- **Beleuchtung (Roadmap):** Die Lichtausbreitung erfolgt später über einen Floodfill-Algorithmus (Breadth-First Search) über die Chunk-Grenzen hinweg.

## 5. Skalierung & Tiefenpufferung

### Reverse Depth Buffer (Infinite Render Distance)
Um das klassische Z-Fighting (Flimmern) bei weit entfernten Chunks zu eliminieren, nutzt die Engine einen **Reverse Z-Buffer** mit 32-Bit Floating-Point-Präzision (`Depth32Float`). 
- Die `Near Plane` wird auf `1.0` und die `Far Plane` auf `0.0` gemappt.
- In der WGPU-Pipeline wird die Depth-Compare-Function auf `GreaterEqual` gesetzt. Dies garantiert extrem hohe Präzision auf große Distanzen.

### Level of Detail (LOD) & Multithreading
- **LOD-System:** Weit entfernte Chunks werden in ihrer Auflösung reduziert gerendert. Das Meshing-System fasst Voxel auf Distanz zu größeren Makro-Blöcken zusammen oder ignoriert sub-Block-Details, um den Indirect-Draw-Buffer klein zu halten.
- **Thread-Architektur:** Sämtliche CPU-intensiven Aufgaben (Noise-Auswertung, Chunk-Initialisierung, Greedy Meshing und Frustum Culling) werden via Thread-Pool (`rayon`) asynchron verarbeitet, um 100%ige Framerate-Stabilität im Render-Thread (Main) zu garantieren.

## 6. Advanced Render Pipeline, Post-Processing & Anti-Aliasing

Um hochwertige visuelle Effekte performant umzusetzen, nutzt die WGPU-Pipeline ein Multi-Pass-System:

1. **Shadow Pass:** Generierung von Cascaded Shadow Maps (CSM) für realistische, dynamische Schattenwürfe bei gleichzeitig hoher Sichtweite.
2. **Main Opaque Pass (inkl. Hardware MSAA):** Rendern der komprimierten Voxel-Geometrie in eine Multisampled-Texture (z.B. 4x MSAA), um harte Geometrie-Kanten auf Hardware-Ebene zu glätten, bevor sie in die finale Auflösung "resolved" werden.
3. **Transparent Pass:** Spezifischer Render-Pass für realistische Wasser-Shader (inklusive Tiefen-Auswertung für Refraktion/Absorption) und Glas.
4. **Post-Processing Pass:** Ausführung von Screen-Space-Effekten auf das fertige Bild.
   - *Ambient Occlusion:* Screen-Space (SSAO) zur Hervorhebung der Voxel-Geometrie.
   - *Volumetric Lighting:* Kosteneffiziente "Godrays" via Screen-Space Radial Blur anstelle echter volumetrischer Raymarching-Verfahren.
   - *Temporal/Post Anti-Aliasing (Optional):* Implementierung von TAA (Temporal Anti-Aliasing) oder SMAA, um spekulares Flimmern und Sub-Pixel-Artefakte in der Bewegung zu glätten, die von MSAA nicht erfasst werden.

## 7. Game-Loop, Input & Physik

Um die Engine deterministisch und wartbar zu halten, wird strikt zwischen Spielzustand (State) und Rendering unterschieden.

- **Fixed Timestep Physik:** Berechnungen wie Gravitation, Kollisionen und Geschwindigkeits-Vektoren laufen in einem isolierten `update(dt: f32)`-Loop mit fester Zeitschrittweite, um physikalische Instabilitäten bei schwankenden Framerates zu verhindern.
- **Kamera & Matrizen:** Die Kamera-Logik (View- und Projection-Matrizen) wird als reines Daten-Struct in `src/game/math/camera.rs` verwaltet. Berechnungen erfolgen über das `glam` Crate. Es handelt sich um eine First-Person-Kamera: Bewegungen (Vor/Zurück/Links/Rechts) erfolgen strikt auf der XZ-Ebene (affin zum Boden), unabhängig vom vertikalen Blickwinkel (Pitch).
- **Input Handling:** Rohe Winit-Events (Tastatur, Maus) werden frühzeitig abgefangen und in abstrakte Engine-Commands (z.B. `MoveForward`, `Jump`) übersetzt, bevor sie an den Player-State weitergereicht werden. Das Rendering-System greift niemals direkt auf Tastatureingaben zu.

## 8. Speichermanagement (Zero-Allocation Runtime)
Um Heap-Fragmentierung und Frame-Stuttering zu verhindern, ist dynamische Speicherallokation während des Game-Loops strikt verboten.
- **Chunk Object Pool:** Chunk-Datenstrukturen (`[u16; 32768]`) und deren zugehörige GPU-Buffer werden beim Engine-Start voralloziert. Wenn Chunks entladen werden, wandern sie zurück in einen Recyling-Pool (Arena) und werden für neu generierte Chunks wiederverwendet (überschrieben).

## 9. Konfiguration & Globale Parameter
"Magic Numbers" im Code sind strengstens verboten. 
- Alle globalen Parameter (Movement Speed, Mouse Sensitivity, FOV, Render Distance, Clear Color) werden zentral im `EngineConfig` Struct in `src/engine/config.rs` verwaltet.
- Neue Systeme müssen ihre Parameter aus dieser Konfiguration beziehen und dürfen keine eigenen hartgecodeten Konstanten verwenden.
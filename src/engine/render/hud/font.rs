pub const GLYPH_WIDTH: usize = 5;
pub const GLYPH_HEIGHT: usize = 7;
pub const GLYPH_PADDING: usize = 1;
pub const CELL_WIDTH: usize = GLYPH_WIDTH + GLYPH_PADDING;
pub const CELL_HEIGHT: usize = GLYPH_HEIGHT + GLYPH_PADDING;
pub const ATLAS_COLS: usize = 8;

struct Glyph {
    ch: char,
    rows: [&'static str; GLYPH_HEIGHT],
}

/// Bitmap-Font, 5x7 Pixel pro Glyph, `#` = an, `.` = aus. Prozedural generiert, keine
/// externen Font-Assets noetig.
const GLYPHS: &[Glyph] = &[
    Glyph { ch: '0', rows: [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."] },
    Glyph { ch: '1', rows: ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."] },
    Glyph { ch: '2', rows: [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"] },
    Glyph { ch: '3', rows: [".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."] },
    Glyph { ch: '4', rows: ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."] },
    Glyph { ch: '5', rows: ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."] },
    Glyph { ch: '6', rows: ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."] },
    Glyph { ch: '7', rows: ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."] },
    Glyph { ch: '8', rows: [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."] },
    Glyph { ch: '9', rows: [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."] },
    Glyph { ch: 'A', rows: ["..#..", ".#.#.", "#...#", "#...#", "#####", "#...#", "#...#"] },
    Glyph { ch: 'B', rows: ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."] },
    Glyph { ch: 'C', rows: [".####", "#....", "#....", "#....", "#....", "#....", ".####"] },
    Glyph { ch: 'D', rows: ["###..", "#..#.", "#...#", "#...#", "#...#", "#..#.", "###.."] },
    Glyph { ch: 'E', rows: ["#####", "#....", "#....", "####.", "#....", "#....", "#####"] },
    Glyph { ch: 'F', rows: ["#####", "#....", "#....", "####.", "#....", "#....", "#...."] },
    Glyph { ch: 'G', rows: [".####", "#....", "#....", "#.###", "#...#", "#...#", ".####"] },
    Glyph { ch: 'H', rows: ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"] },
    Glyph { ch: 'I', rows: [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."] },
    Glyph { ch: 'J', rows: ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."] },
    Glyph { ch: 'K', rows: ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"] },
    Glyph { ch: 'L', rows: ["#....", "#....", "#....", "#....", "#....", "#....", "#####"] },
    Glyph { ch: 'M', rows: ["#...#", "##.##", "#.#.#", "#...#", "#...#", "#...#", "#...#"] },
    Glyph { ch: 'N', rows: ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"] },
    Glyph { ch: 'O', rows: [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."] },
    Glyph { ch: 'P', rows: ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."] },
    Glyph { ch: 'Q', rows: [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"] },
    Glyph { ch: 'R', rows: ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"] },
    Glyph { ch: 'S', rows: [".####", "#....", "#....", ".###.", "....#", "....#", "####."] },
    Glyph { ch: 'T', rows: ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."] },
    Glyph { ch: 'U', rows: ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."] },
    Glyph { ch: 'V', rows: ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."] },
    Glyph { ch: 'W', rows: ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"] },
    Glyph { ch: 'X', rows: ["#...#", ".#.#.", "..#..", "..#..", "..#..", ".#.#.", "#...#"] },
    Glyph { ch: 'Y', rows: ["#...#", ".#.#.", "..#..", "..#..", "..#..", "..#..", "..#.."] },
    Glyph { ch: 'Z', rows: ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"] },
    Glyph { ch: ' ', rows: [".....", ".....", ".....", ".....", ".....", ".....", "....."] },
    Glyph { ch: ':', rows: [".....", "..#..", ".....", ".....", ".....", "..#..", "....."] },
    Glyph { ch: '.', rows: [".....", ".....", ".....", ".....", ".....", ".....", "..#.."] },
    Glyph { ch: '/', rows: ["....#", "...#.", "...#.", "..#..", ".#...", ".#...", "#...."] },
    Glyph { ch: '%', rows: ["#...#", "....#", "...#.", "..#..", ".#...", "#....", "#...#"] },
    Glyph { ch: '-', rows: [".....", ".....", ".....", "#####", ".....", ".....", "....."] },
];

/// Position (Spalte, Zeile) des Glyphen in der Atlas-Textur, in Zellen (nicht Pixeln).
pub fn glyph_cell(ch: char) -> (u32, u32) {
    let index = GLYPHS.iter().position(|glyph| glyph.ch == ch).unwrap_or(36);
    ((index % ATLAS_COLS) as u32, (index / ATLAS_COLS) as u32)
}

pub fn atlas_size() -> (u32, u32) {
    let rows = GLYPHS.len().div_ceil(ATLAS_COLS);
    ((ATLAS_COLS * CELL_WIDTH) as u32, (rows * CELL_HEIGHT) as u32)
}

/// Erzeugt die RGBA8-Pixel der Font-Atlas-Textur: weiss auf transparent, damit der Fragment-Shader
/// die Farbe frei bestimmen kann (Alpha-Blending gegen die Szene).
pub fn generate_font_atlas() -> Vec<u8> {
    let (width, height) = atlas_size();
    let mut data = vec![0u8; (width * height * 4) as usize];

    for (index, glyph) in GLYPHS.iter().enumerate() {
        let cell_x = (index % ATLAS_COLS) * CELL_WIDTH;
        let cell_y = (index / ATLAS_COLS) * CELL_HEIGHT;

        for (row, line) in glyph.rows.iter().enumerate() {
            for (col, pixel) in line.bytes().enumerate() {
                if pixel != b'#' {
                    continue;
                }
                let px = cell_x + col;
                let py = cell_y + row;
                let offset = ((py as u32 * width + px as u32) * 4) as usize;
                data[offset..offset + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
    }

    data
}

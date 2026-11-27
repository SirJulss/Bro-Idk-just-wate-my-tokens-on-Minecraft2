//! The engine's own bitmap font for debug text: printable ASCII in 5 x 8 pixel glyphs.
//!
//! The glyphs were drawn for this engine and are defined in code below (release-safe, no
//! third-party font). A glyph box is 5 columns by 8 rows: rows 0 to 6 hold capitals and
//! digits (cap height 7, baseline under row 6), lower case has its x-height in rows 2 to 6,
//! and row 7 takes the descenders of `g j p q y`, the comma and the underscore. Text advances
//! [`GLYPH_ADVANCE`] pixels per character and [`LINE_HEIGHT`] pixels per line.
//!
//! Characters outside `' '..='~'` are drawn as the hollow box of [`UNKNOWN_GLYPH`].

/// Width of a glyph in pixels.
pub const GLYPH_WIDTH: u32 = 5;
/// Height of a glyph box in pixels, descender row included.
pub const GLYPH_HEIGHT: u32 = 8;
/// Horizontal distance from one character to the next, in pixels: one pixel of spacing.
pub const GLYPH_ADVANCE: u32 = 6;
/// Vertical distance from one text line to the next, in pixels: two pixels of spacing.
pub const LINE_HEIGHT: u32 = 10;

/// First character of the font: the space.
pub const FIRST_CHAR: char = ' ';
/// Number of glyphs: the 95 printable ASCII characters and the box for everything else.
pub const GLYPH_COUNT: usize = 96;
/// Index of the hollow box drawn for characters the font does not have.
pub const UNKNOWN_GLYPH: u32 = 95;

/// Glyph cells per row of the atlas texture.
pub const ATLAS_COLUMNS: u32 = 16;
/// Edge of a square atlas cell in pixels; a glyph sits in its top left corner.
pub const ATLAS_CELL: u32 = 8;
/// Width of the atlas texture in pixels.
pub const ATLAS_WIDTH: u32 = ATLAS_COLUMNS * ATLAS_CELL;
/// Height of the atlas texture in pixels.
pub const ATLAS_HEIGHT: u32 = (GLYPH_COUNT as u32).div_ceil(ATLAS_COLUMNS) * ATLAS_CELL;

/// Glyphs per band of the sheet below.
const BAND_GLYPHS: usize = 8;
/// Characters of a sheet line: eight glyphs of five, separated by blanks.
const SHEET_LINE: usize = BAND_GLYPHS * 6 - 1;
/// Lines of the sheet: one per glyph row of every band.
const SHEET_LINES: usize = GLYPH_COUNT / BAND_GLYPHS * GLYPH_HEIGHT as usize;

/// The font as drawn: twelve bands of eight glyphs, eight rows per band, `#` is ink. The
/// glyphs of a band are in ASCII order; the comment above a band names them.
const SHEET: [&str; SHEET_LINES] = [
    // space  !      "      #      $      %      &      '
    "..... ..#.. .#.#. .#.#. ..#.. ##... .##.. ..#..",
    "..... ..#.. .#.#. .#.#. .#### ##..# #..#. ..#..",
    "..... ..#.. ..... ##### #.#.. ...#. #.#.. .....",
    "..... ..#.. ..... .#.#. .###. ..#.. .#... .....",
    "..... ..#.. ..... ##### ..#.# .#... #.#.# .....",
    "..... ..... ..... .#.#. ####. #..## #..#. .....",
    "..... ..#.. ..... .#.#. ..#.. ...## .##.# .....",
    "..... ..... ..... ..... ..... ..... ..... .....",
    // (      )      *      +      ,      -      .      /
    "...#. .#... ..... ..... ..... ..... ..... .....",
    "..#.. ..#.. ..#.. ..#.. ..... ..... ..... ....#",
    ".#... ...#. #.#.# ..#.. ..... ..... ..... ...#.",
    ".#... ...#. .###. ##### ..... ##### ..... ..#..",
    ".#... ...#. #.#.# ..#.. ..... ..... ..... .#...",
    "..#.. ..#.. ..#.. ..#.. ..#.. ..... ..... #....",
    "...#. .#... ..... ..... ..#.. ..... ..#.. .....",
    "..... ..... ..... ..... .#... ..... ..... .....",
    // 0      1      2      3      4      5      6      7
    ".###. ..#.. .###. ##### ...#. ##### ..##. #####",
    "#...# .##.. #...# ...#. ..##. #.... .#... ....#",
    "#..## ..#.. ....# ..#.. .#.#. ####. #.... ...#.",
    "#.#.# ..#.. ...#. ...#. #..#. ....# ####. ..#..",
    "##..# ..#.. ..#.. ....# ##### ....# #...# .#...",
    "#...# ..#.. .#... #...# ...#. #...# #...# .#...",
    ".###. .###. ##### .###. ...#. .###. .###. .#...",
    "..... ..... ..... ..... ..... ..... ..... .....",
    // 8      9      :      ;      <      =      >      ?
    ".###. .###. ..... ..... ..... ..... ..... .###.",
    "#...# #...# ..... ..... ...#. ..... .#... #...#",
    "#...# #...# ..#.. ..#.. ..#.. ##### ..#.. ....#",
    ".###. .#### ..... ..... .#... ..... ...#. ...#.",
    "#...# ....# ..... ..... ..#.. ##### ..#.. ..#..",
    "#...# ...#. ..#.. ..#.. ...#. ..... .#... .....",
    ".###. .##.. ..... ..#.. ..... ..... ..... ..#..",
    "..... ..... ..... .#... ..... ..... ..... .....",
    // @      A      B      C      D      E      F      G
    ".###. .###. ####. .###. ###.. ##### ##### .###.",
    "#...# #...# #...# #...# #..#. #.... #.... #...#",
    "#.### #...# #...# #.... #...# #.... #.... #....",
    "#.#.# ##### ####. #.... #...# ####. ####. #.###",
    "#.### #...# #...# #.... #...# #.... #.... #...#",
    "#.... #...# #...# #...# #..#. #.... #.... #...#",
    ".###. #...# ####. .###. ###.. ##### #.... .####",
    "..... ..... ..... ..... ..... ..... ..... .....",
    // H      I      J      K      L      M      N      O
    "#...# .###. ..### #...# #.... #...# #...# .###.",
    "#...# ..#.. ...#. #..#. #.... ##.## #...# #...#",
    "#...# ..#.. ...#. #.#.. #.... #.#.# ##..# #...#",
    "##### ..#.. ...#. ##... #.... #.#.# #.#.# #...#",
    "#...# ..#.. ...#. #.#.. #.... #...# #..## #...#",
    "#...# ..#.. #..#. #..#. #.... #...# #...# #...#",
    "#...# .###. .##.. #...# ##### #...# #...# .###.",
    "..... ..... ..... ..... ..... ..... ..... .....",
    // P      Q      R      S      T      U      V      W
    "####. .###. ####. .#### ##### #...# #...# #...#",
    "#...# #...# #...# #.... ..#.. #...# #...# #...#",
    "#...# #...# #...# #.... ..#.. #...# #...# #...#",
    "####. #...# ####. .###. ..#.. #...# #...# #.#.#",
    "#.... #.#.# #.#.. ....# ..#.. #...# #...# #.#.#",
    "#.... #..#. #..#. ....# ..#.. #...# .#.#. #.#.#",
    "#.... .##.# #...# ####. ..#.. .###. ..#.. .#.#.",
    "..... ..... ..... ..... ..... ..... ..... .....",
    // X      Y      Z      [      \      ]      ^      _
    "#...# #...# ##### .###. ..... .###. ..#.. .....",
    "#...# #...# ....# .#... #.... ...#. .#.#. .....",
    ".#.#. .#.#. ...#. .#... .#... ...#. #...# .....",
    "..#.. ..#.. ..#.. .#... ..#.. ...#. ..... .....",
    ".#.#. ..#.. .#... .#... ...#. ...#. ..... .....",
    "#...# ..#.. #.... .#... ....# ...#. ..... .....",
    "#...# ..#.. ##### .###. ..... .###. ..... .....",
    "..... ..... ..... ..... ..... ..... ..... #####",
    // `      a      b      c      d      e      f      g
    ".#... ..... #.... ..... ....# ..... ..##. .....",
    "..#.. ..... #.... ..... ....# ..... .#..# .....",
    "..... .###. #.##. .###. .##.# .###. .#... .####",
    "..... ....# ##..# #.... #..## #...# ###.. #...#",
    "..... .#### #...# #.... #...# ##### .#... #...#",
    "..... #...# #...# #...# #...# #.... .#... .####",
    "..... .#### ####. .###. .#### .###. .#... ....#",
    "..... ..... ..... ..... ..... ..... ..... .###.",
    // h      i      j      k      l      m      n      o
    "#.... ..#.. ...#. #.... .##.. ..... ..... .....",
    "#.... ..... ..... #.... ..#.. ..... ..... .....",
    "#.##. .##.. ..##. #..#. ..#.. ##.#. #.##. .###.",
    "##..# ..#.. ...#. #.#.. ..#.. #.#.# ##..# #...#",
    "#...# ..#.. ...#. ##... ..#.. #.#.# #...# #...#",
    "#...# ..#.. ...#. #.#.. ..#.. #...# #...# #...#",
    "#...# .###. #..#. #..#. .###. #...# #...# .###.",
    "..... ..... .##.. ..... ..... ..... ..... .....",
    // p      q      r      s      t      u      v      w
    "..... ..... ..... ..... .#... ..... ..... .....",
    "..... ..... ..... ..... .#... ..... ..... .....",
    "####. .#### #.##. .#### ###.. #...# #...# #...#",
    "#...# #...# ##..# #.... .#... #...# #...# #...#",
    "#...# #...# #.... .###. .#... #...# #...# #.#.#",
    "####. .#### #.... ....# .#..# #..## .#.#. #.#.#",
    "#.... ....# #.... ####. ..##. .##.# ..#.. .#.#.",
    "#.... ....# ..... ..... ..... ..... ..... .....",
    // x      y      z      {      |      }      ~      (unknown)
    "..... ..... ..... ...## ..#.. ##... ..... #####",
    "..... ..... ..... ..#.. ..#.. ..#.. ..... #...#",
    "#...# #...# ##### ..#.. ..#.. ..#.. .#... #...#",
    ".#.#. #...# ...#. .#... ..#.. ...#. #.#.# #...#",
    "..#.. #...# ..#.. ..#.. ..#.. ..#.. ...#. #...#",
    ".#.#. .#### .#... ..#.. ..#.. ..#.. ..... #...#",
    "#...# ....# ##### ...## ..#.. ##... ..... #####",
    "..... .###. ..... ..... ..#.. ..... ..... .....",
];

/// The glyphs, decoded from the sheet at compile time: `GLYPHS[i][row]` has bit 4 for the
/// leftmost column and bit 0 for the rightmost.
pub const GLYPHS: [[u8; GLYPH_HEIGHT as usize]; GLYPH_COUNT] = decode_sheet(&SHEET);

const fn decode_sheet(sheet: &[&str; SHEET_LINES]) -> [[u8; GLYPH_HEIGHT as usize]; GLYPH_COUNT] {
    let rows = GLYPH_HEIGHT as usize;
    let mut glyphs = [[0u8; GLYPH_HEIGHT as usize]; GLYPH_COUNT];
    let mut line = 0;
    while line < sheet.len() {
        let text = sheet[line].as_bytes();
        assert!(
            text.len() == SHEET_LINE,
            "a font sheet line has the wrong length"
        );
        let band = line / rows;
        let row = line % rows;
        let mut glyph = 0;
        while glyph < BAND_GLYPHS {
            let mut bits = 0u8;
            let mut column = 0;
            while column < GLYPH_WIDTH as usize {
                let ink = match text[glyph * 6 + column] {
                    b'#' => 1,
                    b'.' => 0,
                    _ => panic!("a font sheet glyph has a character other than '#' and '.'"),
                };
                bits = (bits << 1) | ink;
                column += 1;
            }
            if glyph + 1 < BAND_GLYPHS {
                assert!(
                    text[glyph * 6 + 5] == b' ',
                    "font sheet glyphs are separated by blanks"
                );
            }
            glyphs[band * BAND_GLYPHS + glyph][row] = bits;
            glyph += 1;
        }
        line += 1;
    }
    glyphs
}

/// The glyph index of `character`: its offset from [`FIRST_CHAR`], or [`UNKNOWN_GLYPH`].
pub fn glyph_index(character: char) -> u32 {
    match character {
        ' '..='~' => character as u32 - FIRST_CHAR as u32,
        _ => UNKNOWN_GLYPH,
    }
}

/// The rows of the glyph of `character` (see [`GLYPHS`]).
pub fn glyph_rows(character: char) -> [u8; GLYPH_HEIGHT as usize] {
    GLYPHS[glyph_index(character) as usize]
}

/// Whether column `x` (0 is left) of row `y` (0 is top) of glyph `index` has ink.
pub fn glyph_pixel(index: u32, x: u32, y: u32) -> bool {
    if x >= GLYPH_WIDTH || y >= GLYPH_HEIGHT {
        return false;
    }
    let row = GLYPHS[index as usize % GLYPH_COUNT][y as usize];
    row & (1 << (GLYPH_WIDTH - 1 - x)) != 0
}

/// The atlas texture: `ATLAS_WIDTH * ATLAS_HEIGHT` bytes of `R8Unorm`, 255 for ink. Glyph `i`
/// has its top left corner at `(i % ATLAS_COLUMNS, i / ATLAS_COLUMNS) * ATLAS_CELL`.
pub fn atlas_pixels() -> Vec<u8> {
    let mut pixels = vec![0u8; (ATLAS_WIDTH * ATLAS_HEIGHT) as usize];
    for index in 0..GLYPH_COUNT as u32 {
        let origin_x = (index % ATLAS_COLUMNS) * ATLAS_CELL;
        let origin_y = (index / ATLAS_COLUMNS) * ATLAS_CELL;
        for y in 0..GLYPH_HEIGHT {
            for x in 0..GLYPH_WIDTH {
                if glyph_pixel(index, x, y) {
                    pixels[((origin_y + y) * ATLAS_WIDTH + origin_x + x) as usize] = 255;
                }
            }
        }
    }
    pixels
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn every_printable_character_has_its_own_glyph() {
        let mut seen: HashMap<[u8; 8], char> = HashMap::new();
        for character in ' '..='~' {
            let rows = glyph_rows(character);
            let ink: u32 = rows.iter().map(|row| row.count_ones()).sum();
            if character == ' ' {
                assert_eq!(ink, 0);
            } else {
                assert!(ink > 0, "{character:?} has no ink");
            }
            if let Some(other) = seen.insert(rows, character) {
                panic!("{character:?} and {other:?} look the same");
            }
            // Glyphs are five pixels wide.
            assert!(rows.iter().all(|row| *row < 32), "{character:?}");
        }
        assert_eq!(seen.len(), 95);
    }

    #[test]
    fn the_sheet_decodes_to_the_drawn_glyphs() {
        // Golden rows of a few glyphs, written out independently of the sheet parser.
        assert_eq!(
            glyph_rows('A'),
            [
                0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001, 0
            ]
        );
        assert_eq!(
            glyph_rows('g'),
            [0, 0, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110]
        );
        assert_eq!(glyph_rows('_')[7], 0b11111);
        assert_eq!(glyph_rows('|'), [0b00100; 8]);
        assert_eq!(glyph_index(' '), 0);
        assert_eq!(glyph_index('~'), 94);
        // Characters the font lacks get the box.
        for character in ['\n', '\u{7f}', 'é', '°', '\u{1F600}'] {
            assert_eq!(glyph_index(character), UNKNOWN_GLYPH, "{character:?}");
        }
        assert_eq!(
            GLYPHS[UNKNOWN_GLYPH as usize],
            [
                0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111, 0
            ]
        );
    }

    #[test]
    fn descenders_only_where_they_belong() {
        let with_descender: Vec<char> = (' '..='~')
            .filter(|&character| glyph_rows(character)[7] != 0)
            .collect();
        assert_eq!(with_descender.iter().collect::<String>(), ",;_gjpqy|");
    }

    #[test]
    fn the_atlas_holds_every_glyph_in_its_cell() {
        let atlas = atlas_pixels();
        assert_eq!(atlas.len(), (ATLAS_WIDTH * ATLAS_HEIGHT) as usize);
        assert_eq!((ATLAS_WIDTH, ATLAS_HEIGHT), (128, 48));
        for index in 0..GLYPH_COUNT as u32 {
            let origin_x = (index % ATLAS_COLUMNS) * ATLAS_CELL;
            let origin_y = (index / ATLAS_COLUMNS) * ATLAS_CELL;
            for y in 0..ATLAS_CELL {
                for x in 0..ATLAS_CELL {
                    let texel = atlas[((origin_y + y) * ATLAS_WIDTH + origin_x + x) as usize];
                    let expected = if glyph_pixel(index, x, y) { 255 } else { 0 };
                    assert_eq!(texel, expected, "glyph {index} at ({x}, {y})");
                }
            }
        }
        // Outside the glyph box nothing has ink.
        assert!(!glyph_pixel(33, GLYPH_WIDTH, 0));
        assert!(!glyph_pixel(33, 0, GLYPH_HEIGHT));
    }
}

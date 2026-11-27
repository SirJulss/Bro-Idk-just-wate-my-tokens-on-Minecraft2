//! `cpe proctex-sheet`: the contact sheet of `cp-proctex` as a PNG file (contract 4.3), or with
//! `--pack` the textures of content packs as the game makes and shows them
//! ([`crate::pack_sheet`]).

use std::fmt;
use std::path::PathBuf;

use crate::cli::SheetArgs;
use crate::{Failure, image, pack_sheet, write_file};

/// What was written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SheetInfo {
    /// The PNG file.
    pub path: PathBuf,
    /// Width of the image in pixels.
    pub width: u32,
    /// Height of the image in pixels.
    pub height: u32,
    /// Texture size the sheet shows, in texels.
    pub size: u32,
    /// Magnification of every texture.
    pub scale: u32,
}

impl fmt::Display for SheetInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "wrote {}: {} x {} px, textures of {} px shown {}x",
            self.path.display(),
            self.width,
            self.height,
            self.size,
            self.scale
        )
    }
}

/// Renders the contact sheet for `args.size` and `args.scale` (or, with packs, the sheet of the
/// packs' textures) and writes it as an RGBA PNG.
///
/// # Errors
/// A [`Failure`] when a pack does not load, a texture cannot be made, or the PNG cannot be
/// encoded or written.
pub fn run(args: &SheetArgs) -> Result<SheetInfo, Failure> {
    let (width, height, pixels) = if args.packs.is_empty() {
        cp_proctex::contact_sheet(args.size, args.scale)
    } else {
        pack_sheet::render(&args.packs, args.size, args.scale)?
    };
    let png = image::encode_rgba(width, height, &pixels)
        .map_err(|reason| Failure::new(format!("cannot encode the contact sheet: {reason}")))?;
    write_file(&args.out, &png)?;
    Ok(SheetInfo {
        path: args.out.clone(),
        width,
        height,
        size: args.size,
        scale: args.scale,
    })
}

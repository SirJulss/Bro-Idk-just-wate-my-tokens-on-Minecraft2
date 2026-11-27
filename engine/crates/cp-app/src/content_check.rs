//! `cpe content-check`: loads content packs and checks them against `cp-proctex`
//! (contract 4.4, acceptance of milestone E0).
//!
//! `cp-content` does not depend on `cp-proctex`, so whether a texture can really be made is
//! decided here, where both meet: every procedural texture is generated with its recipe,
//! parameters and seed, every image texture is decoded. The textures then give the material
//! colours ([`Registry::set_material_colors`] from [`cp_proctex::average_color`]), the step
//! the application has to take before it hands the tables to the renderer.

use std::fmt;
use std::path::{Path, PathBuf};

use cp_content::{MaterialFlags, PackInfo, Registry, TextureDef, TextureSource};
use cp_proctex::{ALPHA_CUTOFF, MAX_SIZE, MIN_SIZE, PbrTexture};

use crate::cli::ContentCheckArgs;
use crate::{Failure, image};

/// Edge length the procedural textures are generated with, in texels: the native size of
/// the designs, and the layer size of the block texture array.
pub const TEXTURE_SIZE: u32 = MIN_SIZE;

/// The result of a successful check.
#[derive(Clone, Debug, PartialEq)]
pub struct ContentReport {
    /// The loaded packs in load order: directory and what `pack.toml` says.
    pub packs: Vec<(PathBuf, PackInfo)>,
    /// Number of block definitions.
    pub blocks: usize,
    /// Number of block states, air included.
    pub states: usize,
    /// Number of textures (layers of the block texture array).
    pub textures: usize,
    /// How many of them are procedural.
    pub procedural_textures: usize,
    /// Names of the cull groups, in id order (group `i + 1` is `cull_groups[i]`).
    pub cull_groups: Vec<String>,
    /// [`Registry::content_hash`].
    pub content_hash: u64,
    /// Every material with its colour, in [`cp_core::ids::MaterialId`] order.
    pub materials: Vec<MaterialColor>,
    /// Things that load and render, but probably not as meant.
    pub warnings: Vec<String>,
}

/// The colour of one material.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialColor {
    /// Key of the material.
    pub key: String,
    /// Key of its texture.
    pub texture: String,
    /// Linear RGB average albedo of the texture (alpha-weighted, before any tint); what
    /// [`Registry::set_material_colors`] received.
    pub color: [f32; 3],
}

/// Loads the packs of `args` (or the workspace's base pack), makes every texture and fills
/// the material colours.
///
/// # Errors
/// A [`Failure`] when no default pack can be found, when the packs do not load, or when a
/// texture cannot be made; in the last case the message lists every such texture.
pub fn run(args: &ContentCheckArgs) -> Result<ContentReport, Failure> {
    let dirs = if args.packs.is_empty() {
        vec![default_pack_dir().ok_or_else(|| {
            Failure::new(
                "cannot find the base pack of the workspace (content/base next to the \
                 executable's target directory); name a pack with --pack <dir>",
            )
        })?]
    } else {
        args.packs.clone()
    };
    for dir in &dirs {
        log::info!("loading pack {}", dir.display());
    }
    let dir_refs: Vec<&Path> = dirs.iter().map(PathBuf::as_path).collect();
    let mut registry = Registry::load(&dir_refs)
        .map_err(|error| Failure::new(format!("the content packs do not load: {error}")))?;
    let checked = check(&mut registry)?;
    Ok(ContentReport {
        packs: dirs
            .into_iter()
            .zip(registry.packs().iter().cloned())
            .collect(),
        blocks: registry.blocks().len(),
        states: registry.state_count(),
        textures: registry.textures().len(),
        procedural_textures: registry
            .textures()
            .iter()
            .filter(|texture| matches!(texture.source, TextureSource::Procedural { .. }))
            .count(),
        cull_groups: registry.cull_groups().to_vec(),
        content_hash: registry.content_hash(),
        materials: registry
            .materials()
            .iter()
            .zip(&checked.colors)
            .map(|(material, color)| MaterialColor {
                key: material.key.clone(),
                texture: registry.textures()[material.texture.index()].key.clone(),
                color: *color,
            })
            .collect(),
        warnings: checked.warnings,
    })
}

/// What [`check`] found.
#[derive(Clone, Debug, PartialEq)]
pub struct Checked {
    /// The colour of every material, as passed to [`Registry::set_material_colors`].
    pub colors: Vec<[f32; 3]>,
    /// Materials whose flags do not fit their texture.
    pub warnings: Vec<String>,
}

/// Makes every texture of `registry`, cross-checks the materials against them and fills the
/// material colours of the registry.
///
/// Warnings (also logged): a texture with see-through texels (alpha below
/// [`ALPHA_CUTOFF`]) on a material that is neither cut-out nor translucent, whose holes would
/// show the colour hidden in them; and a cut-out material whose texture has no such texel.
///
/// # Errors
/// A [`Failure`] listing every texture that cannot be made (unknown recipe, parameter that
/// does not fit the recipe, unreadable or unsuitable image file). The registry is left
/// unchanged then.
pub fn check(registry: &mut Registry) -> Result<Checked, Failure> {
    check_with_textures(registry).map(|(checked, _)| checked)
}

/// [`check`], also returning every texture it made, indexed by
/// [`cp_core::ids::TextureLayer`]: what the renderer's texture arrays are built from.
///
/// # Errors
/// Those of [`check`].
pub(crate) fn check_with_textures(
    registry: &mut Registry,
) -> Result<(Checked, Vec<PbrTexture>), Failure> {
    let mut textures = Vec::with_capacity(registry.textures().len());
    let mut failures = Vec::new();
    for def in registry.textures() {
        match make_texture(def) {
            Ok(texture) => textures.push(texture),
            Err(reason) => failures.push(format!("texture `{}`: {reason}", def.key)),
        }
    }
    if !failures.is_empty() {
        return Err(Failure::new(format!(
            "{} of {} textures cannot be made:\n  {}",
            failures.len(),
            registry.textures().len(),
            failures.join("\n  ")
        )));
    }

    let mut colors = Vec::with_capacity(registry.materials().len());
    let mut warnings = Vec::new();
    for material in registry.materials() {
        let texture = &textures[material.texture.index()];
        let texture_key = &registry.textures()[material.texture.index()].key;
        let has_holes = texture.albedo.iter().any(|texel| texel[3] < ALPHA_CUTOFF);
        let see_through = material.flags.contains(MaterialFlags::CUTOUT)
            || material.flags.contains(MaterialFlags::TRANSLUCENT);
        if has_holes && !see_through {
            warnings.push(format!(
                "material `{}`: texture `{texture_key}` has see-through texels, but the \
                 material is neither cutout nor translucent, so they render in the colour \
                 hidden in the holes",
                material.key
            ));
        } else if !has_holes && material.flags.contains(MaterialFlags::CUTOUT) {
            warnings.push(format!(
                "material `{}`: cutout, but texture `{texture_key}` has no see-through texel",
                material.key
            ));
        }
        colors.push(cp_proctex::average_color(texture));
    }
    for warning in &warnings {
        log::warn!("{warning}");
    }
    registry.set_material_colors(&colors);
    Ok((Checked { colors, warnings }, textures))
}

/// The pixels of one texture: generated at [`TEXTURE_SIZE`], or read from its image file.
pub(crate) fn make_texture(def: &TextureDef) -> Result<PbrTexture, String> {
    make_texture_sized(def, TEXTURE_SIZE)
}

/// The pixels of one texture: generated at `size` texels (a supported cp-proctex size), or read
/// from its image file at the file's own size.
pub(crate) fn make_texture_sized(def: &TextureDef, size: u32) -> Result<PbrTexture, String> {
    match &def.source {
        TextureSource::Procedural {
            recipe,
            params,
            seed,
        } => {
            let mut recipe_params = cp_proctex::Params::new();
            for (name, value) in params.iter() {
                let value = match value {
                    cp_content::ParamValue::Color(rgb) => cp_proctex::ParamValue::Color(*rgb),
                    cp_content::ParamValue::Variant(variant) => {
                        cp_proctex::ParamValue::Variant(variant.clone())
                    }
                };
                recipe_params.set(name, value);
            }
            log::debug!("generating `{}`: recipe `{recipe}`, seed {seed}", def.key);
            cp_proctex::generate(recipe, &recipe_params, *seed, size)
                .map_err(|error| error.to_string())
        }
        TextureSource::File { path, resolved } => {
            file_texture(resolved).map_err(|reason| format!("{path}: {reason}"))
        }
    }
}

/// An image texture as a [`PbrTexture`]: the file gives the albedo; the surface is flat,
/// fully rough and does not glow.
fn file_texture(resolved: &Path) -> Result<PbrTexture, String> {
    let bytes = std::fs::read(resolved)
        .map_err(|error| format!("cannot read {}: {error}", resolved.display()))?;
    let (width, height, albedo) = image::decode_rgba(&bytes)?;
    if width != height || !width.is_power_of_two() || !(MIN_SIZE..=MAX_SIZE).contains(&width) {
        return Err(format!(
            "the image is {width} x {height} px; a block texture is square with a \
             power-of-two edge from {MIN_SIZE} to {MAX_SIZE} px"
        ));
    }
    let texels = albedo.len();
    Ok(PbrTexture {
        size: width,
        albedo,
        normal: vec![[128, 128]; texels],
        roughness: vec![255; texels],
        emissive: vec![0; texels],
        height: vec![0; texels],
    })
}

/// The base pack of the workspace this executable belongs to.
///
/// Looks for `content/base` (or `engine/content/base`) holding a `pack.toml` in the
/// directories above the executable, which finds the pack for every build under
/// `engine/target`; then in the workspace the binary was compiled in.
#[must_use]
pub fn default_pack_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    let beside_exe = exe
        .as_deref()
        .and_then(|exe| exe.ancestors().skip(1).find_map(base_pack_in));
    beside_exe.or_else(|| {
        // CARGO_MANIFEST_DIR is engine/crates/cp-app.
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest.ancestors().nth(2).and_then(base_pack_in)
    })
}

/// `dir/content/base` or `dir/engine/content/base`, whichever holds a `pack.toml`.
fn base_pack_in(dir: &Path) -> Option<PathBuf> {
    [
        dir.join("content").join("base"),
        dir.join("engine").join("content").join("base"),
    ]
    .into_iter()
    .find(|pack| pack.join("pack.toml").is_file())
}

impl fmt::Display for ContentReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, (dir, info)) in self.packs.iter().enumerate() {
            let label = if index == 0 { "packs:" } else { "" };
            writeln!(
                f,
                "{label:<14}{} {} ({})",
                info.name,
                info.version,
                dir.display()
            )?;
        }
        writeln!(f, "{:<14}{}", "blocks:", self.blocks)?;
        writeln!(f, "{:<14}{}", "states:", self.states)?;
        writeln!(f, "{:<14}{}", "materials:", self.materials.len())?;
        writeln!(
            f,
            "{:<14}{} ({} procedural, {} image files; generated at {TEXTURE_SIZE} px)",
            "textures:",
            self.textures,
            self.procedural_textures,
            self.textures - self.procedural_textures
        )?;
        let names = if self.cull_groups.is_empty() {
            String::new()
        } else {
            format!(" ({})", self.cull_groups.join(", "))
        };
        writeln!(f, "{:<14}{}{names}", "cull groups:", self.cull_groups.len())?;
        writeln!(f, "{:<14}{:#018x}", "content hash:", self.content_hash)?;
        writeln!(f, "{:<14}{}", "warnings:", self.warnings.len())?;
        for warning in &self.warnings {
            writeln!(f, "  {warning}")?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "material colours (linear RGB average albedo of the texture, before tint; sRGB):"
        )?;
        let key_width = self
            .materials
            .iter()
            .map(|m| m.key.len())
            .max()
            .unwrap_or(0);
        let texture_width = self
            .materials
            .iter()
            .map(|m| m.texture.len())
            .max()
            .unwrap_or(0);
        for material in &self.materials {
            let [r, g, b] = material.color;
            writeln!(
                f,
                "  {:<key_width$}  {:<texture_width$}  {r:.3} {g:.3} {b:.3}  {}",
                material.key,
                material.texture,
                srgb_hex(material.color)
            )?;
        }
        Ok(())
    }
}

/// `#rrggbb` of a linear RGB colour, for reading the report.
fn srgb_hex(linear: [f32; 3]) -> String {
    let [r, g, b] = linear.map(|channel| {
        let c = channel.clamp(0.0, 1.0);
        let encoded = if c <= 0.003_130_8 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (encoded * 255.0 + 0.5) as u8
    });
    format!("#{r:02x}{g:02x}{b:02x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_content::NO_MATERIAL_COLOR;

    fn base_pack() -> PathBuf {
        default_pack_dir().expect("the workspace has its base pack")
    }

    #[test]
    fn the_base_pack_is_found_from_the_test_executable() {
        let pack = base_pack();
        assert!(pack.join("pack.toml").is_file());
        assert!(pack.ends_with(Path::new("content").join("base")));
    }

    #[test]
    fn the_base_pack_passes_and_gets_its_material_colours() {
        let mut registry = Registry::load(&[base_pack().as_path()]).expect("the base pack loads");
        let checked = check(&mut registry).expect("every texture of the base pack can be made");
        assert_eq!(checked.colors.len(), registry.materials().len());
        assert_eq!(checked.warnings, Vec::<String>::new());
        for (material, color) in registry.materials().iter().zip(&checked.colors) {
            assert!(
                color
                    .iter()
                    .all(|c| c.is_finite() && (0.0..=1.0).contains(c)),
                "{}: {color:?}",
                material.key
            );
        }
        // Every state with faces now has a colour of its own; air keeps "no colour".
        let tables = registry.tables();
        assert_eq!(tables.material_color[0], NO_MATERIAL_COLOR);
        let stone = registry.state("cubeplanet:stone").expect("stone exists");
        let stone_material = registry
            .material("cubeplanet:stone")
            .expect("stone material");
        assert_eq!(
            tables.material_color[stone.index()],
            checked.colors[stone_material.index()]
        );
    }

    #[test]
    fn the_report_shows_counts_hash_and_colours() {
        let report = run(&ContentCheckArgs::default()).expect("the base pack passes");
        let registry = Registry::load(&[base_pack().as_path()]).expect("the base pack loads");
        assert_eq!(report.content_hash, registry.content_hash());
        assert_eq!(report.states, registry.state_count());
        assert_eq!(report.materials.len(), registry.materials().len());
        let text = report.to_string();
        assert!(text.contains(&format!("{:#018x}", registry.content_hash())));
        assert!(text.contains("cubeplanet:stone"));
        assert!(text.is_ascii());
    }

    #[test]
    fn srgb_hex_inverts_the_transfer_function() {
        // 0.2158605 is the linear value of the sRGB byte 128.
        assert_eq!(srgb_hex([0.0, 1.0, 0.215_860_5]), "#00ff80");
        assert_eq!(srgb_hex([-1.0, 2.0, f32::NAN]), "#00ff00");
    }
}

//! Content → renderer glue (contract §3.1: "cp-app glues content → textures → renderer").
//!
//! [`Assets::load`] loads the content packs, makes every texture with cp-proctex (or reads its
//! image file) and fills the material colours of the registry. From there it builds what the
//! other crates take as plain data: the material table and the block texture array of
//! cp-render, the terrain palette of cp-worldgen, the flag and LOD tables of cp-voxel and the
//! mesh tables of cp-mesh.
//!
//! The renderer of E2 samples albedo only. The textures' normal and roughness channels are
//! generated with them; roughness enters the material table as the material's factor times the
//! texture's mean roughness, normals are not uploaded (cp-render has no binding for them yet).
//!
//! **Variety.** So that a meadow or a cliff is not one tile repeated in one orientation, the
//! renderer turns each cell's tile by a transform its material allows and shows one of a few
//! variants of its texture ([`cp_render::material::TextureTurn`], [`GpuMaterial::variation`]).
//! Which transforms a texture allows follows from what its recipe paints ([`texture_turn`]);
//! the variants are the same recipe with the same parameters and other seeds
//! ([`variant_count`]), made here and appended to the block texture array behind the pack's
//! own layers. They are render data only: the content registry does not know them.
//!
//! **Ground cover.** Behind the variants of `grass_top` follow its ground covers (the same
//! texture with the styles `flowers`, `blossoms`, `litter` and `straw`,
//! [`cp_render::material::variation::GROUND_COVER`]), which the renderer lays on the tops of
//! grass cells by the scene's ground map: fallen leaves under trees, straw in dry meadows,
//! patches of flowers in the open.

use std::path::{Path, PathBuf};

use cp_content::{BlockTables, MaterialFlags, Registry, TextureDef, TextureSource};
use cp_mesh::MeshTables;
use cp_proctex::{AlphaMode, PbrTexture};
use cp_render::gpu::GpuContext;
use cp_render::material::{GpuMaterial, MaterialTable, TextureTurn};
use cp_render::texture::{TextureArray, TextureArrayBuilder, TextureArrayKind};
use cp_voxel::VoxelTables;
use cp_worldgen::{PaletteKeys, TerrainPalette};

use crate::Failure;
use crate::content_check::{TEXTURE_SIZE, check_with_textures, default_pack_dir, make_texture};

/// Emission of a material whose texture glows everywhere at full strength, as a multiple of
/// the albedo the lighting pass adds. A sunlit white surface reflects about 1 (sun illuminance
/// over π), so lava (half of its texels glow) ends up about as bright as a sunlit wall.
pub const EMISSIVE_SCALE: f32 = 2.0;

/// Seed distance between a texture and its variants.
const VARIANT_SEED_STEP: u64 = 7919;

/// How the renderer may turn a texture of `recipe` from cell to cell: freely where the pattern
/// has no direction, along its axes where it has a grain or layers, only mirrored left to
/// right where it has an up side; recipes this table does not know stay fixed.
pub fn texture_turn(recipe: &str) -> TextureTurn {
    match recipe {
        "dirt" | "grass_top" | "gravel" | "snow" | "ice" | "limestone" | "water" | "lava"
        | "leaves" | "log_top" | "cobble" => TextureTurn::Free,
        "stone" | "sand" | "clay" | "slate" | "planks" | "ore" => TextureTurn::Axial,
        "grass_side" | "log_side" | "sandstone" | "basalt" => TextureTurn::Upright,
        _ => TextureTurn::Fixed,
    }
}

/// The ground cover styles of a recipe, in the order the renderer expects them
/// ([`cp_render::material::variation::GROUND_COVER`]); none for recipes without cover.
pub fn cover_styles(recipe: &str) -> &'static [&'static str] {
    match recipe {
        "grass_top" => &["flowers", "blossoms", "litter", "straw"],
        _ => &[],
    }
}

/// How many variants (other seeds) the renderer gets of a texture of `recipe`: three of the
/// ground that covers large areas, two of foliage and of bark (a trunk showed the same marks
/// every metre), none of the rest (cut wood, ores, details).
pub fn variant_count(recipe: &str) -> u32 {
    match recipe {
        "dirt" | "grass_top" | "grass_side" | "gravel" | "sand" | "snow" | "stone" | "clay"
        | "slate" | "limestone" | "sandstone" | "water" => 3,
        "leaves" | "log_side" => 2,
        _ => 0,
    }
}

/// The extra texture layers of one texture of the pack.
struct Variants {
    /// The pack's texture layer they vary.
    layer: usize,
    /// The variants, at the size of the texture: first the ones of other seeds, then the
    /// ground covers.
    textures: Vec<PbrTexture>,
    /// How many of them are variants of other seeds.
    seeds: u32,
    /// Whether the ground covers follow them.
    cover: bool,
}

/// The loaded content packs with their textures, ready to be turned into GPU resources.
pub struct Assets {
    registry: Registry,
    /// One texture per [`cp_core::ids::TextureLayer`], at the size it was made.
    textures: Vec<PbrTexture>,
    /// How the alpha channel of each texture is filtered into its mip chain.
    alpha: Vec<AlphaMode>,
    /// Variants of the textures that have any, in layer order; their layers follow the
    /// pack's in the block texture array.
    variants: Vec<Variants>,
    /// The pack directories in load order.
    packs: Vec<PathBuf>,
}

impl std::fmt::Debug for Assets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Assets")
            .field("packs", &self.packs)
            .field("states", &self.registry.state_count())
            .field("materials", &self.registry.materials().len())
            .field("textures", &self.textures.len())
            .finish()
    }
}

impl Assets {
    /// Loads `packs` in order (later packs override earlier ones), or the workspace's base
    /// pack when `packs` is empty; makes every texture and fills the material colours.
    ///
    /// # Errors
    /// A [`Failure`] when no base pack is found, the packs do not load, or a texture cannot be
    /// made.
    pub fn load(packs: &[PathBuf]) -> Result<Self, Failure> {
        let dirs = if packs.is_empty() {
            vec![default_pack_dir().ok_or_else(|| {
                Failure::new(
                    "cannot find the base pack of the workspace (content/base next to the \
                     executable's target directory)",
                )
            })?]
        } else {
            packs.to_vec()
        };
        let refs: Vec<&Path> = dirs.iter().map(PathBuf::as_path).collect();
        let mut registry = Registry::load(&refs)
            .map_err(|error| Failure::new(format!("the content packs do not load: {error}")))?;
        let (_, textures) = check_with_textures(&mut registry)?;
        let alpha = registry
            .textures()
            .iter()
            .zip(&textures)
            .enumerate()
            .map(|(layer, (def, texture))| alpha_mode(&registry, layer, def, texture))
            .collect();
        let variants = make_variants(&registry)?;
        log::info!(
            "content: {} states, {} materials, {} textures (and {} variants for the renderer) \
             from {}",
            registry.state_count(),
            registry.materials().len(),
            textures.len(),
            variants.iter().map(|v| v.textures.len()).sum::<usize>(),
            dirs.iter()
                .map(|dir| dir.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(Self {
            registry,
            textures,
            alpha,
            variants,
            packs: dirs,
        })
    }

    /// The registry, material colours filled in.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// The per-state tables of the registry.
    pub fn tables(&self) -> &BlockTables {
        self.registry.tables()
    }

    /// The pack directories in load order.
    pub fn packs(&self) -> &[PathBuf] {
        &self.packs
    }

    /// The texture of layer `layer` as made (level 0).
    ///
    /// # Panics
    /// If `layer` is not a texture layer of the registry.
    pub fn texture(&self, layer: usize) -> &PbrTexture {
        &self.textures[layer]
    }

    /// How the alpha of texture `layer` is filtered.
    ///
    /// # Panics
    /// If `layer` is not a texture layer of the registry.
    pub fn alpha_mode(&self, layer: usize) -> AlphaMode {
        self.alpha[layer]
    }

    /// The material table entries, indexed by [`cp_core::ids::MaterialId`]: the texture's
    /// linear average albedo (what far LOD levels draw; of a `grass_side` texture the average
    /// of its fringe rows, `top_rows_average`), roughness = the material's factor
    /// times the texture's mean roughness, emission = the material's factor times the share of
    /// glowing texels times [`EMISSIVE_SCALE`], metalness, texture layer, tint class, and the
    /// variation: how its texture may turn ([`texture_turn`]), its variants, and whether it is
    /// foliage (cutout materials and foliage-tinted ones).
    pub fn gpu_materials(&self) -> Vec<GpuMaterial> {
        // The first layer behind the pack's of each texture's variants.
        let mut next = self.textures.len() as u32;
        let mut first_variant = vec![(0_u32, 0_u32, false); self.textures.len()];
        for variants in &self.variants {
            first_variant[variants.layer] = (next, variants.seeds, variants.cover);
            next += variants.textures.len() as u32;
        }
        self.registry
            .materials()
            .iter()
            .map(|material| {
                let layer = material.texture.index();
                let texture = &self.textures[layer];
                let recipe = match &self.registry.textures()[layer].source {
                    TextureSource::Procedural { recipe, .. } => recipe.as_str(),
                    TextureSource::File { .. } => "",
                };
                // The far levels draw the side of a grass cell in the colour of its fringe, as
                // the textured levels do beyond 80 m (the G-buffer shader squeezes the fringe
                // over the face): from afar the one-metre steps of a hillside are grass, not
                // brown contour lines.
                let average = if recipe == "grass_side" {
                    top_rows_average(texture)
                } else {
                    cp_proctex::average_color(texture)
                }
                .map(|c| c.clamp(0.0, 1.0));
                let foliage = material.flags.contains(MaterialFlags::CUTOUT)
                    || material.tint == cp_content::TintClass::Foliage;
                let (variant_layer, variants, cover) = first_variant[layer];
                let cover_bit = if cover {
                    cp_render::material::variation::GROUND_COVER
                } else {
                    0
                };
                GpuMaterial {
                    average_color: average,
                    roughness: (material.roughness * mean_unit(&texture.roughness)).clamp(0.0, 1.0),
                    emissive: (material.emissive * mean_unit(&texture.emissive) * EMISSIVE_SCALE)
                        .max(0.0),
                    metallic: material.metallic.clamp(0.0, 1.0),
                    texture_layer: material.texture.0.into(),
                    tint_class: material.tint as u32,
                    variation: GpuMaterial::variation_bits(texture_turn(recipe), foliage, variants)
                        | cover_bit,
                    variant_layer,
                    reserved: [0, 0],
                }
            })
            .collect()
    }

    /// The material table on the GPU.
    ///
    /// # Errors
    /// A [`Failure`] when cp-render refuses the table.
    pub fn material_table(&self, gpu: &GpuContext) -> Result<MaterialTable, Failure> {
        MaterialTable::new(gpu, &self.gpu_materials())
            .map_err(|error| Failure::new(format!("cannot create the material table: {error}")))
    }

    /// The block texture array (sRGB albedo, full mip chains filtered by each texture's alpha
    /// mode), layer `i` = texture layer `i`, then the variants of the textures that have any,
    /// in layer order (where [`Self::gpu_materials`] points their materials).
    ///
    /// # Errors
    /// A [`Failure`] when the textures have different sizes or the GPU refuses the array.
    pub fn albedo_array(&self, gpu: &GpuContext) -> Result<TextureArray, Failure> {
        let size = self.textures.first().map_or(16, |texture| texture.size);
        let mut builder =
            TextureArrayBuilder::new("block albedo", TextureArrayKind::Albedo, size, size)
                .map_err(|error| Failure::new(format!("block textures: {error}")))?;
        let variants = self
            .variants
            .iter()
            .flat_map(|variants| variants.textures.iter().map(|t| (variants.layer, t)));
        let layers = self.textures.iter().enumerate().chain(variants);
        for (layer, texture) in layers {
            let key = &self.registry.textures()[layer].key;
            if texture.size != size {
                return Err(Failure::new(format!(
                    "texture `{key}` is {} px, the block texture array holds {size} px layers",
                    texture.size
                )));
            }
            let chain = cp_proctex::mip_chain(texture, self.alpha[layer]);
            let mips: Vec<&[u8]> = chain.iter().map(|mip| mip.albedo.as_flattened()).collect();
            builder
                .push_layer(&mips)
                .map_err(|error| Failure::new(format!("texture `{key}`: {error}")))?;
        }
        builder
            .build(gpu)
            .map_err(|error| Failure::new(format!("cannot create the block textures: {error}")))
    }

    /// The terrain palette of the test generators: the base pack's states
    /// ([`PaletteKeys::BASE`]), with the key hash that enters the generator identities.
    ///
    /// # Errors
    /// A [`Failure`] naming the role whose state the packs do not define.
    pub fn palette(&self) -> Result<TerrainPalette, Failure> {
        TerrainPalette::resolve(&PaletteKeys::BASE, |key| self.registry.state(key))
            .map_err(|error| Failure::new(format!("terrain palette: {error}")))
    }

    /// The flag and LOD tables of cp-voxel.
    pub fn voxel_tables(&self) -> VoxelTables {
        let tables = self.registry.tables();
        VoxelTables::new(tables.flags.clone(), tables.lod.clone())
    }

    /// The tables of cp-mesh, borrowed from the registry.
    pub fn mesh_tables(&self) -> MeshTables<'_> {
        mesh_tables(self.registry.tables())
    }
}

/// The tables of cp-mesh over a bundle of registry tables.
pub fn mesh_tables(tables: &BlockTables) -> MeshTables<'_> {
    MeshTables {
        flags: &tables.flags,
        render: &tables.render,
        cull_group: &tables.cull_group,
    }
}

/// The variants of every procedural texture whose recipe has any ([`variant_count`]): the same
/// recipe and parameters with the seeds `seed + k * VARIANT_SEED_STEP`; then, for recipes with
/// ground cover ([`cover_styles`]), the texture with each cover style.
fn make_variants(registry: &Registry) -> Result<Vec<Variants>, Failure> {
    let mut all = Vec::new();
    for (layer, def) in registry.textures().iter().enumerate() {
        let TextureSource::Procedural { recipe, seed, .. } = &def.source else {
            continue;
        };
        let count = variant_count(recipe);
        let styles = cover_styles(recipe);
        if count == 0 && styles.is_empty() {
            continue;
        }
        let mut textures = (1..=u64::from(count))
            .map(|k| {
                let mut variant = def.clone();
                if let TextureSource::Procedural { seed: other, .. } = &mut variant.source {
                    *other = seed.wrapping_add(k * VARIANT_SEED_STEP);
                }
                make_texture(&variant).map_err(|reason| {
                    Failure::new(format!("variant {k} of texture `{}`: {reason}", def.key))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        for style in styles {
            textures.push(make_cover(def, style).map_err(|reason| {
                Failure::new(format!("{style} cover of texture `{}`: {reason}", def.key))
            })?);
        }
        all.push(Variants {
            layer,
            textures,
            seeds: count,
            cover: !styles.is_empty(),
        });
    }
    Ok(all)
}

/// The procedural texture `def` with the cover style `style` (a variant parameter of its
/// recipe, [`cover_styles`]), at the size of the pack's textures.
fn make_cover(def: &TextureDef, style: &str) -> Result<PbrTexture, String> {
    let TextureSource::Procedural {
        recipe,
        params,
        seed,
    } = &def.source
    else {
        return Err("only procedural textures have ground cover".to_owned());
    };
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
    recipe_params.set("style", cp_proctex::ParamValue::Variant(style.to_owned()));
    cp_proctex::generate(recipe, &recipe_params, *seed, TEXTURE_SIZE)
        .map_err(|error| error.to_string())
}

/// The linear average albedo (alpha-weighted, before any tint) of the top eighth of the rows
/// of `texture`: of a `grass_side` texture the rows that are grass in every column (its fringe
/// is at least two of 16 rows deep).
fn top_rows_average(texture: &PbrTexture) -> [f32; 3] {
    let size = texture.size as usize;
    let rows = (size / 8).max(1);
    let top = PbrTexture {
        albedo: texture.albedo.iter().take(rows * size).copied().collect(),
        ..texture.clone()
    };
    cp_proctex::average_color(&top)
}

/// Mean of byte values as a fraction of 255; 0 for no values.
fn mean_unit(values: &[u8]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: u64 = values.iter().map(|&value| u64::from(value)).sum();
    (sum as f64 / (values.len() as f64 * 255.0)) as f32
}

/// The alpha mode of texture `layer`: the recipe's for procedural textures; for image files
/// coverage when a cutout material uses it, blending when a translucent one does and the image
/// has translucent texels, opaque otherwise.
fn alpha_mode(
    registry: &Registry,
    layer: usize,
    def: &TextureDef,
    texture: &PbrTexture,
) -> AlphaMode {
    if let TextureSource::Procedural { recipe, .. } = &def.source
        && let Some(info) = cp_proctex::recipe_info(recipe)
    {
        return info.alpha_mode();
    }
    let users = || {
        registry
            .materials()
            .iter()
            .filter(move |material| material.texture.index() == layer)
    };
    if users().any(|material| material.flags.contains(MaterialFlags::CUTOUT)) {
        AlphaMode::Coverage
    } else if users().any(|material| material.flags.contains(MaterialFlags::TRANSLUCENT))
        && texture.albedo.iter().any(|texel| texel[3] < u8::MAX)
    {
        AlphaMode::Blend
    } else {
        AlphaMode::Opaque
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_base_pack_gives_valid_materials_and_a_palette() {
        let assets = Assets::load(&[]).expect("the base pack loads");
        let materials = assets.gpu_materials();
        assert_eq!(materials.len(), assets.registry().materials().len());
        for (index, material) in materials.iter().enumerate() {
            material.validate(index).expect("every entry is in range");
            assert!(
                (material.texture_layer as usize) < assets.registry().textures().len(),
                "{index}"
            );
        }
        // Lava glows, stone does not; grass is tinted, stone is not.
        let id = |key: &str| assets.registry().material(key).expect(key).index();
        assert!(materials[id("cubeplanet:lava")].emissive > 0.1);
        assert_eq!(materials[id("cubeplanet:stone")].emissive, 0.0);
        assert_eq!(materials[id("cubeplanet:grass_top")].tint_class, 1);
        assert_eq!(materials[id("cubeplanet:leaves_oak")].tint_class, 2);
        assert_eq!(materials[id("cubeplanet:water")].tint_class, 3);
        assert_eq!(materials[id("cubeplanet:stone")].tint_class, 0);
        // Leaves are cutout textures, filtered by coverage.
        let leaves = assets
            .registry()
            .texture("cubeplanet:leaves_oak")
            .expect("leaves");
        assert_eq!(assets.alpha_mode(leaves.index()), AlphaMode::Coverage);
        let stone = assets
            .registry()
            .texture("cubeplanet:stone")
            .expect("stone");
        assert_eq!(assets.alpha_mode(stone.index()), AlphaMode::Opaque);

        let palette = assets
            .palette()
            .expect("the base pack has every terrain role");
        assert_eq!(
            Some(palette.grass),
            assets.registry().state("cubeplanet:grass")
        );
        assert_ne!(palette.key_hash, 0);
        assert_eq!(assets.voxel_tables().len(), assets.registry().state_count());
        assert_eq!(
            assets.mesh_tables().flags.len(),
            assets.registry().state_count()
        );
    }

    #[test]
    fn materials_vary_by_what_their_textures_paint() {
        use cp_render::material::variation;
        let assets = Assets::load(&[]).expect("the base pack loads");
        let materials = assets.gpu_materials();
        let id = |key: &str| assets.registry().material(key).expect(key).index();
        let turn = |key: &str| materials[id(key)].variation & variation::TURN_MASK;
        let variants = |key: &str| materials[id(key)].variation >> variation::VARIANT_SHIFT;
        let foliage = |key: &str| materials[id(key)].variation & variation::FOLIAGE != 0;
        // Directionless ground turns freely, ripples keep their axes, the fringe its up side.
        assert_eq!(turn("cubeplanet:grass_top"), TextureTurn::Free as u32);
        assert_eq!(turn("cubeplanet:sand"), TextureTurn::Axial as u32);
        assert_eq!(turn("cubeplanet:grass_side"), TextureTurn::Upright as u32);
        assert_eq!(turn("cubeplanet:log_oak_side"), TextureTurn::Upright as u32);
        assert_eq!(variants("cubeplanet:grass_top"), 3);
        assert_eq!(variants("cubeplanet:leaves_birch"), 2);
        assert_eq!(variants("cubeplanet:log_birch_side"), 2);
        assert_eq!(variants("cubeplanet:planks_oak"), 0);
        assert!(foliage("cubeplanet:leaves_oak") && foliage("cubeplanet:leaves_cherry"));
        assert!(!foliage("cubeplanet:grass_top"));
        // The variants follow the pack's layers in the texture array, each where its
        // material points.
        let layers = assets.registry().textures().len() as u32;
        let total: u32 = assets
            .variants
            .iter()
            .map(|v| v.textures.len() as u32)
            .sum();
        for material in &materials {
            let count = material.variation >> variation::VARIANT_SHIFT;
            if count > 0 {
                assert!(material.variant_layer >= layers);
                assert!(material.variant_layer + count <= layers + total);
            }
        }
        assert!(total >= 30, "{total}");
    }

    #[test]
    fn mean_unit_is_a_fraction_of_full_scale() {
        assert_eq!(mean_unit(&[]), 0.0);
        assert_eq!(mean_unit(&[255, 255]), 1.0);
        assert!((mean_unit(&[0, 255]) - 0.5).abs() < 1e-6);
    }
}

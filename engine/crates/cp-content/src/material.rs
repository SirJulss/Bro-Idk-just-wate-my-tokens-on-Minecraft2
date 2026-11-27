//! Material definitions: what a block face looks like to the renderer.

use core::fmt;
use core::ops::{BitOr, BitOrAssign};

use cp_core::ids::TextureLayer;
use serde::Deserialize;

/// Which colour of its surroundings (body, climate) a material takes on.
///
/// The renderer multiplies that colour into the albedo, so textures of a tinted material are
/// painted in pale, near-neutral tones. A texture that carries its final colours belongs to a
/// material with [`TintClass::None`].
///
/// The discriminants are stable so the value can go into a GPU material table as is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum TintClass {
    /// The texture's own colours are used unchanged.
    #[default]
    None = 0,
    /// Tinted with the local grass colour.
    Grass = 1,
    /// Tinted with the local foliage colour.
    Foliage = 2,
    /// Tinted with the local water colour.
    Water = 3,
}

/// How the renderer has to treat a material beyond its numeric properties. A set of bits.
///
/// In TOML: `flags = ["cutout"]`, `flags = ["translucent"]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct MaterialFlags(u8);

impl MaterialFlags {
    /// The texture's alpha channel is a cut-out mask (leaves): texels are either fully there or
    /// fully absent, never blended.
    pub const CUTOUT: Self = Self(1 << 0);
    /// The surface lets light through (water): what lies behind it stays visible. Without this
    /// flag a fluid surface is drawn opaque (lava).
    pub const TRANSLUCENT: Self = Self(1 << 1);

    /// The flags of surfaces that do not hide what lies behind them. A block that is `opaque`
    /// cannot have such a material on any face.
    pub const SEE_THROUGH: Self = Self(Self::CUTOUT.0 | Self::TRANSLUCENT.0);

    /// Every flag with its TOML name, in bit order.
    pub const NAMED: [(&'static str, Self); 2] =
        [("cutout", Self::CUTOUT), ("translucent", Self::TRANSLUCENT)];

    /// No flag set.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// The raw bits, stable for GPU material tables.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// True when no flag is set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// True when every flag of `other` is set in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// True when at least one flag of `other` is set in `self`.
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// The flags of `self` and `other` together.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for MaterialFlags {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl BitOrAssign for MaterialFlags {
    fn bitor_assign(&mut self, other: Self) {
        *self = self.union(other);
    }
}

impl fmt::Debug for MaterialFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MaterialFlags(")?;
        let mut first = true;
        for (name, flag) in Self::NAMED {
            if self.contains(flag) {
                if !first {
                    f.write_str(" | ")?;
                }
                f.write_str(name)?;
                first = false;
            }
        }
        f.write_str(")")
    }
}

/// A material as defined by a content pack, with its texture resolved.
///
/// Obtained from [`Registry::materials`](crate::Registry::materials), where the index of a
/// material is its [`MaterialId`](cp_core::ids::MaterialId).
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialDef {
    /// The material's key, `namespace:name`.
    pub key: String,
    /// The texture of the surface: an index into
    /// [`Registry::textures`](crate::Registry::textures).
    pub texture: TextureLayer,
    /// Factor on the roughness stored in the texture, `0.0..=1.0`. 1, the default, leaves the
    /// surface as rough as it is painted; smaller values make it glossier. For a texture
    /// without a roughness channel it is the roughness itself.
    pub roughness: f32,
    /// Metalness of the surface, `0.0..=1.0`; default 0.
    pub metallic: f32,
    /// Factor on the emission stored in the texture, finite and `>= 0`. 0, the default, does
    /// not glow whatever the texture holds. The renderer's exposure model defines the absolute
    /// brightness.
    pub emissive: f32,
    /// Which environment colour tints the albedo.
    pub tint: TintClass,
    /// Rendering flags.
    pub flags: MaterialFlags,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_are_distinct_bits() {
        let mut union = MaterialFlags::empty();
        assert!(union.is_empty());
        for (index, (name, flag)) in MaterialFlags::NAMED.into_iter().enumerate() {
            assert_eq!(flag.bits(), 1 << index, "{name}");
            assert!(!union.contains(flag));
            union |= flag;
            assert!(union.contains(flag));
        }
        assert_eq!(union, MaterialFlags::CUTOUT | MaterialFlags::TRANSLUCENT);
        assert_eq!(union.bits(), 0b11);
        assert!(union.contains(MaterialFlags::empty()));
        assert!(!MaterialFlags::CUTOUT.contains(union));
        assert_eq!(MaterialFlags::default(), MaterialFlags::empty());
        assert_eq!(size_of::<MaterialFlags>(), 1);
    }

    #[test]
    fn flags_debug_lists_names() {
        assert_eq!(format!("{:?}", MaterialFlags::empty()), "MaterialFlags()");
        assert_eq!(
            format!("{:?}", MaterialFlags::TRANSLUCENT),
            "MaterialFlags(translucent)"
        );
        assert_eq!(
            format!("{:?}", MaterialFlags::CUTOUT | MaterialFlags::TRANSLUCENT),
            "MaterialFlags(cutout | translucent)"
        );
    }

    #[test]
    fn tint_classes_have_stable_numbers_and_names() {
        #[derive(Deserialize)]
        struct Holder {
            tint: TintClass,
        }
        for (text, expected, number) in [
            ("none", TintClass::None, 0),
            ("grass", TintClass::Grass, 1),
            ("foliage", TintClass::Foliage, 2),
            ("water", TintClass::Water, 3),
        ] {
            let holder: Holder = toml::from_str(&format!("tint = \"{text}\"")).unwrap();
            assert_eq!(holder.tint, expected);
            assert_eq!(expected as u8, number);
        }
        assert!(toml::from_str::<Holder>("tint = \"lava\"").is_err());
        assert_eq!(TintClass::default(), TintClass::None);
    }
}

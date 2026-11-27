//! Where a recipe paints: resolution, seed and the window onto the infinitely tiled plane.

use cp_core::hash::{Rng, hash_combine, hash_str};
use cp_core::math::DVec2;

use crate::grid::Grid;
use crate::pattern::TILE;

/// The surface a recipe paints on.
///
/// Recipes describe an infinite, periodic image in design units (16 per tile) and the canvas
/// samples a `size × size` window of it at the texel centres. The public API always uses the
/// window at the origin; the tests move it to prove that nothing depends on where the tile's
/// border happens to lie.
pub(crate) struct Canvas {
    /// Edge length of the texture in texels.
    pub size: usize,
    /// The caller's seed, before it is mixed with a recipe name.
    request_seed: u64,
    /// Seed of this recipe: request seed and recipe name.
    seed: u64,
    /// Texel offset of the window on the tiled plane.
    origin: (i64, i64),
}

impl Canvas {
    pub(crate) fn new(recipe: &str, request_seed: u64, size: usize, origin: (i64, i64)) -> Self {
        Self {
            size,
            request_seed,
            seed: hash_combine(hash_str(recipe), request_seed),
            origin,
        }
    }

    /// The canvas `recipe` gets for the same request. Lets one recipe continue the pattern of
    /// another: the soil under a grass fringe is the `dirt` texture of the same seed.
    pub(crate) fn of_recipe(&self, recipe: &str) -> Self {
        Self::new(recipe, self.request_seed, self.size, self.origin)
    }

    /// Design-space position of the centre of texel `(x, y)`. Not wrapped: a shifted window
    /// reports positions outside `[0, 16)`, and patterns must be periodic on their own.
    pub(crate) fn pos(&self, x: usize, y: usize) -> DVec2 {
        let step = TILE / self.size as f64;
        DVec2::new(
            ((x as i64 + self.origin.0) as f64 + 0.5) * step,
            ((y as i64 + self.origin.1) as f64 + 0.5) * step,
        )
    }

    /// Evaluates `f` at every texel centre.
    pub(crate) fn paint<T: Copy>(&self, mut f: impl FnMut(DVec2) -> T) -> Grid<T> {
        Grid::from_fn(self.size, |x, y| f(self.pos(x, y)))
    }

    /// Value of a 16×16 design layer under each texel centre: how hand-placed pixels reach
    /// textures of any resolution.
    pub(crate) fn stamp<T: Copy>(&self, layer: &Grid<T>) -> Grid<T> {
        self.paint(|p| layer.wrapped(p.x.floor() as i64, p.y.floor() as i64))
    }

    /// Texels per design unit: 1 at the native 16 px.
    pub(crate) fn density(&self) -> f64 {
        self.size as f64 / TILE
    }

    /// An independent seed for one noise layer of the recipe.
    pub(crate) fn seed(&self, layer: u64) -> u64 {
        hash_combine(self.seed, layer)
    }

    /// An independent random stream for one group of hand-placed details.
    pub(crate) fn rng(&self, stream: u64) -> Rng {
        Rng::new(self.seed, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texel_centres_are_in_design_units() {
        let native = Canvas::new("stone", 1, 16, (0, 0));
        assert_eq!(native.pos(0, 0), DVec2::new(0.5, 0.5));
        assert_eq!(native.pos(15, 3), DVec2::new(15.5, 3.5));
        assert_eq!(native.density(), 1.0);
        let fine = Canvas::new("stone", 1, 64, (0, 0));
        assert_eq!(fine.pos(0, 0), DVec2::new(0.125, 0.125));
        assert_eq!(fine.pos(63, 63), DVec2::new(15.875, 15.875));
        assert_eq!(fine.density(), 4.0);
        let moved = Canvas::new("stone", 1, 16, (-3, 40));
        assert_eq!(moved.pos(0, 0), DVec2::new(-2.5, 40.5));
    }

    #[test]
    fn seeds_depend_on_recipe_request_and_layer() {
        let a = Canvas::new("stone", 1, 16, (0, 0));
        let b = Canvas::new("dirt", 1, 16, (0, 0));
        let c = Canvas::new("stone", 2, 16, (0, 0));
        assert_ne!(a.seed(0), b.seed(0));
        assert_ne!(a.seed(0), c.seed(0));
        assert_ne!(a.seed(0), a.seed(1));
        assert_ne!(a.rng(0).next_u32(), a.rng(1).next_u32());
        // Neither the resolution nor the window changes what is painted.
        assert_eq!(a.seed(0), Canvas::new("stone", 1, 64, (5, 9)).seed(0));
        // Another recipe's canvas for the same request.
        assert_eq!(a.of_recipe("dirt").seed(3), b.seed(3));
        assert_eq!(a.of_recipe("dirt").size, 16);
    }

    #[test]
    fn stamps_follow_the_window() {
        let layer = Grid::from_fn(16, |x, y| (x + 16 * y) as u32);
        let native = Canvas::new("stone", 1, 16, (0, 0)).stamp(&layer);
        assert_eq!(native, layer);
        let fine = Canvas::new("stone", 1, 32, (0, 0)).stamp(&layer);
        assert_eq!(fine.get(0, 0), 0);
        assert_eq!(fine.get(1, 1), 0);
        assert_eq!(fine.get(2, 0), 1);
        assert_eq!(fine.get(31, 31), 255);
        let moved = Canvas::new("stone", 1, 16, (17, -1)).stamp(&layer);
        assert_eq!(moved.get(0, 0), 1 + 16 * 15);
    }
}

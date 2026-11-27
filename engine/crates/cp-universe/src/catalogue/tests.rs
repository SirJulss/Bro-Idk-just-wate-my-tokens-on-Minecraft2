//! Tests of the star catalogue: determinism, the fixed home system, the density model, the
//! stellar population and the colours.

use super::*;
use cp_core::hash::hash_combine;

const SEED: u64 = 0x00C0_FFEE;

fn neighbourhood() -> impl Iterator<Item = [i64; 3]> {
    (-1..=1).flat_map(|dx| {
        (-1..=1).flat_map(move |dy| {
            (-1..=1).map(move |dz| [HOME_CELL[0] + dx, HOME_CELL[1] + dy, HOME_CELL[2] + dz])
        })
    })
}

/// A hash over every bit of the entries of some cells.
fn digest(catalogue: &StarCatalogue, cells: impl Iterator<Item = [i64; 3]>) -> (u64, usize) {
    let mut hash = 0_u64;
    let mut count = 0;
    for cell in cells {
        for star in catalogue.cell_stars(cell) {
            hash = hash_combine(hash, star.id.0);
            for sector in star.position.sector {
                hash = hash_combine(hash, sector as u64);
            }
            for local in star.position.local.to_array() {
                hash = hash_combine(hash, local.to_bits());
            }
            hash = hash_combine(hash, star.temperature.to_bits());
            hash = hash_combine(hash, star.luminosity.to_bits());
            hash = hash_combine(hash, star.system_seed);
            count += 1;
        }
    }
    (hash, count)
}

#[test]
fn cells_are_deterministic_and_independent_of_the_query_order() {
    let catalogue = StarCatalogue::new(SEED);
    let forwards = digest(&catalogue, neighbourhood());
    let cells: Vec<_> = neighbourhood().collect();
    let backwards = digest(&catalogue, cells.iter().rev().copied());
    // Same stars, other order: the per-cell lists are equal.
    for cell in &cells {
        assert_eq!(catalogue.cell_stars(*cell), catalogue.cell_stars(*cell));
    }
    assert_eq!(forwards.1, backwards.1);
    // Another seed, another sky; the same seed in a new catalogue, the same sky.
    assert_eq!(digest(&StarCatalogue::new(SEED), neighbourhood()), forwards);
    assert_ne!(
        digest(&StarCatalogue::new(SEED + 1), neighbourhood()).0,
        forwards.0
    );
    assert_eq!(catalogue.seed(), SEED);
}

#[test]
fn catalogue_output_is_pinned() {
    // Contract 2.5: the generated stars are identical on every machine. Bump
    // CATALOGUE_VERSION and re-pin when the generation changes on purpose.
    let catalogue = StarCatalogue::new(SEED);
    let cells = neighbourhood().chain([[0, 0, 0], [100, -3, 40], [-2000, 5, 1500]]);
    let (hash, count) = digest(&catalogue, cells);
    assert_eq!((hash, count), PINNED_DIGEST, "{hash:#018x} {count}");
}

/// Pinned 2026-11-26 on x86_64 Windows (catalogue version 1).
const PINNED_DIGEST: (u64, usize) = (0xd488_1ae4_413c_874f, 981);

#[test]
fn the_home_system_is_fixed() {
    for seed in [SEED, 1, u64::MAX] {
        let catalogue = StarCatalogue::new(seed);
        let home = catalogue.home();
        assert_eq!(home.id, HOME_SYSTEM_ID);
        assert_eq!(home.position, HOME_POSITION);
        assert_eq!(home.temperature, SOLAR_TEMPERATURE);
        assert_eq!(home.solar_luminosities(), 1.0);
        assert_eq!(catalogue.cell_stars(HOME_CELL)[0], home);
        assert_eq!(catalogue.star(HOME_SYSTEM_ID), Some(home));
        // No generated star comes close.
        for cell in neighbourhood() {
            for star in catalogue.cell_stars(cell) {
                if star.id != HOME_SYSTEM_ID {
                    let distance = GalaxyPos::delta(&HOME_POSITION, &star.position).length();
                    assert!(distance >= HOME_EXCLUSION_RADIUS, "{star:?}");
                }
            }
        }
    }
    assert_eq!(cell_of(&HOME_POSITION), HOME_CELL);
    assert_eq!(HOME_CELL, [3509, 8, 0]);
    assert_eq!(split_system_id(HOME_SYSTEM_ID), (HOME_CELL, 0));
    // About 8.2 kpc from the centre, 20 pc above the plane.
    let offset = GalaxyPos::delta(&GalaxyPos::ORIGIN, &HOME_POSITION) / PARSEC;
    assert!((offset.x - 8195.0).abs() < 5.0, "{offset:?}");
    assert!((offset.y - 19.95).abs() < 0.1, "{offset:?}");
}

#[test]
fn ids_name_cell_and_index() {
    let catalogue = StarCatalogue::new(SEED);
    for cell in neighbourhood() {
        let origin = cell_origin(cell);
        for (position, star) in catalogue.cell_stars(cell).into_iter().enumerate() {
            let (id_cell, index) = split_system_id(star.id);
            assert_eq!(id_cell, cell);
            assert_eq!(system_id(cell, index), Some(star.id));
            assert_eq!(cell_of(&star.position), cell, "{star:?}");
            let inside = GalaxyPos::delta(&origin, &star.position);
            assert!(inside.min_element() >= 0.0 && inside.max_element() < CELL_SIZE);
            assert_eq!(catalogue.star(star.id), Some(star));
            if position > 0 {
                assert!(index > 0, "generated stars start at index 1");
            }
        }
    }
    // Negative cells round-trip, cells beyond the range have no id and no stars.
    let cell = [-32_768, 32_767, -1];
    assert_eq!(split_system_id(system_id(cell, 513).unwrap()), (cell, 513));
    assert_eq!(system_id([40_000, 0, 0], 1), None);
    assert!(catalogue.cell_stars([40_000, 0, 0]).is_empty());
    assert_eq!(
        catalogue.star(SystemId(pack_system_id([5, 5, 5], 0).0)),
        None
    );
}

#[test]
fn counts_follow_the_density_model() {
    let catalogue = StarCatalogue::new(SEED);
    // In the solar neighbourhood about 1.3 stars per cell (0.1 per cubic parsec).
    let local = catalogue.expected_stars(HOME_CELL);
    assert!((0.4..3.0).contains(&local), "{local}");
    // The mean over many cells of equal expectation matches it (Poisson, 4 sigma).
    let mut total = 0.0;
    let mut expected = 0.0;
    for k in 0..2000_i64 {
        let cell = [
            HOME_CELL[0] + 3 + (k % 50),
            HOME_CELL[1],
            HOME_CELL[2] + 3 + k / 50,
        ];
        expected += catalogue.expected_stars(cell);
        total += catalogue.cell_stars(cell).len() as f64;
    }
    assert!(
        (total - expected).abs() < 4.0 * expected.sqrt(),
        "{total} vs {expected}"
    );
    // Dense towards the centre, thin above the disc, empty outside the galaxy.
    let centre = catalogue.expected_stars([0, 0, 0]);
    assert!(centre > 100.0 * local, "{centre}");
    let large: usize = (0..20).map(|k| catalogue.cell_stars([k, 0, 0]).len()).sum();
    let large_expected: f64 = (0..20).map(|k| catalogue.expected_stars([k, 0, 0])).sum();
    assert!((large as f64 - large_expected).abs() < 4.0 * large_expected.sqrt());
    let above = catalogue.expected_stars([HOME_CELL[0], HOME_CELL[1] + 400, HOME_CELL[2]]);
    assert!(above < 0.1 * local, "{above}");
    let height = (GALAXY_HALF_THICKNESS * PARSEC / CELL_SIZE) as i64 + 2;
    assert_eq!(catalogue.expected_stars([HOME_CELL[0], height, 0]), 0.0);
    assert_eq!(star_density(DVec3::new(1.0e22, 0.0, 0.0)), 0.0);
    // The spiral arms modulate the disc at constant radius.
    let ring: Vec<f64> = (0..72)
        .map(|k| {
            let angle = f64::from(k) * TAU / 72.0;
            let (sin, cos) = det::sin_cos(angle);
            star_density(DVec3::new(cos, 0.0, sin) * (6_000.0 * PARSEC))
        })
        .collect();
    let (low, high) = ring
        .iter()
        .fold((f64::INFINITY, 0.0_f64), |(low, high), &d| {
            (det::min(low, d), det::max(high, d))
        });
    assert!(high > 2.0 * low, "{low} {high}");
}

#[test]
fn stars_are_plausible() {
    let catalogue = StarCatalogue::new(SEED);
    let mut stars = Vec::new();
    for k in 0..2000 {
        let cell = [
            HOME_CELL[0] + k % 40,
            HOME_CELL[1],
            HOME_CELL[2] + 5 + k / 40,
        ];
        stars.extend(catalogue.cell_stars(cell));
    }
    assert!(stars.len() > 1000, "{}", stars.len());
    let (mut dwarfs, mut bright, mut white_dwarfs) = (0_u32, 0_u32, 0_u32);
    for star in &stars {
        assert!((2_000.0..62_000.0).contains(&star.temperature), "{star:?}");
        let solar = star.solar_luminosities();
        assert!((1e-4..1e7).contains(&solar), "{star:?}");
        if star.temperature < 4_000.0 && solar < 0.1 {
            dwarfs += 1;
        }
        if solar > 10.0 {
            bright += 1;
        }
        if star.temperature > 5_000.0 && solar < 0.011 {
            white_dwarfs += 1;
        }
    }
    // Red dwarfs dominate; about 0.8 % are brighter than ten Suns and a few per cent are white
    // dwarfs, as around the Sun.
    let count = stars.len() as f64;
    assert!(f64::from(dwarfs) / count > 0.5, "{dwarfs}");
    let bright_share = f64::from(bright) / count;
    assert!((0.002..0.02).contains(&bright_share), "{bright_share}");
    let white_share = f64::from(white_dwarfs) / count;
    assert!((0.02..0.10).contains(&white_share), "{white_share}");
    // System seeds differ.
    let mut seeds: Vec<u64> = stars.iter().map(|star| star.system_seed).collect();
    seeds.sort_unstable();
    seeds.dedup();
    assert_eq!(seeds.len(), stars.len());
}

#[test]
fn stars_within_a_radius() {
    let catalogue = StarCatalogue::new(SEED);
    let radius = 6.0 * PARSEC;
    let near = catalogue.stars_within(&HOME_POSITION, radius);
    assert_eq!(near[0], catalogue.home());
    // Sorted by distance, all within the radius.
    let distances: Vec<f64> = near
        .iter()
        .map(|star| GalaxyPos::delta(&HOME_POSITION, &star.position).length())
        .collect();
    assert!(distances.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!(distances.iter().all(|&distance| distance <= radius));
    // Brute force over a larger block of cells finds the same set.
    let mut reference: Vec<SystemId> = Vec::new();
    for dx in -4..=4 {
        for dy in -4..=4 {
            for dz in -4..=4 {
                let cell = [HOME_CELL[0] + dx, HOME_CELL[1] + dy, HOME_CELL[2] + dz];
                for star in catalogue.cell_stars(cell) {
                    if GalaxyPos::delta(&HOME_POSITION, &star.position).length() <= radius {
                        reference.push(star.id);
                    }
                }
            }
        }
    }
    let mut found: Vec<SystemId> = near.iter().map(|star| star.id).collect();
    found.sort_unstable();
    reference.sort_unstable();
    assert_eq!(found, reference);
    assert!(found.len() > 10, "{}", found.len());
    assert_eq!(
        catalogue.stars_within(&HOME_POSITION, 0.0),
        [catalogue.home()]
    );
}

#[test]
#[should_panic(expected = "stars_within: radius")]
fn stars_within_rejects_huge_radii() {
    let _ = StarCatalogue::new(SEED).stars_within(&HOME_POSITION, 2.0 * MAX_QUERY_RADIUS);
}

#[test]
fn blackbody_colours() {
    let white = blackbody_color(6_500.0);
    assert!(white.min_element() > 0.9, "{white:?}");
    let sun = blackbody_color(SOLAR_TEMPERATURE);
    assert!(sun.x == 1.0 && sun.z < sun.y && sun.z > 0.7, "{sun:?}");
    let red = blackbody_color(3_000.0);
    assert!(red.x == 1.0 && red.z < 0.4, "{red:?}");
    let blue = blackbody_color(20_000.0);
    assert!(blue.z == 1.0 && blue.x < 0.85, "{blue:?}");
    // Hotter is bluer, all the way (below 2000 K blue leaves the sRGB gamut).
    let mut previous = 0.0;
    let mut temperature = 2_000.0;
    while temperature < 40_000.0 {
        let color = blackbody_color(temperature);
        let blueness = color.z / color.x;
        assert!(blueness > previous, "{temperature}");
        assert!(color.max_element() == 1.0 && color.min_element() >= 0.0);
        previous = blueness;
        temperature *= 1.1;
    }
    let star = StarCatalogue::new(SEED).home();
    assert_eq!(star.color(), sun);
}

#[test]
fn units() {
    assert!((PARSEC / 3.0857e16 - 1.0).abs() < 1e-4);
    assert_eq!(CELL_SIZE, (1_u64 << 56) as f64);
    assert!((CELL_SIZE / PARSEC - 2.335).abs() < 0.001);
    assert_eq!(MAX_STARS_PER_CELL, 65_535);
}

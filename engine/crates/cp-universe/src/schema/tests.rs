//! Tests of the system file schema: the fixture file reads as the fixture built in code, and
//! every schema or rule violation names the file, the line, the definition and the key.

use super::*;
use crate::fixtures;

const FILE: &str = "systems/test.toml";

fn parse(text: &str) -> Result<SystemDef, UniverseError> {
    SystemDef::from_toml(text, Path::new(FILE))
}

/// The fixture file with the first occurrence of `old` replaced by `new`.
fn edited(old: &str, new: &str) -> String {
    assert!(fixtures::TOML.contains(old), "the fixture contains {old:?}");
    fixtures::TOML.replacen(old, new, 1)
}

/// 1-based line of the first occurrence of `needle` in `text`.
fn line_of(text: &str, needle: &str) -> u32 {
    let at = text.find(needle).expect("the needle occurs in the text");
    u32::try_from(text[..at].matches('\n').count() + 1).unwrap()
}

struct Failure {
    line: u32,
    item: String,
    key: Option<String>,
    message: String,
    text: String,
}

fn failure(text: &str) -> Failure {
    match parse(text).expect_err("the file must be rejected") {
        error @ UniverseError::Definition { .. } => {
            let rendered = error.to_string();
            let UniverseError::Definition {
                file,
                line,
                item,
                key,
                message,
            } = error
            else {
                unreachable!()
            };
            assert_eq!(file, Path::new(FILE));
            Failure {
                line,
                item,
                key,
                message,
                text: rendered,
            }
        }
        other => panic!("expected a definition error, got {other}"),
    }
}

/// Checks a rejected edit: the error is at the line of `at` in the edited text.
fn assert_failure(text: &str, at: &str, item: &str, key: &str, message: &str) {
    let failure = failure(text);
    assert_eq!(failure.line, line_of(text, at), "{}", failure.text);
    assert_eq!(failure.item, item, "{}", failure.text);
    assert_eq!(failure.key.as_deref(), Some(key), "{}", failure.text);
    assert!(
        failure.message.contains(message),
        "expected {message:?} in {}",
        failure.text
    );
    assert!(
        failure
            .text
            .starts_with(&format!("{FILE}:{}: ", failure.line))
    );
}

#[test]
fn the_fixture_file_reads_as_the_fixture_built_in_code() {
    let parsed = parse(fixtures::TOML).unwrap();
    assert_eq!(parsed, fixtures::system());
}

#[test]
fn defaults_of_optional_keys() {
    let text = edited("generator_seed = 7\ntidally_locked", "tidally_locked");
    let system = parse(&text).unwrap();
    let moon = system.body(BodyId(2)).cube().unwrap();
    assert_eq!(moon.generator_seed, hash_str("test:moon"));
    let outer = system.body(BodyId(3));
    assert_eq!(
        outer.rotation,
        Rotation::Uniform {
            period: 10_800.0,
            phase: 0.0
        }
    );
    // Integers are numbers too, and `tidally_locked = false` is a uniform rotation.
    let text = edited(
        "rotation_period_s = 10800.0",
        "rotation_period_s = 10800\ntidally_locked = false",
    );
    assert_eq!(parse(&text).unwrap(), fixtures::system());
}

#[test]
fn unknown_keys_are_errors() {
    let text = edited(
        "axial_tilt_deg = 23.5",
        "axial_tilt_deg = 23.5\ncolour = \"blue\"",
    );
    assert_failure(
        &text,
        "colour",
        "body `test:home`",
        "colour",
        "unknown key; expected one of `key`, `display_name`",
    );
    let text = edited(
        "mean_anomaly_deg = 270.0",
        "mean_anomaly_deg = 270.0\nlongitude_deg = 3",
    );
    assert_failure(
        &text,
        "longitude_deg",
        "body `test:home`",
        "orbit.longitude_deg",
        "unknown key; expected one of `semi_major_axis_m`",
    );
    let text = edited(
        "display_name = \"Test system\"",
        "display_name = \"Test system\"\nauthor = \"me\"",
    );
    assert_failure(&text, "author", "system", "author", "unknown key");
    let text = format!("{}\n[planets]\nx = 1\n", fixtures::TOML);
    assert_failure(&text, "[planets]", "top level", "planets", "unknown key");
    // The inline orbit table of the moon is checked the same way.
    let text = edited(
        "mean_anomaly_deg = 270.0 }",
        "mean_anomaly_deg = 270.0, extra = 1 }",
    );
    assert_failure(
        &text,
        "orbit = {",
        "body `test:moon`",
        "orbit.extra",
        "unknown key",
    );
}

#[test]
fn missing_keys_and_wrong_types_are_errors() {
    let text = edited("surface_gravity_m_s2 = 9.81\n", "");
    let header = "[[body]]\nkey = \"test:home\"";
    assert_failure(
        &text,
        header,
        "body `test:home`",
        "surface_gravity_m_s2",
        "missing",
    );
    let text = edited("edge_m = 40_960", "edge_m = \"big\"");
    assert_failure(
        &text,
        "edge_m",
        "body `test:home`",
        "edge_m",
        "must be an integer, found a string",
    );
    let text = edited("edge_m = 40_960", "edge_m = 40960.0");
    assert_failure(
        &text,
        "edge_m = 4",
        "body `test:home`",
        "edge_m",
        "found a float",
    );
    let text = edited("edge_m = 40_960", "edge_m = -2");
    assert_failure(
        &text,
        "edge_m = -2",
        "body `test:home`",
        "edge_m",
        "must be a positive number of metres, found -2",
    );
    let text = edited("grid_offset = 1365", "grid_offset = 3000000000");
    assert_failure(
        &text,
        "grid_offset",
        "body `test:home`",
        "grid_offset",
        "32-bit",
    );
    let text = edited("kind = \"planet\"", "kind = \"comet\"");
    assert_failure(
        &text,
        "\"comet\"",
        "body `test:home`",
        "kind",
        "not a body kind",
    );
    let text = edited("surface_gravity_m_s2 = 9.81", "surface_gravity_m_s2 = true");
    assert_failure(
        &text,
        "= true",
        "body `test:home`",
        "surface_gravity_m_s2",
        "must be a number, found a boolean",
    );
    let text = edited(
        "rotation_period_s = 43_200",
        "rotation_period_s = 9007199254740993",
    );
    assert_failure(
        &text,
        "9007199254740993",
        "body `test:sun`",
        "rotation_period_s",
        "too large to be represented exactly",
    );
    let text = edited("generator_seed = 7", "generator_seed = -7");
    assert_failure(
        &text,
        "-7",
        "body `test:home`",
        "generator_seed",
        "at least 0",
    );
    let text = edited("tidally_locked = true", "tidally_locked = 1");
    assert_failure(
        &text,
        "= 1\n",
        "body `test:moon`",
        "tidally_locked",
        "a boolean",
    );
    let text = edited("axial_tilt_deg = 0", "axial_tilt_deg = 0\norbit = 5");
    assert_failure(
        &text,
        "orbit = 5",
        "body `test:sun`",
        "orbit",
        "must be a table",
    );
    // (Keys after a header belong to its table, so a top-level key comes first.)
    let text = "body = 5\n[system]\nkey = \"a:b\"\ndisplay_name = \"A\"\n";
    assert_failure(text, "body", "top level", "body", "an array of tables");
    let text = "[[body]]\nkey = \"a:b\"\n";
    assert_failure(text, "[[body]]", "top level", "system", "missing");
    // A body whose key is not even a string is named by its index.
    let text = edited("key = \"test:moon\"", "key = 5");
    assert_failure(&text, "key = 5", "body #2", "key", "must be a string");
}

#[test]
fn keys_that_do_not_belong_to_the_kind_are_errors() {
    let text = edited("axial_tilt_deg = 0", "axial_tilt_deg = 0\nedge_m = 10");
    assert_failure(
        &text,
        "edge_m = 10",
        "body `test:sun`",
        "edge_m",
        "only planets and moons",
    );
    let text = edited(
        "axial_tilt_deg = 23.5",
        "axial_tilt_deg = 23.5\nradius_m = 5.0",
    );
    assert_failure(
        &text,
        "radius_m = 5.0",
        "body `test:home`",
        "radius_m",
        "only stars have a radius",
    );
    let text = edited("parent = \"test:sun\"", "parent = \"test:nowhere\"");
    assert_failure(
        &text,
        "test:nowhere",
        "body `test:home`",
        "parent",
        "no body of this file has the key `test:nowhere`",
    );
    let text = edited(
        "tidally_locked = true",
        "tidally_locked = true\nrotation_period_s = 100.0",
    );
    assert_failure(
        &text,
        "tidally_locked",
        "body `test:moon`",
        "tidally_locked",
        "remove `rotation_period_s`",
    );
    let text = edited(
        "tidally_locked = true",
        "tidally_locked = true\nrotation_phase_deg = 3.0",
    );
    assert_failure(
        &text,
        "rotation_phase_deg = 3.0",
        "body `test:moon`",
        "rotation_phase_deg",
        "no free phase",
    );
    let text = edited("rotation_period_s = 10800.0\n", "");
    assert_failure(
        &text,
        "[[body]]\nkey = \"test:outer\"",
        "body `test:outer`",
        "rotation_period_s",
        "tidally_locked = true",
    );
}

#[test]
fn rule_violations_point_at_the_offending_key() {
    let text = edited("edge_m = 40_960", "edge_m = 40_961");
    assert_failure(&text, "edge_m", "body `test:home`", "edge_m", "even");
    let text = edited("eccentricity = 0.03", "eccentricity = 1.5");
    assert_failure(
        &text,
        "eccentricity = 1.5",
        "body `test:outer`",
        "orbit.eccentricity",
        "below 1",
    );
    // A rule about the whole orbit points at its table.
    let text = edited(
        "semi_major_axis_m = 11_000_000.0",
        "semi_major_axis_m = 8_500_000.0",
    );
    let orbit_header = text.rfind("[body.orbit]").unwrap();
    let overlap = failure(&text);
    assert_eq!(overlap.key.as_deref(), Some("orbit"));
    assert_eq!(
        overlap.line,
        u32::try_from(text[..orbit_header].matches('\n').count() + 1).unwrap()
    );
    assert!(overlap.message.contains("overlaps"), "{}", overlap.text);
    let text = edited("key = \"test:system\"", "key = \"System\"");
    assert_failure(
        &text,
        "key = \"System\"",
        "system",
        "key",
        "not a valid key",
    );
    let text = edited("key = \"test:outer\"", "key = \"test:home\"");
    let duplicate = failure(&text);
    assert_eq!(
        duplicate.line,
        line_of(&text, "key = \"test:home\"\ndisplay_name = \"Outer\"")
    );
    assert!(
        duplicate.message.contains("defined twice"),
        "{}",
        duplicate.text
    );
    // A lock that cannot keep its face towards the parent points at `tidally_locked`.
    let text = edited("axial_tilt_deg = 7.5", "axial_tilt_deg = 89.0");
    assert_failure(
        &text,
        "tidally_locked",
        "body `test:moon`",
        "tidally_locked",
        "stays below 90 degrees",
    );
    // A parent defined after its child.
    let text = edited("parent = \"test:home\"", "parent = \"test:outer\"");
    assert_failure(
        &text,
        "parent = \"test:outer\"",
        "body `test:moon`",
        "parent",
        "must be defined before",
    );
}

#[test]
fn syntax_errors_name_line_and_column() {
    let text = edited("edge_m = 40_960", "edge_m = 40__960");
    match parse(&text).unwrap_err() {
        UniverseError::Syntax {
            file,
            line,
            column,
            message,
        } => {
            assert_eq!(file, Path::new(FILE));
            assert_eq!((line, column), (line_of(&text, "edge_m"), 10));
            assert!(message.contains("not a valid value"), "{message}");
        }
        other => panic!("expected a syntax error, got {other}"),
    }
}

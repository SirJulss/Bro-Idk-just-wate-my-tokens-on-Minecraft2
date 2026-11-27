//! System files on disk: directory loading, file order, duplicates and I/O errors with real
//! paths.

use std::fs;
use std::path::{Path, PathBuf};

use cp_universe::{SystemDef, UniverseError, load_systems};

/// A directory below cargo's scratch directory for integration tests, removed again on drop.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// `test` must be unique among the tests of this file; the process id keeps concurrent
    /// runs of the test binary apart.
    fn new(test: &str) -> Self {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("cp-universe-{test}-{}", std::process::id()));
        if path.exists() {
            fs::remove_dir_all(&path).expect("a stale test directory can be removed");
        }
        fs::create_dir_all(&path).expect("the test directory can be created");
        Self { path }
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path.join(name);
        fs::write(&path, bytes).expect("the file can be written");
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// A minimal system file: one star.
fn star_system(key: &str) -> String {
    format!(
        "[system]\nkey = \"{key}\"\ndisplay_name = \"Test\"\n\n[[body]]\nkey = \"test:star\"\n\
         display_name = \"Star\"\nkind = \"star\"\nradius_m = 1000.0\n\
         surface_gravity_m_s2 = 10.0\nrotation_period_s = 100.0\naxial_tilt_deg = 0.0\n"
    )
}

#[test]
fn a_directory_loads_in_file_name_order() {
    let dir = TempDir::new("order");
    dir.write("b.toml", star_system("test:second").as_bytes());
    dir.write("a.toml", star_system("test:first").as_bytes());
    dir.write("notes.txt", b"not a system");
    fs::create_dir_all(dir.path.join("sub.toml")).unwrap();
    let systems = load_systems(&dir.path).unwrap();
    let keys: Vec<&str> = systems.iter().map(SystemDef::key).collect();
    assert_eq!(keys, ["test:first", "test:second"]);
    assert_eq!(systems[0].body_count(), 1);
}

#[test]
fn duplicate_system_keys_are_rejected() {
    let dir = TempDir::new("duplicate");
    let first = dir.write("a.toml", star_system("test:same").as_bytes());
    let second = dir.write("b.toml", star_system("test:same").as_bytes());
    match load_systems(&dir.path).unwrap_err() {
        UniverseError::DuplicateSystem {
            key,
            file,
            first_file,
        } => {
            assert_eq!(key, "test:same");
            assert_eq!(file, second);
            assert_eq!(first_file, first);
        }
        other => panic!("expected a duplicate, got {other}"),
    }
}

#[test]
fn errors_name_the_real_file() {
    let dir = TempDir::new("errors");
    let broken = dir.write(
        "broken.toml",
        star_system("test:x")
            .replace("radius_m = 1000.0", "radius_m = -1.0")
            .as_bytes(),
    );
    let error = load_systems(&dir.path).unwrap_err();
    let text = error.to_string();
    assert!(
        text.starts_with(&format!("{}:9: ", broken.display())),
        "{text}"
    );
    assert!(
        text.contains("body `test:star`: `radius_m`: must be positive"),
        "{text}"
    );
    // Not UTF-8, not there.
    let invalid = dir.write("invalid.toml", &[0xff, 0xfe, 0x00]);
    assert!(matches!(
        SystemDef::load(&invalid),
        Err(UniverseError::Io { path, .. }) if path == invalid
    ));
    let missing = dir.path.join("missing.toml");
    assert!(matches!(
        SystemDef::load(&missing),
        Err(UniverseError::Io { path, .. }) if path == missing
    ));
    let gone = dir.path.join("no-such-directory");
    assert!(matches!(
        load_systems(&gone),
        Err(UniverseError::Io { path, .. }) if path == gone
    ));
    // An empty directory holds no systems.
    let empty = TempDir::new("empty");
    assert!(load_systems(&empty.path).unwrap().is_empty());
}

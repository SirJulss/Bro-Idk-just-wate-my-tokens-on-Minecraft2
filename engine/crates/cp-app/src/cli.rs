//! The command line of `cpe`: a small hand-written parser (the workspace has no CLI crate,
//! contract 3.2).
//!
//! Options take their value as the next argument (`--out caps.md`) or after `=`
//! (`--out=caps.md`). Paths may be any `OsString`; the `=` form needs valid Unicode. Every
//! problem is a [`UsageError`], never a panic.

use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

use cp_proctex::{MAX_SIZE, MIN_SIZE};
use cp_render::debug_view::DebugView;
use cp_render::voxel::DrawMode;

use crate::scene::SceneName;
use crate::scene::sky::Sun;

/// Name of the binary in the usage text.
pub const BIN: &str = "cpe";

/// Default texture size of `proctex-sheet`, in texels: the native size of the designs.
pub const DEFAULT_SHEET_SIZE: u32 = 16;
/// Default magnification of `proctex-sheet`, lowered automatically for large `--size`.
pub const DEFAULT_SHEET_SCALE: u32 = 8;
/// Widest tile of the contact sheet in pixels, `size * scale`; `cp_proctex::contact_sheet`
/// clamps the scale to it, so a larger request is rejected here instead of shrunk silently.
pub const MAX_SHEET_TILE: u32 = 256;

/// Default size of shots and benchmarks: 1920 x 1080 (contract §7: 1080p).
pub const DEFAULT_WIDTH: u32 = 1920;
/// See [`DEFAULT_WIDTH`].
pub const DEFAULT_HEIGHT: u32 = 1080;
/// Largest side of a shot or benchmark target in pixels.
pub const MAX_SIDE: u32 = 8192;
/// Default number of measured frames of `bench`.
pub const DEFAULT_BENCH_FRAMES: u32 = 300;

/// What the command line asks for.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Print the usage text: of everything (`None`) or of one subcommand.
    Help(Option<Subcommand>),
    /// `cpe caps`: the GPU capability report.
    Caps(CapsArgs),
    /// `cpe proctex-sheet`: the texture contact sheet.
    ProctexSheet(SheetArgs),
    /// `cpe content-check`: load and check content packs.
    ContentCheck(ContentCheckArgs),
    /// `cpe run`: a window with a free-fly camera.
    Run(RunArgs),
    /// `cpe shot`: one headless frame as PNG.
    Shot(ShotArgs),
    /// `cpe bench`: renderer measurements as JSON.
    Bench(BenchArgs),
}

/// The subcommands of `cpe`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Subcommand {
    /// `caps`
    Caps,
    /// `proctex-sheet`
    ProctexSheet,
    /// `content-check`
    ContentCheck,
    /// `run`
    Run,
    /// `shot`
    Shot,
    /// `bench`
    Bench,
}

impl Subcommand {
    /// Every subcommand, in the order of the usage text.
    pub const ALL: [Self; 6] = [
        Self::Caps,
        Self::ProctexSheet,
        Self::ContentCheck,
        Self::Run,
        Self::Shot,
        Self::Bench,
    ];

    /// The name on the command line.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Caps => "caps",
            Self::ProctexSheet => "proctex-sheet",
            Self::ContentCheck => "content-check",
            Self::Run => "run",
            Self::Shot => "shot",
            Self::Bench => "bench",
        }
    }

    /// The subcommand called `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|command| command.name() == name)
    }

    /// One-line synopsis: `cpe caps [--out <file.md>] [--json <file.json>]`.
    #[must_use]
    pub const fn synopsis(self) -> &'static str {
        match self {
            Self::Caps => "cpe caps [--out <file.md>] [--json <file.json>]",
            Self::ProctexSheet => {
                "cpe proctex-sheet --out <file.png> [--size <px>] [--scale <n>] [--pack <dir>]..."
            }
            Self::ContentCheck => "cpe content-check [--pack <dir>]...",
            Self::Run => "cpe run [--scene <name>] [--frames <n>]",
            Self::Shot => {
                "cpe shot --scene <name> --out <file.png> [--camera <n>] [--time <s>] \
                 [--sun <elevation>,<azimuth>] [--view <view>] [--path a|b|direct] \
                 [--width <px>] [--height <px>]"
            }
            Self::Bench => {
                "cpe bench --scene <name> --out <file.json> [--frames <n>] [--path a|b|direct] \
                 [--width <px>] [--height <px>]"
            }
        }
    }

    /// What the subcommand does and what its options mean, indented for the usage text.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Caps => {
                "    Opens the GPU without a window (Vulkan first, then DX12) and prints the\n    \
                 capability report as Markdown: adapter, draw path, quality tier, features,\n    \
                 limits.\n    \
                 --out <file.md>      also save the Markdown report\n    \
                 --json <file.json>   also save the report as JSON\n"
            }
            Self::ProctexSheet => {
                "    Writes the contact sheet of every procedural texture (recipes, tinted\n    \
                 recipes, presets) as an RGBA PNG. With --pack it shows instead the textures\n    \
                 of content packs as the game uses them: each texture's own recipe,\n    \
                 parameters and seed, labelled with its key, tinted like its material.\n    \
                 --out <file.png>     where to write the sheet (required)\n    \
                 --size <px>          texture size: 16, 32, 64, 128 or 256 (default 16)\n    \
                 --scale <n>          magnification (default 8, at most 256 / size)\n    \
                 --pack <dir>         a pack directory; repeat for several (later packs\n    \
                 \x20                    override earlier ones)\n"
            }
            Self::ContentCheck => {
                "    Loads the content packs (later packs override earlier ones), generates\n    \
                 every procedural texture with cp-proctex, reads every image texture, fills\n    \
                 the material colours and prints the counts, the content hash and the\n    \
                 average colour of every material. Fails if a pack does not load or a\n    \
                 texture cannot be made.\n    \
                 --pack <dir>         a pack directory; repeat for several (default: the\n    \
                 \x20                    base pack of this workspace, content/base)\n"
            }
            Self::Run => {
                "    Opens a window over a scene (default hills). Mouse and WASD fly the\n    \
                 camera (Space up, C down, Shift fast; the speed grows with the height),\n    \
                 F1 shows the HUD, F2 cycles the debug views, F3 the draw paths, F4 the\n    \
                 cameras of the scene, F5 flies the scene's flight path. A click captures\n    \
                 the mouse, Esc releases it, Esc again quits.\n    \
                 --scene <name>       hills, structures or calib\n    \
                 --frames <n>         quit after n frames once the world has loaded\n    \
                 \x20                    (smoke tests)\n"
            }
            Self::Shot => {
                "    Renders one frame of a scene camera without a window, after the camera's\n    \
                 world is generated and uploaded, and writes it as PNG.\n    \
                 --scene <name>       hills (cameras 0-7), structures (0-2), calib (0-6)\n    \
                 --out <file.png>     where to write the image (required)\n    \
                 --camera <n>         camera of the scene (default 0)\n    \
                 --time <s>           hills: the flight path at this time instead\n    \
                 --sun <el>,<az>      sun elevation and azimuth in degrees (default\n    \
                 \x20                    elevation 45, azimuth behind the camera)\n    \
                 --view <view>        off, albedo, normal, depth, lod_level, overdraw,\n    \
                 \x20                    chunk_bounds or draw_path (default off)\n    \
                 --path <path>        draw path a, b or direct (default: the device's)\n    \
                 --width <px>         image width (default 1920)\n    \
                 --height <px>        image height (default 1080)\n"
            }
            Self::Bench => {
                "    Measures the renderer on every sub-scene of a scene (release builds\n    \
                 only): GPU time per pass (mean, p95, p99, max), CPU frame time, triangles,\n    \
                 draws, ns per draw, and for generated worlds the generation and meshing\n    \
                 times; writes them as JSON.\n    \
                 --scene <name>       hills, structures or calib\n    \
                 --out <file.json>    where to write the report (required)\n    \
                 --frames <n>         measured frames per sub-scene (default 300)\n    \
                 --path <path>        draw path a, b or direct (default: the device's)\n    \
                 --width <px>         target width (default 1920)\n    \
                 --height <px>        target height (default 1080)\n"
            }
        }
    }
}

/// Options of `cpe caps`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CapsArgs {
    /// Where to save the Markdown report, if anywhere.
    pub out: Option<PathBuf>,
    /// Where to save the JSON report, if anywhere.
    pub json: Option<PathBuf>,
}

/// Options of `cpe proctex-sheet`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SheetArgs {
    /// Where to write the PNG.
    pub out: PathBuf,
    /// Texture size in texels: a power of two from [`MIN_SIZE`] to [`MAX_SIZE`].
    pub size: u32,
    /// Magnification, `1..=MAX_SHEET_TILE / size`.
    pub scale: u32,
    /// Pack directories in load order; empty for the recipe and preset sheet.
    pub packs: Vec<PathBuf>,
}

/// Options of `cpe content-check`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ContentCheckArgs {
    /// Pack directories in load order; empty for the workspace's base pack.
    pub packs: Vec<PathBuf>,
}

/// Options of `cpe run`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunArgs {
    /// The scene.
    pub scene: SceneName,
    /// Quit by itself after this many frames drawn once the world has loaded (smoke tests).
    pub frames: Option<u32>,
}

/// Options of `cpe shot`.
#[derive(Clone, Debug, PartialEq)]
pub struct ShotArgs {
    /// The scene.
    pub scene: SceneName,
    /// Camera of the scene.
    pub camera: usize,
    /// Time on the flight path (seconds) instead of a fixed camera.
    pub time: Option<f64>,
    /// The sun, if not the camera's default.
    pub sun: Option<Sun>,
    /// Debug view.
    pub view: DebugView,
    /// Draw path, if not the device's default.
    pub path: Option<DrawMode>,
    /// Where to write the PNG.
    pub out: PathBuf,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

/// Options of `cpe bench`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchArgs {
    /// The scene.
    pub scene: SceneName,
    /// Measured frames per sub-scene.
    pub frames: u32,
    /// Draw path, if not the device's default.
    pub path: Option<DrawMode>,
    /// Where to write the JSON report.
    pub out: PathBuf,
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
}

/// A command line that cannot be run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsageError {
    /// What is wrong, without a trailing full stop.
    pub message: String,
    /// The subcommand the error concerns, if one was recognised.
    pub command: Option<Subcommand>,
}

impl UsageError {
    fn new(command: Option<Subcommand>, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            command,
        }
    }

    /// A line that points to the right usage: the synopsis of the subcommand, or the
    /// general help.
    #[must_use]
    pub fn hint(&self) -> String {
        match self.command {
            Some(command) => format!(
                "usage: {}\nrun `{BIN} {} --help` for details",
                command.synopsis(),
                command.name()
            ),
            None => format!("run `{BIN} --help` for the usage"),
        }
    }
}

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for UsageError {}

/// The usage text: of everything for `None`, of one subcommand otherwise.
#[must_use]
pub fn usage(topic: Option<Subcommand>) -> String {
    let mut text = String::new();
    match topic {
        Some(command) => {
            text.push_str(&format!("usage: {}\n\n", command.synopsis()));
            text.push_str(command.description());
        }
        None => {
            text.push_str(&format!(
                "{BIN} {} - tools of the Cubeplanet engine\n\nUsage:\n",
                env!("CARGO_PKG_VERSION")
            ));
            for command in Subcommand::ALL {
                text.push_str(&format!("  {}\n", command.synopsis()));
                text.push_str(command.description());
                text.push('\n');
            }
            text.push_str(&format!(
                "  {BIN} --help | {BIN} help [<command>] | {BIN} <command> --help\n    \
                 Shows this text, or the text of one command. Without arguments, {BIN}\n    \
                 shows this text too.\n\n"
            ));
        }
    }
    let gpu_commands = [
        Subcommand::Caps,
        Subcommand::Run,
        Subcommand::Shot,
        Subcommand::Bench,
    ];
    if topic.is_none_or(|command| gpu_commands.contains(&command)) {
        text.push_str(
            "Environment:\n  \
             CP_BACKEND=vulkan|dx12    use only this GPU backend\n  \
             CP_DXC=<dxcompiler.dll>   compile DX12 shaders with DXC instead of FXC\n",
        );
        if topic.is_none() {
            text.push_str(
                "  CP_LOG=<level>            log level of the engine: off, error, warn,\n  \
                 \x20                         info (default), debug or trace; other crates\n  \
                 \x20                         (wgpu, naga) log at most warnings\n",
            );
        }
        text.push('\n');
    }
    if topic.is_none() {
        text.push_str("Exit codes: 0 success, 1 the command failed, 2 bad command line.\n");
    }
    text
}

/// Parses the arguments after the program name.
///
/// No arguments ask for the general help.
///
/// # Errors
/// A [`UsageError`] for an unknown command or option, a missing, repeated or invalid value,
/// and for arguments that are not valid Unicode where a name is expected.
pub fn parse<I>(args: I) -> Result<Command, UsageError>
where
    I: IntoIterator<Item = OsString>,
{
    let mut args = args.into_iter();
    let Some(first) = args.next() else {
        return Ok(Command::Help(None));
    };
    let Some(first) = first.to_str() else {
        return Err(UsageError::new(
            None,
            format!("unknown command `{}`", first.to_string_lossy()),
        ));
    };
    match first {
        "-h" | "--help" => {
            if let Some(extra) = args.next() {
                return Err(UsageError::new(
                    None,
                    format!("unexpected argument `{}`", extra.to_string_lossy()),
                ));
            }
            Ok(Command::Help(None))
        }
        "help" => parse_help(args),
        name => {
            let Some(command) = Subcommand::from_name(name) else {
                let what = if name.starts_with('-') {
                    "option"
                } else {
                    "command"
                };
                return Err(UsageError::new(None, format!("unknown {what} `{name}`")));
            };
            let mut scanner = Scanner { command, args };
            match command {
                Subcommand::Caps => parse_caps(&mut scanner),
                Subcommand::ProctexSheet => parse_sheet(&mut scanner),
                Subcommand::ContentCheck => parse_content_check(&mut scanner),
                Subcommand::Run => parse_run(&mut scanner),
                Subcommand::Shot => parse_shot(&mut scanner),
                Subcommand::Bench => parse_bench(&mut scanner),
            }
        }
    }
}

/// `help [<command>]`.
fn parse_help(mut args: impl Iterator<Item = OsString>) -> Result<Command, UsageError> {
    let topic = match args.next() {
        None => None,
        Some(name) => {
            let name = name.to_string_lossy();
            Some(Subcommand::from_name(&name).ok_or_else(|| {
                UsageError::new(
                    None,
                    format!("no help for `{name}`: there is no such command"),
                )
            })?)
        }
    };
    if let Some(extra) = args.next() {
        return Err(UsageError::new(
            None,
            format!("unexpected argument `{}`", extra.to_string_lossy()),
        ));
    }
    Ok(Command::Help(topic))
}

fn parse_caps<I: Iterator<Item = OsString>>(
    scanner: &mut Scanner<I>,
) -> Result<Command, UsageError> {
    let mut args = CapsArgs::default();
    while let Some(flag) = scanner.next_flag()? {
        match flag {
            Flag::Help => return Ok(Command::Help(Some(scanner.command))),
            Flag::Named { name, inline } => match name.as_str() {
                "--out" => scanner.set_path(&mut args.out, &name, inline)?,
                "--json" => scanner.set_path(&mut args.json, &name, inline)?,
                _ => return Err(scanner.unknown_option(&name)),
            },
        }
    }
    Ok(Command::Caps(args))
}

fn parse_sheet<I: Iterator<Item = OsString>>(
    scanner: &mut Scanner<I>,
) -> Result<Command, UsageError> {
    let mut out = None;
    let mut size = None;
    let mut scale = None;
    let mut packs = Vec::new();
    while let Some(flag) = scanner.next_flag()? {
        match flag {
            Flag::Help => return Ok(Command::Help(Some(scanner.command))),
            Flag::Named { name, inline } => match name.as_str() {
                "--out" => scanner.set_path(&mut out, &name, inline)?,
                "--size" => scanner.set_number(&mut size, &name, inline)?,
                "--scale" => scanner.set_number(&mut scale, &name, inline)?,
                "--pack" => {
                    let value = scanner.value(&name, inline)?;
                    packs.push(PathBuf::from(value));
                }
                _ => return Err(scanner.unknown_option(&name)),
            },
        }
    }
    let out = out.ok_or_else(|| scanner.error("--out <file.png> is required"))?;
    let size = size.unwrap_or(DEFAULT_SHEET_SIZE);
    if !size.is_power_of_two() || !(MIN_SIZE..=MAX_SIZE).contains(&size) {
        return Err(scanner.error(format!(
            "--size {size} is not supported: use 16, 32, 64, 128 or 256"
        )));
    }
    let max_scale = MAX_SHEET_TILE / size;
    let scale = scale.unwrap_or(DEFAULT_SHEET_SCALE.min(max_scale));
    if scale == 0 || scale > max_scale {
        return Err(scanner.error(format!(
            "--scale {scale} is not supported with --size {size}: use 1 to {max_scale} \
             (a tile is at most {MAX_SHEET_TILE} px)"
        )));
    }
    Ok(Command::ProctexSheet(SheetArgs {
        out,
        size,
        scale,
        packs,
    }))
}

fn parse_content_check<I: Iterator<Item = OsString>>(
    scanner: &mut Scanner<I>,
) -> Result<Command, UsageError> {
    let mut args = ContentCheckArgs::default();
    while let Some(flag) = scanner.next_flag()? {
        match flag {
            Flag::Help => return Ok(Command::Help(Some(scanner.command))),
            Flag::Named { name, inline } => match name.as_str() {
                "--pack" => {
                    let value = scanner.value(&name, inline)?;
                    args.packs.push(PathBuf::from(value));
                }
                _ => return Err(scanner.unknown_option(&name)),
            },
        }
    }
    Ok(Command::ContentCheck(args))
}

fn parse_run<I: Iterator<Item = OsString>>(
    scanner: &mut Scanner<I>,
) -> Result<Command, UsageError> {
    let mut scene = None;
    let mut frames = None;
    while let Some(flag) = scanner.next_flag()? {
        match flag {
            Flag::Help => return Ok(Command::Help(Some(scanner.command))),
            Flag::Named { name, inline } => match name.as_str() {
                "--scene" => scanner.set_scene(&mut scene, &name, inline)?,
                "--frames" => scanner.set_number(&mut frames, &name, inline)?,
                _ => return Err(scanner.unknown_option(&name)),
            },
        }
    }
    Ok(Command::Run(RunArgs {
        scene: scene.unwrap_or(SceneName::Hills),
        frames,
    }))
}

fn parse_shot<I: Iterator<Item = OsString>>(
    scanner: &mut Scanner<I>,
) -> Result<Command, UsageError> {
    let mut scene = None;
    let mut camera = None;
    let mut time = None;
    let mut sun = None;
    let mut view = None;
    let mut path = None;
    let mut out = None;
    let mut width = None;
    let mut height = None;
    while let Some(flag) = scanner.next_flag()? {
        match flag {
            Flag::Help => return Ok(Command::Help(Some(scanner.command))),
            Flag::Named { name, inline } => match name.as_str() {
                "--scene" => scanner.set_scene(&mut scene, &name, inline)?,
                "--camera" => scanner.set_number(&mut camera, &name, inline)?,
                "--time" => scanner.set_float(&mut time, &name, inline)?,
                "--sun" => scanner.set_sun(&mut sun, &name, inline)?,
                "--view" => scanner.set_view(&mut view, &name, inline)?,
                "--path" => scanner.set_path_mode(&mut path, &name, inline)?,
                "--out" => scanner.set_path(&mut out, &name, inline)?,
                "--width" => scanner.set_number(&mut width, &name, inline)?,
                "--height" => scanner.set_number(&mut height, &name, inline)?,
                _ => return Err(scanner.unknown_option(&name)),
            },
        }
    }
    let scene = scene.ok_or_else(|| scanner.error("--scene <name> is required"))?;
    let out = out.ok_or_else(|| scanner.error("--out <file.png> is required"))?;
    if time.is_some() && scene != SceneName::Hills {
        return Err(scanner.error("--time needs --scene hills (the flight path)"));
    }
    if time.is_some() && camera.is_some() {
        return Err(scanner.error("--time and --camera exclude each other"));
    }
    if let Some(time) = time
        && !(time.is_finite() && time >= 0.0)
    {
        return Err(scanner.error(format!("--time {time} must be a time of 0 s or more")));
    }
    let (width, height) = scanner.size(width, height)?;
    Ok(Command::Shot(ShotArgs {
        scene,
        camera: camera.unwrap_or(0) as usize,
        time,
        sun,
        view: view.unwrap_or(DebugView::Off),
        path,
        out,
        width,
        height,
    }))
}

fn parse_bench<I: Iterator<Item = OsString>>(
    scanner: &mut Scanner<I>,
) -> Result<Command, UsageError> {
    let mut scene = None;
    let mut frames = None;
    let mut path = None;
    let mut out = None;
    let mut width = None;
    let mut height = None;
    while let Some(flag) = scanner.next_flag()? {
        match flag {
            Flag::Help => return Ok(Command::Help(Some(scanner.command))),
            Flag::Named { name, inline } => match name.as_str() {
                "--scene" => scanner.set_scene(&mut scene, &name, inline)?,
                "--frames" => scanner.set_number(&mut frames, &name, inline)?,
                "--path" => scanner.set_path_mode(&mut path, &name, inline)?,
                "--out" => scanner.set_path(&mut out, &name, inline)?,
                "--width" => scanner.set_number(&mut width, &name, inline)?,
                "--height" => scanner.set_number(&mut height, &name, inline)?,
                _ => return Err(scanner.unknown_option(&name)),
            },
        }
    }
    let scene = scene.ok_or_else(|| scanner.error("--scene <name> is required"))?;
    let out = out.ok_or_else(|| scanner.error("--out <file.json> is required"))?;
    let frames = frames.unwrap_or(DEFAULT_BENCH_FRAMES);
    if frames == 0 || frames > 100_000 {
        return Err(scanner.error(format!(
            "--frames {frames} is not supported: use 1 to 100000"
        )));
    }
    let (width, height) = scanner.size(width, height)?;
    Ok(Command::Bench(BenchArgs {
        scene,
        frames,
        path,
        out,
        width,
        height,
    }))
}

/// One option on the command line of a subcommand.
enum Flag {
    /// `-h` or `--help`.
    Help,
    /// `--name`, with the value after `=` if it was written that way.
    Named {
        name: String,
        inline: Option<OsString>,
    },
}

/// Walks the arguments of one subcommand.
struct Scanner<I> {
    command: Subcommand,
    args: I,
}

impl<I: Iterator<Item = OsString>> Scanner<I> {
    fn error(&self, message: impl Into<String>) -> UsageError {
        UsageError::new(Some(self.command), message)
    }

    fn unknown_option(&self, name: &str) -> UsageError {
        self.error(format!(
            "unknown option `{name}` for `{BIN} {}`",
            self.command.name()
        ))
    }

    /// The next option; `None` after the last argument.
    fn next_flag(&mut self) -> Result<Option<Flag>, UsageError> {
        let Some(arg) = self.args.next() else {
            return Ok(None);
        };
        let Some(text) = arg.to_str() else {
            return Err(self.error(format!("unexpected argument `{}`", arg.to_string_lossy())));
        };
        if text == "-h" || text == "--help" {
            return Ok(Some(Flag::Help));
        }
        if let Some(body) = text.strip_prefix("--").filter(|body| !body.is_empty()) {
            let (name, inline) = match body.split_once('=') {
                Some((name, value)) => (name, Some(OsString::from(value))),
                None => (body, None),
            };
            return Ok(Some(Flag::Named {
                name: format!("--{name}"),
                inline,
            }));
        }
        if text.starts_with('-') && text.len() > 1 {
            return Err(self.unknown_option(text));
        }
        Err(self.error(format!("unexpected argument `{text}`")))
    }

    /// The value of the option `name`: the text after `=`, or the next argument. A next
    /// argument that looks like an option is not taken as the value; one that starts with a
    /// minus sign and a digit is (a negative number).
    fn value(&mut self, name: &str, inline: Option<OsString>) -> Result<OsString, UsageError> {
        let value = match inline {
            Some(value) => value,
            None => self
                .args
                .next()
                .ok_or_else(|| self.error(format!("{name} needs a value")))?,
        };
        if value.is_empty() {
            return Err(self.error(format!("{name} needs a non-empty value")));
        }
        let text = value.to_string_lossy();
        let negative_number = text
            .strip_prefix('-')
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit() || c == '.'));
        if text.starts_with('-') && !negative_number {
            return Err(self.error(format!("{name} needs a value, not `{text}`")));
        }
        Ok(value)
    }

    fn set_path(
        &mut self,
        slot: &mut Option<PathBuf>,
        name: &str,
        inline: Option<OsString>,
    ) -> Result<(), UsageError> {
        if slot.is_some() {
            return Err(self.error(format!("{name} is given twice")));
        }
        *slot = Some(PathBuf::from(self.value(name, inline)?));
        Ok(())
    }

    /// The value of `name` as text.
    fn text(
        &mut self,
        is_set: bool,
        name: &str,
        inline: Option<OsString>,
    ) -> Result<String, UsageError> {
        if is_set {
            return Err(self.error(format!("{name} is given twice")));
        }
        Ok(self.value(name, inline)?.to_string_lossy().into_owned())
    }

    fn set_number(
        &mut self,
        slot: &mut Option<u32>,
        name: &str,
        inline: Option<OsString>,
    ) -> Result<(), UsageError> {
        let text = self.text(slot.is_some(), name, inline)?;
        let number = text.parse::<u32>().map_err(|_| {
            self.error(format!(
                "invalid value `{text}` for {name}: expected a whole number"
            ))
        })?;
        *slot = Some(number);
        Ok(())
    }

    fn set_float(
        &mut self,
        slot: &mut Option<f64>,
        name: &str,
        inline: Option<OsString>,
    ) -> Result<(), UsageError> {
        let text = self.text(slot.is_some(), name, inline)?;
        let number = text
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                self.error(format!(
                    "invalid value `{text}` for {name}: expected a number"
                ))
            })?;
        *slot = Some(number);
        Ok(())
    }

    fn set_scene(
        &mut self,
        slot: &mut Option<SceneName>,
        name: &str,
        inline: Option<OsString>,
    ) -> Result<(), UsageError> {
        let text = self.text(slot.is_some(), name, inline)?;
        let scene = SceneName::from_id(&text).ok_or_else(|| {
            self.error(format!(
                "unknown scene `{text}`: use {}",
                SceneName::ALL.map(SceneName::id).join(", ")
            ))
        })?;
        *slot = Some(scene);
        Ok(())
    }

    fn set_view(
        &mut self,
        slot: &mut Option<DebugView>,
        name: &str,
        inline: Option<OsString>,
    ) -> Result<(), UsageError> {
        let text = self.text(slot.is_some(), name, inline)?;
        let view = DebugView::from_id(&text.replace('-', "_")).ok_or_else(|| {
            self.error(format!(
                "unknown view `{text}`: use {}",
                DebugView::ALL.map(DebugView::id).join(", ")
            ))
        })?;
        *slot = Some(view);
        Ok(())
    }

    fn set_path_mode(
        &mut self,
        slot: &mut Option<DrawMode>,
        name: &str,
        inline: Option<OsString>,
    ) -> Result<(), UsageError> {
        let text = self.text(slot.is_some(), name, inline)?;
        let mode = DrawMode::from_id(&text.to_ascii_lowercase())
            .ok_or_else(|| self.error(format!("unknown draw path `{text}`: use a, b or direct")))?;
        *slot = Some(mode);
        Ok(())
    }

    fn set_sun(
        &mut self,
        slot: &mut Option<Sun>,
        name: &str,
        inline: Option<OsString>,
    ) -> Result<(), UsageError> {
        let text = self.text(slot.is_some(), name, inline)?;
        let invalid = || {
            self.error(format!(
                "invalid value `{text}` for {name}: expected <elevation>,<azimuth> in degrees, \
                 elevation from -90 to 90"
            ))
        };
        let (elevation, azimuth) = text.split_once(',').ok_or_else(invalid)?;
        let parse = |part: &str| part.trim().parse::<f64>().ok().filter(|v| v.is_finite());
        let (Some(elevation_deg), Some(azimuth_deg)) = (parse(elevation), parse(azimuth)) else {
            return Err(invalid());
        };
        if !(-90.0..=90.0).contains(&elevation_deg) {
            return Err(invalid());
        }
        *slot = Some(Sun {
            elevation_deg,
            azimuth_deg: azimuth_deg.rem_euclid(360.0),
        });
        Ok(())
    }

    /// Width and height of a target, defaults for the ones not given.
    fn size(&self, width: Option<u32>, height: Option<u32>) -> Result<(u32, u32), UsageError> {
        let width = width.unwrap_or(DEFAULT_WIDTH);
        let height = height.unwrap_or(DEFAULT_HEIGHT);
        for (name, value) in [("--width", width), ("--height", height)] {
            if value == 0 || value > MAX_SIDE {
                return Err(self.error(format!(
                    "{name} {value} is not supported: use 1 to {MAX_SIDE}"
                )));
            }
        }
        Ok((width, height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_strs(args: &[&str]) -> Result<Command, UsageError> {
        parse(args.iter().map(OsString::from))
    }

    fn error_of(args: &[&str]) -> UsageError {
        match parse_strs(args) {
            Ok(command) => panic!("{args:?} must be rejected, got {command:?}"),
            Err(error) => error,
        }
    }

    #[test]
    fn help_in_every_spelling() {
        assert_eq!(parse_strs(&[]), Ok(Command::Help(None)));
        assert_eq!(parse_strs(&["--help"]), Ok(Command::Help(None)));
        assert_eq!(parse_strs(&["-h"]), Ok(Command::Help(None)));
        assert_eq!(parse_strs(&["help"]), Ok(Command::Help(None)));
        for command in Subcommand::ALL {
            let expected = Ok(Command::Help(Some(command)));
            assert_eq!(parse_strs(&["help", command.name()]), expected);
            assert_eq!(parse_strs(&[command.name(), "--help"]), expected);
            assert_eq!(parse_strs(&[command.name(), "-h"]), expected);
        }
        assert!(
            error_of(&["help", "frobnicate"])
                .message
                .contains("no such command")
        );
        assert!(
            error_of(&["help", "caps", "extra"])
                .message
                .contains("extra")
        );
        assert!(error_of(&["--help", "extra"]).message.contains("extra"));
    }

    #[test]
    fn the_usage_text_lists_exactly_the_commands_that_exist() {
        let text = usage(None);
        for command in Subcommand::ALL {
            assert!(text.contains(command.synopsis()), "{}", command.name());
            assert!(usage(Some(command)).contains(command.synopsis()));
        }
        // The E2 commands exist now; nothing of later milestones is announced.
        for now in ["cpe run", "cpe shot", "cpe bench"] {
            assert!(text.contains(now), "{now}");
        }
        for later in ["cpe physics-lab", "cpe bench-physics"] {
            assert!(!text.contains(later), "{later}");
        }
        assert!(text.is_ascii());
    }

    #[test]
    fn caps_takes_two_optional_paths() {
        assert_eq!(
            parse_strs(&["caps"]),
            Ok(Command::Caps(CapsArgs::default()))
        );
        assert_eq!(
            parse_strs(&["caps", "--json", "a.json", "--out=b.md"]),
            Ok(Command::Caps(CapsArgs {
                out: Some("b.md".into()),
                json: Some("a.json".into()),
            }))
        );
        assert!(
            error_of(&["caps", "--out", "a", "--out", "b"])
                .message
                .contains("twice")
        );
        assert!(
            error_of(&["caps", "--out"])
                .message
                .contains("needs a value")
        );
        assert!(error_of(&["caps", "--out="]).message.contains("non-empty"));
        assert!(
            error_of(&["caps", "--out", "--json", "x"])
                .message
                .contains("not `--json`")
        );
        assert!(
            error_of(&["caps", "--md", "x"])
                .message
                .contains("unknown option `--md`")
        );
        assert!(
            error_of(&["caps", "-x"])
                .message
                .contains("unknown option `-x`")
        );
        assert!(
            error_of(&["caps", "report.md"])
                .message
                .contains("unexpected argument")
        );
        assert_eq!(error_of(&["caps", "--md"]).command, Some(Subcommand::Caps));
    }

    #[test]
    fn proctex_sheet_needs_an_output_and_checks_its_numbers() {
        assert_eq!(
            parse_strs(&["proctex-sheet", "--out", "sheet.png"]),
            Ok(Command::ProctexSheet(SheetArgs {
                out: "sheet.png".into(),
                size: 16,
                scale: 8,
                packs: Vec::new(),
            }))
        );
        assert_eq!(
            parse_strs(&["proctex-sheet", "--size=64", "--out", "s.png"]),
            Ok(Command::ProctexSheet(SheetArgs {
                out: "s.png".into(),
                size: 64,
                scale: 4,
                packs: Vec::new(),
            }))
        );
        assert_eq!(
            parse_strs(&[
                "proctex-sheet",
                "--out",
                "s.png",
                "--size",
                "32",
                "--scale",
                "2",
                "--pack",
                "content/base",
                "--pack=mine"
            ]),
            Ok(Command::ProctexSheet(SheetArgs {
                out: "s.png".into(),
                size: 32,
                scale: 2,
                packs: vec!["content/base".into(), "mine".into()],
            }))
        );
        assert!(error_of(&["proctex-sheet"]).message.contains("--out"));
        for size in ["0", "8", "24", "512"] {
            let error = error_of(&["proctex-sheet", "--out", "s.png", "--size", size]);
            assert!(error.message.contains("not supported"), "{size}");
        }
        for scale in ["0", "17"] {
            let error = error_of(&["proctex-sheet", "--out", "s.png", "--scale", scale]);
            assert!(error.message.contains("use 1 to 16"), "{scale}");
        }
        let error = error_of(&["proctex-sheet", "--out", "s.png", "--scale", "-3"]);
        assert!(error.message.contains("expected a whole number"));
        let error = error_of(&["proctex-sheet", "--out", "s.png", "--size", "big"]);
        assert!(error.message.contains("expected a whole number"));
        let error = error_of(&["proctex-sheet", "--out", "s.png", "--size", "99999999999"]);
        assert!(error.message.contains("expected a whole number"));
        let error = error_of(&["proctex-sheet", "--out", "s.png", "--pack"]);
        assert!(error.message.contains("needs a value"));
    }

    #[test]
    fn content_check_collects_packs_in_order() {
        assert_eq!(
            parse_strs(&["content-check"]),
            Ok(Command::ContentCheck(ContentCheckArgs::default()))
        );
        assert_eq!(
            parse_strs(&["content-check", "--pack", "a", "--pack=b"]),
            Ok(Command::ContentCheck(ContentCheckArgs {
                packs: vec!["a".into(), "b".into()],
            }))
        );
        assert!(
            error_of(&["content-check", "--pack"])
                .message
                .contains("needs a value")
        );
    }

    #[test]
    fn run_takes_a_scene() {
        assert_eq!(
            parse_strs(&["run"]),
            Ok(Command::Run(RunArgs {
                scene: SceneName::Hills,
                frames: None,
            }))
        );
        assert_eq!(
            parse_strs(&["run", "--scene", "calib", "--frames", "30"]),
            Ok(Command::Run(RunArgs {
                scene: SceneName::Calib,
                frames: Some(30),
            }))
        );
        assert!(
            error_of(&["run", "--scene", "moon"])
                .message
                .contains("unknown scene `moon`")
        );
        assert!(
            error_of(&["run", "--camera", "3"])
                .message
                .contains("unknown option")
        );
    }

    #[test]
    fn shot_reads_every_option() {
        let command = parse_strs(&[
            "shot",
            "--scene",
            "hills",
            "--camera",
            "3",
            "--sun",
            "-5,270",
            "--view",
            "lod-level",
            "--path",
            "B",
            "--out",
            "x.png",
            "--width",
            "640",
            "--height=360",
        ])
        .expect("a valid shot");
        assert_eq!(
            command,
            Command::Shot(ShotArgs {
                scene: SceneName::Hills,
                camera: 3,
                time: None,
                sun: Some(Sun {
                    elevation_deg: -5.0,
                    azimuth_deg: 270.0
                }),
                view: DebugView::LodLevel,
                path: Some(DrawMode::B),
                out: "x.png".into(),
                width: 640,
                height: 360,
            })
        );
        let Ok(Command::Shot(flight)) = parse_strs(&[
            "shot", "--scene", "hills", "--time", "12.5", "--out", "f.png",
        ]) else {
            panic!("a flight shot");
        };
        assert_eq!(flight.time, Some(12.5));
        assert_eq!((flight.width, flight.height), (1920, 1080));
        assert_eq!(flight.sun, None);
        assert!(
            error_of(&["shot", "--out", "x.png"])
                .message
                .contains("--scene")
        );
        assert!(
            error_of(&["shot", "--scene", "hills"])
                .message
                .contains("--out")
        );
        for (args, needle) in [
            (&["--sun", "45"][..], "expected <elevation>,<azimuth>"),
            (&["--sun", "95,10"], "elevation from -90 to 90"),
            (&["--view", "pretty"], "unknown view"),
            (&["--path", "c"], "unknown draw path"),
            (&["--width", "0"], "not supported"),
            (&["--height", "9000"], "not supported"),
            (&["--time", "-1"], "0 s or more"),
            (&["--time", "x"], "expected a number"),
        ] {
            let mut full = vec!["shot", "--scene", "hills", "--out", "x.png"];
            full.extend_from_slice(args);
            assert!(error_of(&full).message.contains(needle), "{args:?}");
        }
        assert!(
            error_of(&["shot", "--scene", "calib", "--time", "1", "--out", "x.png"])
                .message
                .contains("--scene hills")
        );
        assert!(
            error_of(&[
                "shot", "--scene", "hills", "--time", "1", "--camera", "2", "--out", "x.png"
            ])
            .message
            .contains("exclude")
        );
    }

    #[test]
    fn bench_needs_a_scene_and_an_output() {
        assert_eq!(
            parse_strs(&[
                "bench", "--scene", "calib", "--out", "b.json", "--frames", "60"
            ]),
            Ok(Command::Bench(BenchArgs {
                scene: SceneName::Calib,
                frames: 60,
                path: None,
                out: "b.json".into(),
                width: 1920,
                height: 1080,
            }))
        );
        assert!(
            error_of(&[
                "bench", "--scene", "calib", "--out", "b.json", "--frames", "0"
            ])
            .message
            .contains("not supported")
        );
        assert!(
            error_of(&["bench", "--out", "b.json"])
                .message
                .contains("--scene")
        );
    }

    #[test]
    fn unknown_commands_and_options_are_named() {
        assert_eq!(
            error_of(&["frobnicate"]).message,
            "unknown command `frobnicate`"
        );
        assert_eq!(
            error_of(&["--verbose"]).message,
            "unknown option `--verbose`"
        );
        assert_eq!(error_of(&["walk"]).message, "unknown command `walk`");
        assert!(error_of(&["frobnicate"]).hint().contains("--help"));
        assert!(
            error_of(&["caps", "--x"])
                .hint()
                .contains(Subcommand::Caps.synopsis())
        );
    }

    #[cfg(windows)]
    #[test]
    fn arguments_that_are_not_unicode_are_rejected_or_kept_as_paths() {
        use std::os::windows::ffi::OsStringExt;

        // An unpaired surrogate: valid in a Windows path, not valid Unicode.
        let odd = OsString::from_wide(&[0x0061, 0xD800]);
        let error = parse([odd.clone()]).expect_err("not a command name");
        assert!(error.message.starts_with("unknown command"));
        let error = parse([OsString::from("caps"), odd.clone()]).expect_err("not an option");
        assert!(error.message.starts_with("unexpected argument"));
        let parsed = parse([OsString::from("caps"), OsString::from("--out"), odd.clone()]);
        assert_eq!(
            parsed,
            Ok(Command::Caps(CapsArgs {
                out: Some(PathBuf::from(odd)),
                json: None,
            }))
        );
    }
}

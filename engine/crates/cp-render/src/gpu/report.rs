//! The capability report in its two forms: Markdown for people, JSON for tools.
//!
//! Both forms are pure functions of [`GpuCaps`] and contain no time stamps or other
//! run-dependent data, so two runs on the same machine produce identical reports. The Markdown
//! form is ASCII as long as the driver's own strings are, because `cpe caps` prints it to the
//! console.

use super::caps::GpuCaps;
use super::features::wanted_features;
use super::json::Json;
use super::limits::{LimitUnit, REPORTED_LIMITS, limit_values, wanted_limits};

/// Version of the JSON layout; raised whenever a key changes its meaning or disappears.
///
/// 2: draw paths `a`/`b` (contract v2), `dx12_shader_compiler`, `derived` without
/// `can_cache_pipelines`.
const JSON_SCHEMA_VERSION: u32 = 2;

impl GpuCaps {
    /// The capability report as a Markdown document.
    ///
    /// Sections: adapter summary (adapter, backend, driver, quality tier, draw path), derived
    /// capabilities, the optional features with availability, state, purpose and fallback, the
    /// important limits (WebGPU default, requested, adapter, enabled), and the downlevel flags.
    pub fn report_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# GPU capability report\n\n");
        self.summary_table().render(&mut out);

        out.push_str("\n## Derived capabilities\n\n");
        let mut table = Table::new(["Capability", "Value", "Meaning"]);
        for (name, value, meaning) in self.derived.rows() {
            table.row([name, yes_no(value), meaning]);
        }
        table.render(&mut out);

        self.features_section(&mut out);
        self.limits_section(&mut out);
        self.downlevel_section(&mut out);
        out
    }

    /// The capability report as a JSON document (pretty-printed, UTF-8, final newline).
    ///
    /// Top-level keys: `schema`, `adapter`, `instance_flags`, `dx12_shader_compiler` (`id`,
    /// `dxc_path`), `tier`, `draw_path` (`a` or `b`), `derived`, `features` (`optional`,
    /// `available`, `enabled`), `limits` (every wgpu limit with `default`, `requested`,
    /// `adapter`, `enabled`) and `downlevel`.
    pub fn report_json(&self) -> String {
        self.to_json().to_pretty_string()
    }

    fn summary_table(&self) -> Table<2> {
        let info = &self.adapter;
        let mut table = Table::new(["Property", "Value"]);
        table.row(["Adapter", &info.name]);
        table.row(["Backend", backend_name(info.backend)]);
        table.row(["Device type", device_type_name(info.device_type)]);
        table.row([
            "Vendor / device id",
            &format!("0x{:04X} / 0x{:04X}", info.vendor, info.device),
        ]);
        table.row(["Driver", &info.driver]);
        table.row(["Driver info", &info.driver_info]);
        if !info.device_pci_bus_id.is_empty() {
            table.row(["PCI bus id", &info.device_pci_bus_id]);
        }
        table.row([
            "Quality tier",
            &format!("{} - {}", self.tier.id(), self.tier.description()),
        ]);
        table.row(["Draw path", self.derived.draw_path.description()]);
        table.row([
            "Instance flags",
            &flag_list(self.instance_flags.iter_names()),
        ]);
        let compiler = &self.dx12_shader_compiler;
        let compiler = match compiler.dxc_path() {
            Some(path) => format!("{} ({path})", compiler.id()),
            None => compiler.id().to_owned(),
        };
        table.row(["DX12 shader compiler", &compiler]);
        table.row([
            "WebGPU compliant",
            yes_no(self.downlevel.is_webgpu_compliant()),
        ]);
        table
    }

    fn features_section(&self, out: &mut String) {
        let features = self.optional_features();
        let enabled = features.iter().filter(|status| status.enabled).count();
        out.push_str("\n## Optional features\n\n");
        out.push_str(&format!(
            "{enabled} of the {} optional features the pipeline can use are enabled.\n\n",
            features.len()
        ));
        let mut table = Table::new(["Feature", "Available", "Enabled", "Used for", "Without it"]);
        for status in &features {
            table.row([
                status.name,
                yes_no(status.available),
                yes_no(status.enabled),
                status.used_for,
                status.fallback,
            ]);
        }
        table.render(out);

        let unused = self.available_features - wanted_features();
        out.push_str(&format!(
            "\nOther features of the adapter (not requested): {}\n",
            flag_list(unused.iter_names())
        ));
        let unexpected = self.enabled_features - wanted_features();
        if !unexpected.is_empty() {
            out.push_str(&format!(
                "\nEnabled outside the pipeline's list: {}\n",
                flag_list(unexpected.iter_names())
            ));
        }
    }

    fn limits_section(&self, out: &mut String) {
        let defaults = wgpu::Limits::defaults();
        let requested = wanted_limits(self.enabled_features);
        out.push_str("\n## Limits\n\n");
        out.push_str(
            "Requested is what the pipeline asks for; enabled is that value clamped to the \
             adapter.\n\n",
        );
        let mut table = Table::new([
            "Limit",
            "WebGPU default",
            "Requested",
            "Adapter",
            "Enabled",
            "Used for",
        ]);
        for limit in REPORTED_LIMITS {
            let show = |limits: &wgpu::Limits| format_limit((limit.get)(limits), limit.unit);
            table.row([
                limit.name,
                &show(&defaults),
                &show(&requested),
                &show(&self.adapter_limits),
                &show(&self.enabled_limits),
                limit.used_for,
            ]);
        }
        table.render(out);
    }

    fn downlevel_section(&self, out: &mut String) {
        let flags = self.downlevel.flags;
        let missing = wgpu::DownlevelFlags::all() - flags;
        out.push_str("\n## Downlevel flags\n\n");
        out.push_str(&format!(
            "Shader model: {:?}. Missing flags: {}\n\n",
            self.downlevel.shader_model,
            flag_list(missing.iter_names())
        ));
        let mut table = Table::new(["Flag", "Present"]);
        for (name, flag) in wgpu::DownlevelFlags::all().iter_names() {
            table.row([name, yes_no(flags.contains(flag))]);
        }
        table.render(out);
    }

    fn to_json(&self) -> Json {
        let info = &self.adapter;
        let adapter = Json::object([
            ("name", Json::from(info.name.as_str())),
            ("backend", Json::from(info.backend.to_str())),
            ("device_type", Json::from(device_type_id(info.device_type))),
            ("vendor", Json::from(info.vendor)),
            ("device", Json::from(info.device)),
            ("pci_bus_id", Json::from(info.device_pci_bus_id.as_str())),
            ("driver", Json::from(info.driver.as_str())),
            ("driver_info", Json::from(info.driver_info.as_str())),
            ("subgroup_min_size", Json::from(info.subgroup_min_size)),
            ("subgroup_max_size", Json::from(info.subgroup_max_size)),
            (
                "transient_saves_memory",
                info.transient_saves_memory.map_or(Json::Null, Json::from),
            ),
        ]);

        let derived = Json::object(
            self.derived
                .rows()
                .into_iter()
                .map(|(name, value, _)| (name, Json::from(value))),
        );

        let optional = self.optional_features().into_iter().map(|status| {
            Json::object([
                ("name", Json::from(status.name)),
                ("available", Json::from(status.available)),
                ("enabled", Json::from(status.enabled)),
                ("used_for", Json::from(status.used_for)),
                ("fallback", Json::from(status.fallback)),
            ])
        });
        let features = Json::object([
            ("optional", Json::Array(optional.collect())),
            (
                "available",
                flag_names(self.available_features.iter_names()),
            ),
            ("enabled", flag_names(self.enabled_features.iter_names())),
        ]);

        let defaults = limit_values(&wgpu::Limits::defaults());
        let requested = limit_values(&wanted_limits(self.enabled_features));
        let adapter_limits = limit_values(&self.adapter_limits);
        let enabled_limits = limit_values(&self.enabled_limits);
        let limits = Json::object(
            defaults
                .iter()
                .enumerate()
                .map(|(index, &(name, default))| {
                    let entry = Json::object([
                        ("default", Json::from(default)),
                        ("requested", Json::from(requested[index].1)),
                        ("adapter", Json::from(adapter_limits[index].1)),
                        ("enabled", Json::from(enabled_limits[index].1)),
                    ]);
                    (name, entry)
                }),
        );

        let downlevel_flags = wgpu::DownlevelFlags::all()
            .iter_names()
            .map(|(name, flag)| (name, Json::from(self.downlevel.flags.contains(flag))));
        let downlevel = Json::object([
            (
                "webgpu_compliant",
                Json::from(self.downlevel.is_webgpu_compliant()),
            ),
            (
                "shader_model",
                Json::from(format!("{:?}", self.downlevel.shader_model)),
            ),
            ("flags", Json::object(downlevel_flags)),
        ]);

        Json::object([
            ("schema", Json::from(JSON_SCHEMA_VERSION)),
            ("adapter", adapter),
            (
                "instance_flags",
                flag_names(self.instance_flags.iter_names()),
            ),
            (
                "dx12_shader_compiler",
                Json::object([
                    ("id", Json::from(self.dx12_shader_compiler.id())),
                    (
                        "dxc_path",
                        self.dx12_shader_compiler
                            .dxc_path()
                            .map_or(Json::Null, Json::from),
                    ),
                ]),
            ),
            ("tier", Json::from(self.tier.id())),
            ("draw_path", Json::from(self.derived.draw_path.id())),
            ("derived", derived),
            ("features", features),
            ("limits", limits),
            ("downlevel", downlevel),
        ])
    }
}

/// A Markdown table with `N` columns whose cells are padded to equal width, so that it is
/// also readable as plain text.
struct Table<const N: usize> {
    header: [&'static str; N],
    rows: Vec<[String; N]>,
}

impl<const N: usize> Table<N> {
    fn new(header: [&'static str; N]) -> Self {
        Self {
            header,
            rows: Vec::new(),
        }
    }

    fn row(&mut self, cells: [&str; N]) {
        self.rows.push(cells.map(escape_cell));
    }

    fn render(&self, out: &mut String) {
        let mut widths = self.header.map(|title| title.chars().count());
        for row in &self.rows {
            for (width, cell) in widths.iter_mut().zip(row) {
                *width = (*width).max(cell.chars().count());
            }
        }

        let write_row = |out: &mut String, cells: [&str; N]| {
            out.push('|');
            for (cell, width) in cells.into_iter().zip(widths) {
                let padding = width - cell.chars().count();
                out.push(' ');
                out.push_str(cell);
                out.push_str(&" ".repeat(padding));
                out.push_str(" |");
            }
            out.push('\n');
        };

        write_row(out, self.header);
        out.push('|');
        for width in widths {
            out.push_str(&"-".repeat(width + 2));
            out.push('|');
        }
        out.push('\n');
        for row in &self.rows {
            write_row(out, row.each_ref().map(String::as_str));
        }
    }
}

/// Makes arbitrary text safe for one table cell: no line breaks, no unescaped column
/// separator. Empty text becomes `-` so that the cell stays visible.
fn escape_cell(text: &str) -> String {
    let mut cell = String::with_capacity(text.len());
    for character in text.trim().chars() {
        match character {
            '|' => cell.push_str("\\|"),
            '\\' => cell.push_str("\\\\"),
            control if control.is_control() => cell.push(' '),
            other => cell.push(other),
        }
    }
    if cell.is_empty() {
        "-".to_owned()
    } else {
        cell
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

/// Flag names joined with commas, or `none` for the empty set.
fn flag_list<'a, F>(names: impl Iterator<Item = (&'a str, F)>) -> String {
    let names: Vec<&str> = names.map(|(name, _)| name).collect();
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

fn flag_names<'a, F>(names: impl Iterator<Item = (&'a str, F)>) -> Json {
    Json::strings(names.map(|(name, _)| name))
}

fn backend_name(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Noop => "noop",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "DX12",
        wgpu::Backend::Gl => "OpenGL",
        wgpu::Backend::BrowserWebGpu => "WebGPU",
    }
}

fn device_type_name(device_type: wgpu::DeviceType) -> &'static str {
    match device_type {
        wgpu::DeviceType::Other => "other or unknown",
        wgpu::DeviceType::IntegratedGpu => "integrated GPU",
        wgpu::DeviceType::DiscreteGpu => "discrete GPU",
        wgpu::DeviceType::VirtualGpu => "virtual GPU",
        wgpu::DeviceType::Cpu => "CPU (software rasteriser)",
    }
}

fn device_type_id(device_type: wgpu::DeviceType) -> &'static str {
    match device_type {
        wgpu::DeviceType::Other => "other",
        wgpu::DeviceType::IntegratedGpu => "integrated_gpu",
        wgpu::DeviceType::DiscreteGpu => "discrete_gpu",
        wgpu::DeviceType::VirtualGpu => "virtual_gpu",
        wgpu::DeviceType::Cpu => "cpu",
    }
}

fn format_limit(value: u64, unit: LimitUnit) -> String {
    match unit {
        LimitUnit::Count => value.to_string(),
        LimitUnit::Bytes => format_bytes(value),
    }
}

/// A byte count with a binary unit: exact where the value is a whole number of units
/// (`1 GiB`), otherwise the exact byte count with an approximation (`4294967295 B (~4.0 GiB)`).
fn format_bytes(bytes: u64) -> String {
    const UNITS: [(u64, &str); 3] = [(1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")];
    for (size, unit) in UNITS {
        if bytes >= size {
            return if bytes.is_multiple_of(size) {
                format!("{} {unit}", bytes / size)
            } else {
                // Display only; the exact value stands next to it.
                format!("{bytes} B (~{:.1} {unit})", bytes as f64 / size as f64)
            };
        }
    }
    format!("{bytes} B")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::features::WANTED_FEATURES;
    use crate::gpu::json::read::parse;
    use crate::gpu::testing::{desktop_profile, indirect_validation};
    use wgpu::{Backend, DeviceType};

    fn desktop_caps() -> GpuCaps {
        GpuCaps::negotiate(
            desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu),
            indirect_validation(),
        )
    }

    /// A weak adapter: no optional features, WebGL2-class limits, missing downlevel flags and
    /// strings that try to break both output formats.
    fn hostile_caps() -> GpuCaps {
        let mut profile = desktop_profile(Backend::Dx12, DeviceType::Cpu);
        profile.info.name = "Soft \"GPU\" | v1\\2\nline two\ttab é漢😀".to_owned();
        profile.info.driver = String::new();
        profile.info.driver_info = "a|b||c".to_owned();
        profile.info.device_pci_bus_id = "0000:01:00.0".to_owned();
        profile.info.transient_saves_memory = Some(false);
        profile.features = wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        profile.limits = wgpu::Limits::downlevel_webgl2_defaults();
        profile.downlevel.flags = wgpu::DownlevelFlags::ANISOTROPIC_FILTERING;
        profile.downlevel.shader_model = wgpu::ShaderModel::Sm4;
        GpuCaps::negotiate(profile, wgpu::InstanceFlags::empty())
    }

    /// Splits a table line at the unescaped `|` separators and trims the cells.
    fn cells(line: &str) -> Vec<String> {
        let mut cells = Vec::new();
        let mut current = String::new();
        let mut characters = line.chars();
        while let Some(character) = characters.next() {
            match character {
                '\\' => {
                    current.push(character);
                    current.extend(characters.next());
                }
                '|' => cells.push(std::mem::take(&mut current).trim().to_owned()),
                other => current.push(other),
            }
        }
        assert!(
            current.trim().is_empty(),
            "text after the last column in {line:?}"
        );
        assert_eq!(
            cells.remove(0),
            "",
            "table line must start with '|': {line:?}"
        );
        cells
    }

    /// All tables of a Markdown document as rows of cells (header first, separator removed).
    /// Panics when a table is malformed, which makes it a structural check as well.
    fn parse_tables(markdown: &str) -> Vec<Vec<Vec<String>>> {
        let mut tables: Vec<Vec<Vec<String>>> = Vec::new();
        let mut in_table = false;
        for line in markdown.lines() {
            if !line.starts_with('|') {
                in_table = false;
                continue;
            }
            if !in_table {
                tables.push(Vec::new());
                in_table = true;
            }
            tables.last_mut().unwrap().push(cells(line));
        }
        for table in &mut tables {
            assert!(table.len() >= 2, "a table needs a header and a separator");
            let columns = table[0].len();
            for row in table.iter() {
                assert_eq!(row.len(), columns, "ragged table row {row:?}");
            }
            let separator = table.remove(1);
            assert!(
                separator
                    .iter()
                    .all(|cell| cell.len() >= 3 && cell.chars().all(|c| c == '-')),
                "second table line must be the separator: {separator:?}"
            );
        }
        tables
    }

    fn table_with_header<'a>(tables: &'a [Vec<Vec<String>>], first: &str) -> &'a [Vec<String>] {
        tables
            .iter()
            .find(|table| table[0][0] == first)
            .unwrap_or_else(|| panic!("no table with first column {first:?}"))
    }

    fn row<'a>(table: &'a [Vec<String>], key: &str) -> &'a [String] {
        table
            .iter()
            .find(|row| row[0] == key)
            .unwrap_or_else(|| panic!("no row {key:?}"))
    }

    #[test]
    fn markdown_has_all_sections_in_order() {
        let markdown = desktop_caps().report_markdown();
        let headings: Vec<&str> = markdown
            .lines()
            .filter(|line| line.starts_with('#'))
            .collect();
        assert_eq!(
            headings,
            [
                "# GPU capability report",
                "## Derived capabilities",
                "## Optional features",
                "## Limits",
                "## Downlevel flags",
            ]
        );
        assert!(markdown.ends_with("|\n"));
        assert!(!markdown.contains("\n\n\n"), "no double blank lines");
        assert_eq!(parse_tables(&markdown).len(), 5);
    }

    #[test]
    fn markdown_summary_names_adapter_backend_driver_and_tier() {
        let markdown = desktop_caps().report_markdown();
        let tables = parse_tables(&markdown);
        let summary = table_with_header(&tables, "Property");
        assert_eq!(row(summary, "Adapter")[1], "Test GPU 9000");
        assert_eq!(row(summary, "Backend")[1], "Vulkan");
        assert_eq!(row(summary, "Device type")[1], "discrete GPU");
        assert_eq!(row(summary, "Vendor / device id")[1], "0x10DE / 0x1B80");
        assert_eq!(row(summary, "Driver")[1], "TestDriver");
        assert_eq!(row(summary, "Driver info")[1], "1.2.3");
        assert!(row(summary, "Quality tier")[1].starts_with("high - "));
        assert_eq!(
            row(summary, "Draw path")[1],
            "A: GPU culling, multi_draw_indexed_indirect_count"
        );
        assert_eq!(
            row(summary, "Instance flags")[1],
            "VALIDATION_INDIRECT_CALL"
        );
        assert_eq!(row(summary, "DX12 shader compiler")[1], "fxc");
        let dxc =
            desktop_caps().with_dx12_shader_compiler(crate::gpu::Dx12ShaderCompiler::DynamicDxc {
                path: "C:/dxc/dxcompiler.dll".to_owned(),
            });
        let markdown = dxc.report_markdown();
        let tables = parse_tables(&markdown);
        let summary = table_with_header(&tables, "Property");
        assert_eq!(
            row(summary, "DX12 shader compiler")[1],
            "dynamic_dxc (C:/dxc/dxcompiler.dll)"
        );
        assert_eq!(row(summary, "WebGPU compliant")[1], "yes");
        // No bus id known for the synthetic adapter: the row is left out.
        assert!(summary.iter().all(|row| row[0] != "PCI bus id"));
    }

    #[test]
    fn markdown_lists_every_optional_feature_with_its_state() {
        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        let missing = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::TIMESTAMP_QUERY;
        profile.features.remove(missing);
        let caps = GpuCaps::negotiate(profile, indirect_validation());
        let markdown = caps.report_markdown();
        let tables = parse_tables(&markdown);
        let features = table_with_header(&tables, "Feature");
        assert_eq!(
            features[0],
            ["Feature", "Available", "Enabled", "Used for", "Without it"]
        );
        assert_eq!(features.len(), 1 + WANTED_FEATURES.len());
        for status in caps.optional_features() {
            let cells = row(features, status.name);
            assert_eq!(cells[1], yes_no(status.available), "{}", status.name);
            assert_eq!(cells[2], yes_no(status.enabled), "{}", status.name);
            assert_eq!(cells[3], status.used_for);
            assert_eq!(cells[4], status.fallback);
        }
        assert_eq!(
            row(features, "TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES")[1..3],
            ["no", "no"]
        );
        assert_eq!(
            row(features, "TIMESTAMP_QUERY_INSIDE_PASSES")[1..3],
            ["yes", "no"]
        );
        assert_eq!(
            row(features, "MULTI_DRAW_INDIRECT_COUNT")[1..3],
            ["yes", "yes"]
        );
        assert_eq!(
            row(features, "TEXTURE_FORMAT_16BIT_NORM")[1..3],
            ["yes", "yes"]
        );
        assert!(features.iter().all(|row| row[0] != "PIPELINE_CACHE"));
        // Missing: the format capabilities, the timestamp base feature and with it both
        // sub-features.
        assert!(
            markdown.contains("9 of the 13 optional features the pipeline can use are enabled.")
        );
        let other = markdown
            .lines()
            .find_map(|line| line.strip_prefix("Other features of the adapter (not requested): "))
            .expect("the line with the unrequested features");
        let mut other: Vec<&str> = other.split(", ").collect();
        other.sort_unstable();
        assert_eq!(other, ["CLEAR_TEXTURE", "POLYGON_MODE_LINE", "SHADER_F16"]);
        assert!(!markdown.contains("Enabled outside the pipeline's list"));
    }

    #[test]
    fn markdown_limits_show_default_requested_adapter_and_enabled() {
        let markdown = desktop_caps().report_markdown();
        let tables = parse_tables(&markdown);
        let limits = table_with_header(&tables, "Limit");
        assert_eq!(limits.len(), 1 + REPORTED_LIMITS.len());
        assert_eq!(
            row(limits, "max_buffer_size")[1..5],
            ["256 MiB", "1 GiB", "4294967295 B (~4.0 GiB)", "1 GiB"]
        );
        assert_eq!(
            row(limits, "max_storage_buffer_binding_size")[1..5],
            ["128 MiB", "1 GiB", "2147483644 B (~2.0 GiB)", "1 GiB"]
        );
        // Requested: the adapter maximum, i.e. as many as there are.
        assert_eq!(
            row(limits, "max_texture_array_layers")[1..5],
            ["256", "4294967295", "2048", "2048"]
        );
        assert_eq!(
            row(limits, "max_sampled_textures_per_shader_stage")[1..5],
            ["16", "32", "524288", "32"]
        );
        assert_eq!(
            row(limits, "max_storage_textures_per_shader_stage")[1..5],
            ["4", "16", "524288", "16"]
        );
        assert_eq!(
            row(limits, "max_color_attachment_bytes_per_sample")[1..5],
            ["32 B", "64 B", "128 B", "64 B"]
        );
        // Not raised (contract §4.5 does not list it).
        assert_eq!(
            row(limits, "max_storage_buffers_per_shader_stage")[1..5],
            ["8", "8", "524288", "8"]
        );
        assert_eq!(row(limits, "max_bind_groups")[1..5], ["4", "8", "8", "8"]);
        assert_eq!(
            row(limits, "max_immediate_size")[1..5],
            ["0 B", "128 B", "256 B", "128 B"]
        );
        assert_eq!(
            row(limits, "min_uniform_buffer_offset_alignment")[1..5],
            ["256 B", "256 B", "64 B", "256 B"]
        );
    }

    #[test]
    fn markdown_downlevel_section_lists_all_flags_and_names_the_missing() {
        let all = wgpu::DownlevelFlags::all().iter_names().count();
        let markdown = desktop_caps().report_markdown();
        assert!(markdown.contains("Shader model: Sm5. Missing flags: none\n"));
        let tables = parse_tables(&markdown);
        let flags = table_with_header(&tables, "Flag");
        assert_eq!(flags.len(), 1 + all);
        assert!(flags[1..].iter().all(|row| row[1] == "yes"));

        let markdown = hostile_caps().report_markdown();
        let tables = parse_tables(&markdown);
        let flags = table_with_header(&tables, "Flag");
        assert_eq!(row(flags, "ANISOTROPIC_FILTERING")[1], "yes");
        assert_eq!(row(flags, "COMPUTE_SHADERS")[1], "no");
        assert_eq!(
            flags[1..].iter().filter(|row| row[1] == "no").count(),
            all - 1
        );
        assert!(markdown.contains("Shader model: Sm4. Missing flags: COMPUTE_SHADERS, "));
    }

    #[test]
    fn markdown_shows_a_degraded_device_without_breaking_the_tables() {
        let caps = hostile_caps();
        let markdown = caps.report_markdown();
        // parse_tables() panics on ragged rows: the hostile strings added no columns or lines.
        let tables = parse_tables(&markdown);
        let summary = table_with_header(&tables, "Property");
        assert_eq!(
            row(summary, "Adapter")[1],
            "Soft \"GPU\" \\| v1\\\\2 line two tab é漢😀"
        );
        assert_eq!(row(summary, "Driver")[1], "-");
        assert_eq!(row(summary, "Driver info")[1], "a\\|b\\|\\|c");
        assert_eq!(row(summary, "PCI bus id")[1], "0000:01:00.0");
        assert_eq!(row(summary, "Backend")[1], "DX12");
        assert!(row(summary, "Quality tier")[1].starts_with("low - "));
        assert!(row(summary, "Draw path")[1].starts_with("B: "));
        assert_eq!(row(summary, "Instance flags")[1], "none");
        assert_eq!(row(summary, "WebGPU compliant")[1], "no");

        let derived = table_with_header(&tables, "Capability");
        assert_eq!(derived.len(), 1 + caps.derived.rows().len());
        assert!(derived[1..].iter().all(|row| row[1] == "no"));

        let features = table_with_header(&tables, "Feature");
        assert_eq!(
            row(features, "TIMESTAMP_QUERY_INSIDE_PASSES")[1..3],
            ["yes", "no"]
        );
        assert!(markdown.contains("0 of the 13 optional features"));
        assert!(markdown.contains("Other features of the adapter (not requested): none\n"));

        let limits = table_with_header(&tables, "Limit");
        // Not raised (contract §4.5 does not list it), but clamped to the adapter.
        assert_eq!(
            row(limits, "max_storage_buffers_per_shader_stage")[1..5],
            ["8", "8", "0", "0"]
        );
        assert_eq!(
            row(limits, "max_storage_textures_per_shader_stage")[1..5],
            ["4", "16", "0", "0"]
        );
    }

    #[test]
    fn markdown_tables_are_aligned_as_plain_text() {
        for caps in [desktop_caps(), hostile_caps()] {
            let markdown = caps.report_markdown();
            let mut width = None;
            for line in markdown.lines() {
                if line.starts_with('|') {
                    let length = line.chars().count();
                    assert_eq!(*width.get_or_insert(length), length, "misaligned: {line:?}");
                } else {
                    width = None;
                }
            }
        }
    }

    #[test]
    fn markdown_mentions_features_enabled_outside_the_list() {
        let profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        let limits = profile.plan_device().limits;
        let caps = GpuCaps::new(
            profile,
            indirect_validation(),
            wgpu::Features::CLEAR_TEXTURE,
            limits,
        );
        assert!(
            caps.report_markdown()
                .contains("Enabled outside the pipeline's list: CLEAR_TEXTURE\n")
        );
    }

    #[test]
    fn reports_are_deterministic() {
        let caps = desktop_caps();
        assert_eq!(caps.report_markdown(), caps.clone().report_markdown());
        assert_eq!(caps.report_json(), caps.clone().report_json());
    }

    #[test]
    fn json_is_valid_and_has_the_documented_structure() {
        let caps = desktop_caps();
        let text = caps.report_json();
        assert!(text.ends_with("}\n"));
        let json = parse(&text).unwrap_or_else(|error| panic!("invalid JSON: {error}\n{text}"));

        let keys: Vec<&str> = json
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, _)| key.as_str())
            .collect();
        assert_eq!(
            keys,
            [
                "schema",
                "adapter",
                "instance_flags",
                "dx12_shader_compiler",
                "tier",
                "draw_path",
                "derived",
                "features",
                "limits",
                "downlevel"
            ]
        );
        assert_eq!(json.at(&["schema"]).as_u64(), Some(2));
        assert_eq!(json.at(&["tier"]).as_str(), Some("high"));
        assert_eq!(json.at(&["draw_path"]).as_str(), Some("a"));
        assert_eq!(
            json.at(&["dx12_shader_compiler", "id"]).as_str(),
            Some("fxc")
        );
        assert_eq!(json.at(&["dx12_shader_compiler", "dxc_path"]), &Json::Null);
        let dxc =
            desktop_caps().with_dx12_shader_compiler(crate::gpu::Dx12ShaderCompiler::DynamicDxc {
                path: "C:/dxc/dxcompiler.dll".to_owned(),
            });
        let dxc = parse(&dxc.report_json()).unwrap();
        assert_eq!(
            dxc.at(&["dx12_shader_compiler", "id"]).as_str(),
            Some("dynamic_dxc")
        );
        assert_eq!(
            dxc.at(&["dx12_shader_compiler", "dxc_path"]).as_str(),
            Some("C:/dxc/dxcompiler.dll")
        );
        assert_eq!(
            json.at(&["instance_flags"]),
            &Json::strings(["VALIDATION_INDIRECT_CALL"])
        );

        let adapter = json.at(&["adapter"]);
        assert_eq!(adapter.at(&["name"]).as_str(), Some("Test GPU 9000"));
        assert_eq!(adapter.at(&["backend"]).as_str(), Some("vulkan"));
        assert_eq!(adapter.at(&["device_type"]).as_str(), Some("discrete_gpu"));
        assert_eq!(adapter.at(&["vendor"]).as_u64(), Some(0x10DE));
        assert_eq!(adapter.at(&["device"]).as_u64(), Some(0x1B80));
        assert_eq!(adapter.at(&["driver"]).as_str(), Some("TestDriver"));
        assert_eq!(adapter.at(&["driver_info"]).as_str(), Some("1.2.3"));
        assert_eq!(adapter.at(&["pci_bus_id"]).as_str(), Some(""));
        assert_eq!(adapter.at(&["transient_saves_memory"]), &Json::Null);
    }

    #[test]
    fn json_derived_section_matches_the_record() {
        for caps in [desktop_caps(), hostile_caps()] {
            let json = parse(&caps.report_json()).unwrap();
            let derived = json.at(&["derived"]).as_object().unwrap();
            assert_eq!(derived.len(), caps.derived.rows().len());
            for (name, value, _) in caps.derived.rows() {
                assert_eq!(json.at(&["derived", name]).as_bool(), Some(value), "{name}");
            }
            assert_eq!(json.at(&["tier"]).as_str(), Some(caps.tier.id()));
            assert_eq!(
                json.at(&["draw_path"]).as_str(),
                Some(caps.derived.draw_path.id())
            );
        }
    }

    #[test]
    fn json_features_section_matches_the_record() {
        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        let missing = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::TIMESTAMP_QUERY;
        profile.features.remove(missing);
        let caps = GpuCaps::negotiate(profile, indirect_validation());
        let json = parse(&caps.report_json()).unwrap();

        let optional = json.at(&["features", "optional"]).as_array().unwrap();
        let expected = caps.optional_features();
        assert_eq!(optional.len(), expected.len());
        for (entry, status) in optional.iter().zip(&expected) {
            assert_eq!(entry.at(&["name"]).as_str(), Some(status.name));
            assert_eq!(entry.at(&["available"]).as_bool(), Some(status.available));
            assert_eq!(entry.at(&["enabled"]).as_bool(), Some(status.enabled));
            assert_eq!(entry.at(&["used_for"]).as_str(), Some(status.used_for));
            assert_eq!(entry.at(&["fallback"]).as_str(), Some(status.fallback));
        }

        let names = |key: &str| -> Vec<String> {
            let list = json.at(&["features", key]).as_array().unwrap();
            list.iter()
                .map(|name| name.as_str().unwrap().to_owned())
                .collect()
        };
        let set = |names: &[String]| {
            names.iter().fold(wgpu::Features::empty(), |set, name| {
                set | wgpu::Features::from_name(name).unwrap_or_else(|| panic!("unknown {name}"))
            })
        };
        assert_eq!(set(&names("available")), caps.available_features);
        assert_eq!(set(&names("enabled")), caps.enabled_features);
        assert!(names("available").contains(&"SHADER_F16".to_owned()));
        assert!(!names("enabled").contains(&"SHADER_F16".to_owned()));
    }

    #[test]
    fn json_limits_section_covers_every_limit() {
        let caps = desktop_caps();
        let json = parse(&caps.report_json()).unwrap();
        let limits = json.at(&["limits"]).as_object().unwrap();
        let adapter = limit_values(&caps.adapter_limits);
        let enabled = limit_values(&caps.enabled_limits);
        assert_eq!(limits.len(), adapter.len());
        for (index, (name, entry)) in limits.iter().enumerate() {
            assert_eq!(name, adapter[index].0);
            assert_eq!(entry.as_object().unwrap().len(), 4);
            assert_eq!(
                entry.at(&["adapter"]).as_u64(),
                Some(adapter[index].1),
                "{name}"
            );
            assert_eq!(
                entry.at(&["enabled"]).as_u64(),
                Some(enabled[index].1),
                "{name}"
            );
        }
        let buffer = json.at(&["limits", "max_buffer_size"]);
        assert_eq!(buffer.at(&["default"]).as_u64(), Some(256 << 20));
        assert_eq!(buffer.at(&["requested"]).as_u64(), Some(1 << 30));
        assert_eq!(buffer.at(&["adapter"]).as_u64(), Some(u64::from(u32::MAX)));
        assert_eq!(buffer.at(&["enabled"]).as_u64(), Some(1 << 30));
    }

    #[test]
    fn json_downlevel_section_matches_the_record() {
        for caps in [desktop_caps(), hostile_caps()] {
            let json = parse(&caps.report_json()).unwrap();
            let flags = json.at(&["downlevel", "flags"]).as_object().unwrap();
            assert_eq!(
                flags.len(),
                wgpu::DownlevelFlags::all().iter_names().count()
            );
            for (name, flag) in wgpu::DownlevelFlags::all().iter_names() {
                assert_eq!(
                    json.at(&["downlevel", "flags", name]).as_bool(),
                    Some(caps.downlevel.flags.contains(flag)),
                    "{name}"
                );
            }
            assert_eq!(
                json.at(&["downlevel", "webgpu_compliant"]).as_bool(),
                Some(caps.downlevel.is_webgpu_compliant())
            );
            assert_eq!(
                json.at(&["downlevel", "shader_model"]).as_str(),
                Some(format!("{:?}", caps.downlevel.shader_model).as_str())
            );
        }
    }

    #[test]
    fn json_survives_hostile_adapter_strings() {
        let caps = hostile_caps();
        let text = caps.report_json();
        let json = parse(&text).unwrap_or_else(|error| panic!("invalid JSON: {error}\n{text}"));
        // The strings come back unchanged: escaped in the text, intact after parsing.
        assert_eq!(
            json.at(&["adapter", "name"]).as_str(),
            Some(caps.adapter.name.as_str())
        );
        assert_eq!(json.at(&["adapter", "driver"]).as_str(), Some(""));
        assert_eq!(
            json.at(&["adapter", "driver_info"]).as_str(),
            Some("a|b||c")
        );
        assert_eq!(json.at(&["adapter", "backend"]).as_str(), Some("dx12"));
        assert_eq!(json.at(&["adapter", "device_type"]).as_str(), Some("cpu"));
        assert_eq!(
            json.at(&["adapter", "transient_saves_memory"]).as_bool(),
            Some(false)
        );
        assert_eq!(json.at(&["instance_flags"]), &Json::Array(Vec::new()));
        assert_eq!(json.at(&["tier"]).as_str(), Some("low"));
        assert_eq!(json.at(&["draw_path"]).as_str(), Some("b"));
    }

    #[test]
    fn byte_sizes_are_exact_or_marked_as_approximate() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(128), "128 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1 KiB");
        assert_eq!(format_bytes(65536), "64 KiB");
        assert_eq!(format_bytes(1536), "1536 B (~1.5 KiB)");
        assert_eq!(format_bytes(256 << 20), "256 MiB");
        assert_eq!(format_bytes(1 << 30), "1 GiB");
        assert_eq!(format_bytes((1 << 31) - 4), "2147483644 B (~2.0 GiB)");
        assert_eq!(format_bytes(1 << 52), "4194304 GiB");
        assert_eq!(format_limit(2048, LimitUnit::Count), "2048");
        assert_eq!(format_limit(2048, LimitUnit::Bytes), "2 KiB");
    }

    #[test]
    fn cells_are_escaped_and_never_empty() {
        assert_eq!(escape_cell("plain"), "plain");
        assert_eq!(escape_cell("  padded  "), "padded");
        assert_eq!(escape_cell(""), "-");
        assert_eq!(escape_cell(" \n "), "-");
        assert_eq!(escape_cell("a|b"), "a\\|b");
        assert_eq!(escape_cell("a\\b"), "a\\\\b");
        assert_eq!(escape_cell("a\r\nb\tc"), "a  b c");
    }
}

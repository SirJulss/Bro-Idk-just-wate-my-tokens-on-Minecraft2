//! The binary greedy mesher (contract §5.2).
//!
//! # How it works
//! 1. **Masks.** One pass over the padded cells builds bit masks with one `u64` row per padded
//!    `(y, z)` and one bit per padded `x` (bit `x + 1`): `opaque`, `render` (chunk cells of a
//!    renderable class), `grouped` (cells with a cull group) and one plane per bit of the cull
//!    group number. What a state means to the mesher (flags, class, group, the six face keys)
//!    is worked out once per state and call into a dense table indexed by state id, so every
//!    other lookup costs the same whatever the ids are. Rows whose 32 chunk cells hold one
//!    state are remembered.
//! 2. **Visibility.** Per direction, the neighbour masks of a row are the same row shifted by
//!    one bit (±X) or the adjacent row (±Y, ±Z), so "hidden" is a handful of word operations,
//!    `opaque(B) | (grouped(A) & grouped(B) & all group planes equal)`, done for all rows in
//!    branch-free loops.
//! 3. **Planes.** Every regular face, and every hidden face on the boundary (a wall face), is put
//!    into the 32 × 32 bit matrix of its `(slice, key)`; the key is the merge key plus layer and
//!    wall flag. A row of one state needs one key lookup for all its faces.
//! 4. **Greedy cover** of each slice's matrices in `(v, u)` order, which is the in-range sort
//!    order, so nothing is sorted.
//! 5. **Assembly** of the per-range buffers into the output in storage order.

use core::fmt;

use cp_core::block::{BlockFlags, BlockRender, RenderClass};
use cp_core::face::Face;
use cp_core::ids::{BlockStateId, MaterialId};
use cp_core::quad::PackedQuad;

use crate::mesh::{
    CellBounds, ChunkMesh, Layer, LayerRanges, RANGES_PER_LAYER, WALL_RANGE_OFFSET, storage_slot,
};
use crate::{PADDED_SIZE, PADDED_VOLUME};

/// The per-state tables the mesher reads, indexed by [`BlockStateId::index`] (the registry's
/// `BlockTables`, contract §4.4). All three have the same length, at least 1 (state 0 is air).
///
/// Contract §5.2 also lists the `lod` table; the face rule does not read it, so it is not an
/// input (a recorded deviation, see the [crate documentation](crate)). The tables may differ
/// from one call to the next: a [`Mesher`] keeps nothing of them between calls.
#[derive(Clone, Copy, Debug)]
pub struct MeshTables<'a> {
    /// Block flags; the mesher reads [`BlockFlags::OPAQUE`].
    pub flags: &'a [BlockFlags],
    /// Render class and the appearance of each world face.
    pub render: &'a [BlockRender],
    /// Cull group number per state, 0 = none.
    pub cull_group: &'a [u16],
}

/// One chunk to mesh: see the [crate documentation](crate) for the meaning of the fields.
#[derive(Clone, Copy, Debug)]
pub struct MeshInput<'a> {
    /// The [`PADDED_VOLUME`] states of the padded chunk, at [`padded_index`](crate::padded_index).
    pub cells: &'a [BlockStateId],
    /// The chunk's authored cells (bit = cell index `x + 32·z + 1024·y`), `None` = none.
    pub authored: Option<&'a [u64; 512]>,
    /// LOD level of the chunk (cells of `2^level` m); decides the layer of cutout states.
    pub level: u8,
    /// Per-state tables.
    pub tables: MeshTables<'a>,
}

/// Why a chunk could not be meshed. The input is rejected as a whole; the output mesh is left
/// empty.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeshError {
    /// `cells` does not hold exactly [`PADDED_VOLUME`] states.
    CellCount {
        /// The length that was passed.
        len: usize,
    },
    /// The tables are empty (air, state 0, is missing) or differ in length.
    TableLengths {
        /// Length of [`MeshTables::flags`].
        flags: usize,
        /// Length of [`MeshTables::render`].
        render: usize,
        /// Length of [`MeshTables::cull_group`].
        cull_group: usize,
    },
    /// Air (state 0) is `OPAQUE` or has a cull group: it would hide faces, which the wall rule
    /// ("faces that would exist if the halo were air") does not allow.
    AirHidesFaces,
    /// A cell that the mesher reads holds a state the tables do not describe
    /// ([`BlockStateId::UNCHANGED`] included).
    UnknownState {
        /// The offending state.
        state: BlockStateId,
        /// Its padded cell coordinates, each in `-1..=32`.
        cell: [i8; 3],
    },
    /// A state shows a face whose appearance has a `uv_rot` outside `0..4`.
    InvalidUvRot {
        /// The state.
        state: BlockStateId,
        /// The world face.
        face: Face,
        /// The offending value.
        uv_rot: u8,
    },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::CellCount { len } => write!(
                f,
                "padded chunk has {len} cells, expected {PADDED_VOLUME} (34^3)"
            ),
            Self::TableLengths {
                flags,
                render,
                cull_group,
            } => write!(
                f,
                "mesh tables must be non-empty and equally long \
                 (flags {flags}, render {render}, cull_group {cull_group})"
            ),
            Self::AirHidesFaces => write!(
                f,
                "air (state 0) must be neither OPAQUE nor in a cull group"
            ),
            Self::UnknownState { state, cell } => write!(
                f,
                "padded cell {cell:?} holds state {state}, which the mesh tables do not describe"
            ),
            Self::InvalidUvRot {
                state,
                face,
                uv_rot,
            } => write!(
                f,
                "state {state} has uv_rot {uv_rot} on face {face:?} (must be 0..4)"
            ),
        }
    }
}

impl std::error::Error for MeshError {}

/// Edge of the padded chunk.
const P: usize = PADDED_SIZE;
/// Rows of the masks: one per padded `(y, z)`, index `py * 34 + pz`.
const ROWS: usize = P * P;
/// The rows `py = 1..=32` (all `pz`): every row of chunk cells lies in between, and the
/// neighbour rows of all of them (`row ± 1`, `row ± 34`) lie inside the masks.
const CELL_ROWS_START: usize = P;
const CELL_ROWS_END: usize = ROWS - P;
/// Bits of the chunk's own cells in a row word (padded x = 1..=32).
const INTERIOR: u64 = 0x1_FFFF_FFFE;
/// Number of cull-group bit planes (cull groups are `u16`).
const GROUP_PLANES: usize = 16;

/// One bit-mask row per padded `(y, z)`.
type RowMask = [u64; ROWS];

/// Bits of a face key: the merge key `(material, uv_rot, authored)` in the bit positions of
/// `PackedQuad::w1` (material 0–15, uv_rot 16–17, authored 18, contract §4.1), so that
/// `w1 = key & KEY_W1_BITS`; then the layer and the wall flag, which separate ranges.
const KEY_UV_ROT_SHIFT: u32 = 16;
const KEY_AUTHORED_SHIFT: u32 = 18;
const KEY_W1_BITS: u32 = (1 << 19) - 1;
const KEY_LAYER_SHIFT: u32 = 19;
const KEY_WALL: u32 = 1 << 21;
/// Marks the face key of a face whose `uv_rot` is invalid; never reaches a plane.
const KEY_INVALID: u32 = 1 << 22;

/// Fields of `PackedQuad::w0` (contract §4.1): 5 bits per cell coordinate (x, y, z), then
/// `w - 1`, `h - 1` and the face.
const QUAD_COORD_BITS: u32 = 5;
const QUAD_W_SHIFT: u32 = 15;
const QUAD_H_SHIFT: u32 = 20;
const QUAD_FACE_SHIFT: u32 = 25;
/// No key that reaches a plane has bits above 21, so this never equals one.
const NO_KEY: u32 = u32::MAX;
/// `Masks::row_state` of a row whose chunk cells hold more than one state. No row holds
/// `BlockStateId::UNCHANGED`, which the tables never describe.
const MIXED: u32 = u32::MAX;
/// Marks an unused `(key, slice)` slot.
const NO_PLANE: u32 = u32::MAX;

/// A zeroed mask on the heap.
fn zeroed_rows() -> Box<RowMask> {
    vec![0; ROWS]
        .into_boxed_slice()
        .try_into()
        .expect("the vector has ROWS elements")
}

/// A reusable binary greedy mesher (contract §5.2).
///
/// Holds about 60 KiB of fixed scratch buffers plus buffers that grow with the faces of the
/// largest chunk meshed so far (about 60 KiB for dense terrain, 9 KiB per cull-group bit in
/// use) and a per-state table of 64 bytes per state of the largest tables seen (2.8 KiB for
/// the 45 states of the base pack, 6.1 MiB for 100 000 states); they are kept for the next
/// chunk, so meshing allocates only when a buffer has to grow. `Send`, so a worker thread can
/// own one.
#[derive(Debug)]
pub struct Mesher {
    states: StateTable,
    masks: Masks,
    visibility: Visibility,
    planes: PlaneStore,
    ranges: RangeBuffers,
    slice_planes: Vec<u32>,
}

impl Default for Mesher {
    fn default() -> Self {
        Self::new()
    }
}

impl Mesher {
    /// A mesher with empty scratch buffers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            states: StateTable::default(),
            masks: Masks::new(),
            visibility: Visibility::new(),
            planes: PlaneStore::default(),
            ranges: RangeBuffers::new(),
            slice_planes: Vec::new(),
        }
    }

    /// Meshes one chunk into a new [`ChunkMesh`]; see [`Mesher::mesh_into`].
    ///
    /// # Errors
    /// Like [`Mesher::mesh_into`].
    pub fn mesh(&mut self, input: &MeshInput<'_>) -> Result<ChunkMesh, MeshError> {
        let mut mesh = ChunkMesh::new();
        self.mesh_into(input, &mut mesh)?;
        Ok(mesh)
    }

    /// Meshes one chunk into `out`, replacing its contents and reusing its allocation.
    ///
    /// The result depends only on `input` (see the [crate documentation](crate) for the rules
    /// and the output layout).
    ///
    /// # Errors
    /// [`MeshError`] when `cells` has the wrong length, the tables are inconsistent, a cell
    /// the mesher reads holds an unknown state, or a shown face has an invalid `uv_rot`.
    /// `out` is then empty.
    pub fn mesh_into(
        &mut self,
        input: &MeshInput<'_>,
        out: &mut ChunkMesh,
    ) -> Result<(), MeshError> {
        cp_core::profile_scope!("cp_mesh::mesh");
        out.clear();
        validate(input)?;
        self.states.begin(input.tables.flags.len());
        {
            cp_core::profile_scope!("cp_mesh::masks");
            self.masks.build(input, &mut self.states)?;
        }
        self.ranges.clear();
        if self.masks.any_render {
            cp_core::profile_scope!("cp_mesh::faces");
            // One instance per direction, so that everything that depends on the face is a
            // constant in the hot loops.
            self.mesh_face::<0>(input)?;
            self.mesh_face::<1>(input)?;
            self.mesh_face::<2>(input)?;
            self.mesh_face::<3>(input)?;
            self.mesh_face::<4>(input)?;
            self.mesh_face::<5>(input)?;
        }
        self.ranges.write(out);
        Ok(())
    }

    /// Collects and greedily merges the faces of the direction `Face::ALL[FACE]`.
    fn mesh_face<const FACE: usize>(&mut self, input: &MeshInput<'_>) -> Result<(), MeshError> {
        let face = Face::ALL[FACE];
        let Self {
            states,
            masks,
            visibility,
            planes,
            ranges,
            slice_planes,
        } = self;
        let scatter_scope = cp_core::profile::scope("cp_mesh::scatter");
        masks.visibility(face, visibility);
        planes.reset();
        let mut scatter = Scatter {
            input,
            row_state: &masks.row_state,
            states,
            face,
        };
        for py in 1..P - 1 {
            for pz in 1..P - 1 {
                let row = py * P + pz;
                let faces = (visibility.visible[row] >> 1) as u32;
                if faces != 0 {
                    scatter.add(planes, row, faces, 0)?;
                }
            }
        }
        // Walls: hidden faces on the chunk boundary that `face` points to.
        let boundary = |row: usize| ((visibility.hidden[row] & INTERIOR) >> 1) as u32;
        match face {
            Face::PosX | Face::NegX => {
                let bit = if face == Face::PosX { 1 << 31 } else { 1 };
                for py in 1..P - 1 {
                    for pz in 1..P - 1 {
                        let row = py * P + pz;
                        let walls = boundary(row) & bit;
                        if walls != 0 {
                            scatter.add(planes, row, walls, KEY_WALL)?;
                        }
                    }
                }
            }
            Face::PosY | Face::NegY => {
                let py = if face == Face::PosY { P - 2 } else { 1 };
                for pz in 1..P - 1 {
                    let walls = boundary(py * P + pz);
                    if walls != 0 {
                        scatter.add(planes, py * P + pz, walls, KEY_WALL)?;
                    }
                }
            }
            Face::PosZ | Face::NegZ => {
                let pz = if face == Face::PosZ { P - 2 } else { 1 };
                for py in 1..P - 1 {
                    let walls = boundary(py * P + pz);
                    if walls != 0 {
                        scatter.add(planes, py * P + pz, walls, KEY_WALL)?;
                    }
                }
            }
        }
        drop(scatter_scope);

        cp_core::profile_scope!("cp_mesh::greedy");
        let PlaneStore {
            rows,
            occupied,
            plane_keys,
            lookup,
            ..
        } = planes;
        for slice in 0..32 {
            slice_planes.clear();
            slice_planes.extend(
                lookup
                    .iter()
                    .skip(slice)
                    .step_by(32)
                    .copied()
                    .filter(|&plane| plane != NO_PLANE),
            );
            for &plane in slice_planes.iter() {
                let plane = plane as usize;
                ranges.include(
                    face,
                    plane_keys[plane],
                    slice,
                    &rows[plane],
                    occupied[plane],
                );
            }
            match slice_planes.as_slice() {
                [] => {}
                &[plane] => {
                    let plane = plane as usize;
                    let key = plane_keys[plane];
                    greedy_single(&mut rows[plane], occupied[plane], |rect| {
                        ranges.push(face, key, slice, rect);
                    });
                }
                many => greedy_multi(rows, occupied, many, |plane, rect| {
                    ranges.push(face, plane_keys[plane], slice, rect);
                }),
            }
        }
        Ok(())
    }
}

/// Checks lengths and the air row before any cell is read.
fn validate(input: &MeshInput<'_>) -> Result<(), MeshError> {
    if input.cells.len() != PADDED_VOLUME {
        return Err(MeshError::CellCount {
            len: input.cells.len(),
        });
    }
    let tables = &input.tables;
    let count = tables.flags.len();
    if count == 0 || tables.render.len() != count || tables.cull_group.len() != count {
        return Err(MeshError::TableLengths {
            flags: count,
            render: tables.render.len(),
            cull_group: tables.cull_group.len(),
        });
    }
    let air = BlockStateId::AIR.index();
    if tables.flags[air].contains(BlockFlags::OPAQUE) || tables.cull_group[air] != 0 {
        return Err(MeshError::AirHidesFaces);
    }
    Ok(())
}

/// Padded coordinates of a padded index, for error messages.
fn padded_cell(index: usize) -> [i8; 3] {
    let coordinate = |value: usize| (value % P) as i8 - 1;
    [
        coordinate(index),
        coordinate(index / (P * P)),
        coordinate(index / P),
    ]
}

/// The keys of the six faces of the renderable `state`, by [`Face::index`]: material and
/// `uv_rot` in the bits of the merge key (without the authored bit) and the layer. A face
/// whose `uv_rot` is not in `0..4` gets `KEY_INVALID`, which is an error only where such a
/// face is shown.
fn face_keys(input: &MeshInput<'_>, state: BlockStateId) -> [u32; 6] {
    let render = &input.tables.render[state.index()];
    let layer = Layer::of_class(render.class, input.level)
        .expect("only states of renderable classes have face keys");
    Face::ALL.map(|face| {
        let appearance = render.face(face);
        if appearance.uv_rot >= 4 {
            return KEY_INVALID;
        }
        u32::from(appearance.material.0)
            | (u32::from(appearance.uv_rot) << KEY_UV_ROT_SHIFT)
            | ((layer as u32) << KEY_LAYER_SHIFT)
    })
}

/// The error for a shown face of `state` towards `face` whose key is `KEY_INVALID`.
#[cold]
fn invalid_uv_rot(input: &MeshInput<'_>, state: BlockStateId, face: Face) -> MeshError {
    MeshError::InvalidUvRot {
        state,
        face,
        uv_rot: input.tables.render[state.index()].face(face).uv_rot,
    }
}

/// What the mask pass needs to know about one state, as masks of all ones or all zeros: adding
/// a cell to a row is then a few ANDs and ORs and no branch (neighbouring cells alternate
/// between grouped and ungrouped states, such as leaves and air, at random).
#[derive(Clone, Copy, Debug)]
struct CellInfo {
    /// All ones if the state is `OPAQUE`.
    opaque: u64,
    /// All ones if the state is renderable (`Cube`, `Cutout`, `Fluid`).
    render: u64,
    /// All ones if the state has a cull group.
    grouped: u64,
    /// The cull group, 0 = none.
    group: u32,
    /// The cull group, or all ones for none: the term that [`RowBits::all_groups`] ANDs.
    group_and: u32,
}

impl CellInfo {
    /// Hides nothing, shows nothing, has no group.
    const NOTHING: Self = Self {
        opaque: 0,
        render: 0,
        grouped: 0,
        group: 0,
        group_and: u32::MAX,
    };

    /// The mesher's view of `state`; `None` if the tables do not describe it.
    fn of(tables: &MeshTables<'_>, state: BlockStateId) -> Option<Self> {
        let index = state.index();
        let flags = *tables.flags.get(index)?;
        let class = tables.render.get(index)?.class;
        let group = u32::from(*tables.cull_group.get(index)?);
        let all = |condition: bool| u64::from(condition).wrapping_neg();
        let renderable = matches!(
            class,
            RenderClass::Cube | RenderClass::Cutout | RenderClass::Fluid
        );
        Some(Self {
            opaque: all(flags.contains(BlockFlags::OPAQUE)),
            render: all(renderable),
            grouped: all(group != 0),
            group,
            group_and: if group == 0 { u32::MAX } else { group },
        })
    }
}

/// What the hot loops need to know about one state.
#[derive(Clone, Copy, Debug)]
struct StateInfo {
    /// For the mask pass.
    cell: CellInfo,
    /// The call that filled the entry ([`StateTable::generation`]); any other value means
    /// "not filled in this call".
    stamp: u32,
    /// [`face_keys`] of a renderable state; zero for the others, which show no faces.
    keys: [u32; 6],
}

impl StateInfo {
    /// Not filled in any call: generations start at 1.
    const UNFILLED: Self = Self {
        cell: CellInfo::NOTHING,
        stamp: 0,
        keys: [0; 6],
    };
}

/// A dense table of [`StateInfo`], indexed by [`BlockStateId::index`] and filled on first use
/// in each call.
///
/// An entry counts only in the call that filled it (its `stamp` equals `generation`), so
/// nothing carries over between calls: the tables may differ from one call to the next
/// (another registry, a hot reload), and no pointer or length can tell. A call pays one fill
/// per distinct state it meets; every other lookup is one bounds-checked load and one
/// compare, whatever the ids are. (Small caches keyed by `id % size` missed on every switch
/// between two states with equal low id bits, such as the base pack's stone, 43, and gravel,
/// 11.) The table only grows: entries beyond the current tables are never stamped in the
/// current call, and their states fail [`CellInfo::of`].
#[derive(Debug, Default)]
struct StateTable {
    generation: u32,
    entries: Vec<StateInfo>,
}

impl StateTable {
    /// Starts a call with tables of `count` states.
    fn begin(&mut self, count: usize) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped after 2^32 calls: stamps that old would look current again.
            self.entries.fill(StateInfo::UNFILLED);
            self.generation = 1;
        }
        if self.entries.len() < count {
            self.entries.resize(count, StateInfo::UNFILLED);
        }
    }

    /// The [`CellInfo`] of `state`, read at the padded index `padded`; fills the state's entry
    /// on first use in this call. A reference into the table, so that the hot loops read the
    /// fields in place instead of copying the struct.
    ///
    /// # Errors
    /// [`MeshError::UnknownState`] if the tables do not describe `state`.
    #[inline]
    fn info(
        &mut self,
        input: &MeshInput<'_>,
        state: BlockStateId,
        padded: usize,
    ) -> Result<&CellInfo, MeshError> {
        let index = state.index();
        let filled = self
            .entries
            .get(index)
            .is_some_and(|entry| entry.stamp == self.generation);
        if !filled {
            self.fill(input, state, padded)?;
        }
        Ok(&self.entries[index].cell)
    }

    #[cold]
    #[inline(never)]
    fn fill(
        &mut self,
        input: &MeshInput<'_>,
        state: BlockStateId,
        padded: usize,
    ) -> Result<(), MeshError> {
        let cell = CellInfo::of(&input.tables, state).ok_or(MeshError::UnknownState {
            state,
            cell: padded_cell(padded),
        })?;
        let keys = if cell.render != 0 {
            face_keys(input, state)
        } else {
            [0; 6]
        };
        // `CellInfo::of` succeeded, so the index is below the table length.
        self.entries[state.index()] = StateInfo {
            cell,
            stamp: self.generation,
            keys,
        };
        Ok(())
    }

    /// The key of the face `face` (a [`Face::index`]) of `state`, which the mask pass of this
    /// call has met: it reads every chunk cell.
    #[inline]
    fn key(&self, state: BlockStateId, face: usize) -> u32 {
        let entry = &self.entries[state.index()];
        debug_assert_eq!(
            entry.stamp, self.generation,
            "state {state} was not met by the mask pass"
        );
        entry.keys[face]
    }
}

/// The bits of one mask row under construction.
struct RowBits {
    opaque: u64,
    render: u64,
    grouped: u64,
    /// OR of the cull groups of the row's cells: which `planes` may be non-zero.
    any_group: u32,
    /// AND of the non-zero cull groups of the row's cells, all ones while there is none: equal
    /// to `any_group` iff every grouped cell has the same group.
    all_groups: u32,
    /// The cull-group bit planes of the row; filled by [`RowBits::spread_group`] or
    /// [`RowBits::add_group`] once the row is complete.
    planes: [u64; GROUP_PLANES],
}

impl RowBits {
    const EMPTY: Self = Self {
        opaque: 0,
        render: 0,
        grouped: 0,
        any_group: 0,
        all_groups: u32::MAX,
        planes: [0; GROUP_PLANES],
    };

    /// Adds the chunk cells `bits`, all of a state with the view `info`.
    #[inline]
    fn add(&mut self, info: &CellInfo, bits: u64) {
        self.render |= bits & info.render;
        self.add_halo(info, bits);
    }

    /// Adds the halo cells `bits`, all of a state with the view `info`: they hide faces but
    /// show none.
    #[inline]
    fn add_halo(&mut self, info: &CellInfo, bits: u64) {
        self.opaque |= bits & info.opaque;
        self.grouped |= bits & info.grouped;
        self.any_group |= info.group;
        self.all_groups &= info.group_and;
    }

    /// True if every grouped cell of the row has the same cull group (false for a row without
    /// grouped cells).
    #[inline]
    fn one_group(&self) -> bool {
        self.any_group == self.all_groups
    }

    /// Fills the planes of a row whose grouped cells all have the group `any_group`.
    #[inline]
    fn spread_group(&mut self) {
        let mut remaining = self.any_group;
        while remaining != 0 {
            self.planes[remaining.trailing_zeros() as usize] = self.grouped;
            remaining &= remaining - 1;
        }
    }

    /// Adds the cells `bits`, of cull group `group`, to the planes.
    #[inline]
    fn add_group(&mut self, group: u32, bits: u64) {
        let mut remaining = group;
        while remaining != 0 {
            self.planes[remaining.trailing_zeros() as usize] |= bits;
            remaining &= remaining - 1;
        }
    }
}

/// Bit masks of the padded chunk, one `u64` row per padded `(y, z)`, bit `x + 1` per cell.
///
/// Rows of chunk cells (`py, pz ∈ 1..=32`) hold all 34 cells (the x halo in bits 0 and 33);
/// halo rows hold the 32 face-halo cells only; the four rows of edge cells are never written
/// or read. `render` is set for chunk cells only.
#[derive(Debug)]
struct Masks {
    opaque: Box<RowMask>,
    render: Box<RowMask>,
    grouped: Box<RowMask>,
    /// `planes[i]`: bit `i` of each cell's cull group, allocated on first use. Only planes in
    /// `active` belong to the current chunk; they are cleared when first written.
    planes: [Option<Box<RowMask>>; GROUP_PLANES],
    active: u16,
    /// Per row of chunk cells: the state of all 32 chunk cells if they are equal, else
    /// `MIXED`.
    row_state: Box<[u32; ROWS]>,
    /// Whether any chunk cell is renderable.
    any_render: bool,
}

impl Masks {
    fn new() -> Self {
        Self {
            opaque: zeroed_rows(),
            render: zeroed_rows(),
            grouped: zeroed_rows(),
            planes: Default::default(),
            active: 0,
            row_state: vec![MIXED; ROWS]
                .into_boxed_slice()
                .try_into()
                .expect("the vector has ROWS elements"),
            any_render: false,
        }
    }

    /// Fills the masks from the cells; reads chunk cells and face-halo cells only, and meets
    /// the state of every chunk cell in `states`.
    fn build(&mut self, input: &MeshInput<'_>, states: &mut StateTable) -> Result<(), MeshError> {
        let cells = input.cells;
        let mut bits = RowBits::EMPTY;
        let mut any_render = 0;
        self.active = 0;
        for py in 0..P {
            let y_halo = py == 0 || py == P - 1;
            for pz in 0..P {
                let z_halo = pz == 0 || pz == P - 1;
                if y_halo && z_halo {
                    continue;
                }
                let row = py * P + pz;
                let base = row * P;
                let interior = &cells[base + 1..base + P - 1];
                let first = interior[0];
                // Branch-free comparison of the whole row, so it vectorises.
                let uniform = interior
                    .iter()
                    .fold(0, |diff, state| diff | (state.0 ^ first.0))
                    == 0;
                if uniform {
                    let info = states.info(input, first, base + 1)?;
                    bits.add(info, INTERIOR);
                } else {
                    for (offset, &state) in interior.iter().enumerate() {
                        let info = states.info(input, state, base + 1 + offset)?;
                        bits.add(info, 2 << offset);
                    }
                }
                self.row_state[row] = if uniform { first.0 } else { MIXED };
                if y_halo || z_halo {
                    // A row of halo cells: they hide faces, only chunk cells show them.
                    bits.render = 0;
                } else {
                    for px in [0, P - 1] {
                        let info = states.info(input, cells[base + px], base + px)?;
                        bits.add_halo(info, 1 << px);
                    }
                }
                if bits.grouped != 0 {
                    if bits.one_group() {
                        bits.spread_group();
                    } else {
                        // Cells of several cull groups in one row (water next to leaves): the
                        // planes cell by cell. Bit `px` of the row is the cell `base + px`.
                        let mut grouped = bits.grouped;
                        while grouped != 0 {
                            let px = grouped.trailing_zeros() as usize;
                            grouped &= grouped - 1;
                            let info = states.info(input, cells[base + px], base + px)?;
                            bits.add_group(info.group, 1 << px);
                        }
                    }
                }
                any_render |= bits.render;
                self.store(row, &mut bits);
            }
        }
        self.any_render = any_render != 0;
        Ok(())
    }

    /// Writes a finished row and resets `bits`.
    #[inline]
    fn store(&mut self, row: usize, bits: &mut RowBits) {
        self.opaque[row] = bits.opaque;
        self.render[row] = bits.render;
        self.grouped[row] = bits.grouped;
        let mut groups = bits.any_group;
        while groups != 0 {
            let index = groups.trailing_zeros() as usize;
            groups &= groups - 1;
            let plane = self.planes[index].get_or_insert_with(zeroed_rows);
            if self.active & (1 << index) == 0 {
                // First use in this chunk: every row written so far has this bit clear.
                self.active |= 1 << index;
                plane.fill(0);
            }
            plane[row] = bits.planes[index];
            bits.planes[index] = 0;
        }
        bits.opaque = 0;
        bits.render = 0;
        bits.grouped = 0;
        bits.any_group = 0;
        bits.all_groups = u32::MAX;
    }

    /// Fills `out` for the faces towards `face` of every row of chunk cells.
    fn visibility(&self, face: Face, out: &mut Visibility) {
        match face {
            Face::PosX => self.visibility_by(out, |mask, row| mask[row] >> 1),
            Face::NegX => self.visibility_by(out, |mask, row| mask[row] << 1),
            Face::PosY => self.visibility_by(out, |mask, row| mask[row + P]),
            Face::NegY => self.visibility_by(out, |mask, row| mask[row - P]),
            Face::PosZ => self.visibility_by(out, |mask, row| mask[row + 1]),
            Face::NegZ => self.visibility_by(out, |mask, row| mask[row - 1]),
        }
    }

    /// [`Masks::visibility`] for one direction: `neighbour(mask, row)` gives the mask bits of
    /// the neighbours of the cells of `row`, aligned with the cells' own bits. Branch-free
    /// loops over fixed-size rows, which the compiler vectorises.
    #[inline]
    fn visibility_by(&self, out: &mut Visibility, neighbour: impl Fn(&RowMask, usize) -> u64) {
        let rows = CELL_ROWS_START..CELL_ROWS_END;
        if self.active == 0 {
            for row in rows {
                let render = self.render[row];
                let hidden = neighbour(&self.opaque, row);
                out.visible[row] = render & !hidden;
                out.hidden[row] = render & hidden;
            }
            return;
        }
        for row in rows.clone() {
            out.same[row] = self.grouped[row] & neighbour(&self.grouped, row);
        }
        let mut active = self.active;
        while active != 0 {
            let plane = self.planes[active.trailing_zeros() as usize]
                .as_deref()
                .expect("active planes are allocated");
            active &= active - 1;
            for row in rows.clone() {
                out.same[row] &= !(plane[row] ^ neighbour(plane, row));
            }
        }
        for row in rows {
            let render = self.render[row];
            let hidden = neighbour(&self.opaque, row) | out.same[row];
            out.visible[row] = render & !hidden;
            out.hidden[row] = render & hidden;
        }
    }
}

/// Face visibility of one direction, per row of chunk cells (the rows of halo cells between
/// them come out empty).
#[derive(Debug)]
struct Visibility {
    /// Renderable chunk cells whose face shows: regular faces.
    visible: Box<RowMask>,
    /// Renderable chunk cells whose face the neighbour hides; on the boundary these are the
    /// wall faces.
    hidden: Box<RowMask>,
    /// Scratch: cells whose neighbour has the same non-zero cull group.
    same: Box<RowMask>,
}

impl Visibility {
    fn new() -> Self {
        Self {
            visible: zeroed_rows(),
            hidden: zeroed_rows(),
            same: zeroed_rows(),
        }
    }
}

/// The authored bits of the chunk cells `x = 0..32` of row `(y, z)`.
#[inline]
fn authored_row(mask: Option<&[u64; 512]>, y: usize, z: usize) -> u32 {
    // Cell index x + 32z + 1024y: word 16y + z/2, bit x + 32 (z & 1).
    mask.map_or(0, |mask| (mask[y * 16 + (z >> 1)] >> ((z & 1) * 32)) as u32)
}

/// Puts the faces of one direction into the plane store, row by row.
struct Scatter<'i, 'a> {
    input: &'i MeshInput<'a>,
    row_state: &'i [u32; ROWS],
    states: &'i StateTable,
    face: Face,
}

impl Scatter<'_, '_> {
    /// The key of the face of `state` towards `self.face`.
    ///
    /// # Errors
    /// [`MeshError::InvalidUvRot`] if that face has an invalid `uv_rot`.
    #[inline]
    fn key(&self, state: BlockStateId) -> Result<u32, MeshError> {
        let key = self.states.key(state, self.face.index());
        if key & KEY_INVALID != 0 {
            return Err(invalid_uv_rot(self.input, state, self.face));
        }
        Ok(key)
    }

    /// Adds the faces `faces` (bit = x) of the chunk cells of row `row`; `wall` is `KEY_WALL`
    /// for wall faces, else 0.
    #[inline]
    fn add(
        &mut self,
        planes: &mut PlaneStore,
        row: usize,
        faces: u32,
        wall: u32,
    ) -> Result<(), MeshError> {
        let (y, z) = (row / P - 1, row % P - 1);
        let axis = self.face.axis();
        let authored = authored_row(self.input.authored, y, z) & faces;
        let uniform = self.row_state[row];
        if uniform != MIXED && (authored == 0 || authored == faces) {
            // One state and one authored value for all faces of the row: one key.
            let mut key = self.key(BlockStateId(uniform))? | wall;
            if authored != 0 {
                key |= 1 << KEY_AUTHORED_SHIFT;
            }
            planes.add_row(axis, key, faces, y, z);
            return Ok(());
        }
        let states = &self.input.cells[row * P + 1..row * P + P - 1];
        let mut remaining = faces;
        while remaining != 0 {
            let x = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            let key = self.key(states[x])? | wall | (((authored >> x) & 1) << KEY_AUTHORED_SHIFT);
            planes.add_cell(axis, key, x, y, z);
        }
        Ok(())
    }
}

/// The face matrices of one direction: one 32 × 32 bit matrix (rows = `v`, bits = `u`) per
/// `(key, slice)` that has faces.
#[derive(Debug)]
struct PlaneStore {
    /// Distinct keys of this direction in order of first appearance.
    keys: Vec<u32>,
    /// `lookup[key_index * 32 + slice]`: index into `rows`, or `NO_PLANE`.
    lookup: Vec<u32>,
    rows: Vec<[u32; 32]>,
    /// Per matrix: bit `v` is set if row `v` received faces.
    occupied: Vec<u32>,
    /// Key of each matrix in `rows`.
    plane_keys: Vec<u32>,
    /// `(key, index into keys)` of recently used keys, direct-mapped by a hash of the key;
    /// `NO_KEY` marks an empty entry.
    recent: [(u32, u32); KEY_CACHE],
}

/// Entries of [`PlaneStore::recent`]; a power of two.
const KEY_CACHE: usize = 32;

impl Default for PlaneStore {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            lookup: Vec::new(),
            rows: Vec::new(),
            occupied: Vec::new(),
            plane_keys: Vec::new(),
            recent: [(NO_KEY, 0); KEY_CACHE],
        }
    }
}

impl PlaneStore {
    fn reset(&mut self) {
        self.keys.clear();
        self.lookup.clear();
        self.rows.clear();
        self.occupied.clear();
        self.plane_keys.clear();
        self.recent = [(NO_KEY, 0); KEY_CACHE];
    }

    /// Index of `key` in `keys`, added on first use.
    #[inline]
    fn key_index(&mut self, key: u32) -> usize {
        let entry = (key.wrapping_mul(0x9E37_79B1) >> (32 - KEY_CACHE.trailing_zeros())) as usize;
        if self.recent[entry].0 == key {
            return self.recent[entry].1 as usize;
        }
        let index = match self.keys.iter().position(|&known| known == key) {
            Some(index) => index,
            None => {
                self.keys.push(key);
                self.lookup.extend_from_slice(&[NO_PLANE; 32]);
                self.keys.len() - 1
            }
        };
        self.recent[entry] = (key, index as u32);
        index
    }

    /// The matrix of `(key, slice)`, created empty on first use.
    #[inline]
    fn plane(&mut self, key: u32, slice: usize) -> usize {
        let slot = self.key_index(key) * 32 + slice;
        match self.lookup[slot] {
            NO_PLANE => {
                let plane = self.rows.len();
                self.rows.push([0; 32]);
                self.occupied.push(0);
                self.plane_keys.push(key);
                self.lookup[slot] = plane as u32;
                plane
            }
            plane => plane as usize,
        }
    }

    /// Adds the face of the chunk cell `(x, y, z)` with key `key` to the matrices of a
    /// direction along `axis`.
    #[inline]
    fn add_cell(&mut self, axis: usize, key: u32, x: usize, y: usize, z: usize) {
        let (slice, u, v) = match axis {
            0 => (x, y, z),
            1 => (y, z, x),
            _ => (z, x, y),
        };
        let plane = self.plane(key, slice);
        self.rows[plane][v] |= 1 << u;
        self.occupied[plane] |= 1 << v;
    }

    /// Adds the faces `bits` (bit = x) of the chunk-cell row `(y, z)`, all with key `key`, to
    /// the matrices of a direction along `axis`.
    #[inline]
    fn add_row(&mut self, axis: usize, key: u32, mut bits: u32, y: usize, z: usize) {
        match axis {
            // Slice x, matrix row v = z, bit u = y.
            0 => {
                while bits != 0 {
                    let x = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let plane = self.plane(key, x);
                    self.rows[plane][z] |= 1 << y;
                    self.occupied[plane] |= 1 << z;
                }
            }
            // Slice y, matrix rows v = x, bit u = z.
            1 => {
                let plane = self.plane(key, y);
                self.occupied[plane] |= bits;
                let rows = &mut self.rows[plane];
                while bits != 0 {
                    rows[bits.trailing_zeros() as usize] |= 1 << z;
                    bits &= bits - 1;
                }
            }
            // Slice z, matrix row v = y, bits u = x: the row word itself.
            _ => {
                let plane = self.plane(key, z);
                self.rows[plane][y] |= bits;
                self.occupied[plane] |= 1 << y;
            }
        }
    }
}

/// A rectangle of a face matrix: min corner `(u, v)`, size `w × h`, all in cells.
#[derive(Clone, Copy, Debug)]
struct Rect {
    u: u32,
    v: u32,
    w: u32,
    h: u32,
}

/// `w` set bits starting at bit `u`; `w` in `1..=32`, `u + w <= 32`.
#[inline]
fn run_mask(u: u32, w: u32) -> u32 {
    (u32::MAX >> (32 - w)) << u
}

/// Covers the set bits of one matrix greedily and clears it; `occupied` has a bit for every
/// non-empty row. Rectangles come out in `(v, u)` order of their min corner.
fn greedy_single(rows: &mut [u32; 32], mut occupied: u32, mut emit: impl FnMut(Rect)) {
    while occupied != 0 {
        let v = occupied.trailing_zeros() as usize;
        occupied &= occupied - 1;
        let mut row = rows[v];
        rows[v] = 0;
        while row != 0 {
            let u = row.trailing_zeros();
            let w = (row >> u).trailing_ones();
            let mask = run_mask(u, w);
            row &= !mask;
            let mut h = 1;
            while v + h < 32 && rows[v + h] & mask == mask {
                rows[v + h] &= !mask;
                h += 1;
            }
            emit(Rect {
                u,
                v: v as u32,
                w,
                h: h as u32,
            });
        }
    }
}

/// [`greedy_single`] over several disjoint matrices of one slice at once, so that the
/// rectangles of all of them come out in `(v, u)` order. Each rectangle lies in one matrix and
/// is the one [`greedy_single`] would produce for that matrix alone.
fn greedy_multi(
    rows: &mut [[u32; 32]],
    occupied: &[u32],
    planes: &[u32],
    mut emit: impl FnMut(usize, Rect),
) {
    let mut combined = [0_u32; 32];
    let mut rows_left = 0;
    for &plane in planes {
        let plane = plane as usize;
        rows_left |= occupied[plane];
        for (all, bits) in combined.iter_mut().zip(&rows[plane]) {
            *all |= bits;
        }
    }
    while rows_left != 0 {
        let v = rows_left.trailing_zeros() as usize;
        rows_left &= rows_left - 1;
        while combined[v] != 0 {
            let u = combined[v].trailing_zeros();
            let plane = planes
                .iter()
                .map(|&plane| plane as usize)
                .find(|&plane| rows[plane][v] & (1 << u) != 0)
                .expect("every combined bit comes from one of the matrices");
            let matrix = &mut rows[plane];
            let w = (matrix[v] >> u).trailing_ones();
            let mask = run_mask(u, w);
            matrix[v] &= !mask;
            combined[v] &= !mask;
            let mut h = 1;
            while v + h < 32 && matrix[v + h] & mask == mask {
                matrix[v + h] &= !mask;
                combined[v + h] &= !mask;
                h += 1;
            }
            emit(
                plane,
                Rect {
                    u,
                    v: v as u32,
                    w,
                    h: h as u32,
                },
            );
        }
    }
}

/// The quad of `rect` made by [`PackedQuad::new`], which [`RangeBuffers::push`] reproduces.
fn checked_quad(
    face: Face,
    key: u32,
    slice: usize,
    rect: Rect,
) -> Result<PackedQuad, cp_core::quad::QuadError> {
    let mut cell = [0_u8; 3];
    cell[face.axis()] = slice as u8;
    cell[face.u_axis()] = rect.u as u8;
    cell[face.v_axis()] = rect.v as u8;
    PackedQuad::new(
        cell,
        rect.w as u8,
        rect.h as u8,
        face,
        MaterialId(key as u16),
        ((key >> KEY_UV_ROT_SHIFT) & 3) as u8,
        (key >> KEY_AUTHORED_SHIFT) & 1 != 0,
    )
}

/// Cell box under construction; empty while `min > max`.
#[derive(Clone, Copy, Debug)]
struct BoundsAcc {
    min: [u8; 3],
    max: [u8; 3],
}

impl BoundsAcc {
    const EMPTY: Self = Self {
        min: [u8::MAX; 3],
        max: [0; 3],
    };

    #[inline]
    fn include(&mut self, low: [u8; 3], high: [u8; 3]) {
        for (min, low) in self.min.iter_mut().zip(low) {
            *min = (*min).min(low);
        }
        for (max, high) in self.max.iter_mut().zip(high) {
            *max = (*max).max(high);
        }
    }

    fn get(self) -> Option<CellBounds> {
        (self.min[0] <= self.max[0]).then_some(CellBounds {
            min: self.min,
            max: self.max,
        })
    }
}

/// The output under construction: one buffer per `(layer, range slot)` and the bounds per
/// `(layer, wall)`.
#[derive(Debug)]
struct RangeBuffers {
    quads: Vec<Vec<PackedQuad>>,
    bounds: [BoundsAcc; Layer::COUNT * 2],
}

impl RangeBuffers {
    fn new() -> Self {
        Self {
            quads: vec![Vec::new(); Layer::COUNT * RANGES_PER_LAYER],
            bounds: [BoundsAcc::EMPTY; Layer::COUNT * 2],
        }
    }

    fn clear(&mut self) {
        for quads in &mut self.quads {
            quads.clear();
        }
        self.bounds = [BoundsAcc::EMPTY; Layer::COUNT * 2];
    }

    /// Adds the cells of one face matrix (key `key`, slice `slice` of `face`, rows `rows`,
    /// non-empty rows `occupied`) to the bounds of the range its key selects.
    #[inline]
    fn include(&mut self, face: Face, key: u32, slice: usize, rows: &[u32; 32], occupied: u32) {
        let columns = rows.iter().fold(0, |all, row| all | row);
        if columns == 0 {
            return;
        }
        let mut low = [0_u8; 3];
        let mut high = [0_u8; 3];
        low[face.axis()] = slice as u8;
        high[face.axis()] = slice as u8;
        low[face.u_axis()] = columns.trailing_zeros() as u8;
        high[face.u_axis()] = (31 - columns.leading_zeros()) as u8;
        low[face.v_axis()] = occupied.trailing_zeros() as u8;
        high[face.v_axis()] = (31 - occupied.leading_zeros()) as u8;
        let layer = ((key >> KEY_LAYER_SHIFT) & 3) as usize;
        let wall = key & KEY_WALL != 0;
        self.bounds[layer * 2 + usize::from(wall)].include(low, high);
    }

    /// Appends the quad of `rect` in slice `slice` of `face` to the range its key selects.
    #[inline]
    fn push(&mut self, face: Face, key: u32, slice: usize, rect: Rect) {
        // The words `PackedQuad::new` would make (contract §4.1), built directly because its
        // range checks cost as much as the rest of the emission: greedy rectangles lie inside
        // the chunk by construction and `face_key` has checked uv_rot. Debug builds (and so
        // every test) compare with `PackedQuad::new`.
        let quad = PackedQuad {
            w0: ((slice as u32) << (QUAD_COORD_BITS * face.axis() as u32))
                | (rect.u << (QUAD_COORD_BITS * face.u_axis() as u32))
                | (rect.v << (QUAD_COORD_BITS * face.v_axis() as u32))
                | ((rect.w - 1) << QUAD_W_SHIFT)
                | ((rect.h - 1) << QUAD_H_SHIFT)
                | ((face as u32) << QUAD_FACE_SHIFT),
            w1: key & KEY_W1_BITS,
        };
        debug_assert_eq!(Ok(quad), checked_quad(face, key, slice, rect));
        let layer = ((key >> KEY_LAYER_SHIFT) & 3) as usize;
        let slot = storage_slot(face)
            + if key & KEY_WALL != 0 {
                WALL_RANGE_OFFSET
            } else {
                0
            };
        self.quads[layer * RANGES_PER_LAYER + slot].push(quad);
    }

    /// Concatenates the buffers into `out` in storage order.
    fn write(&self, out: &mut ChunkMesh) {
        out.quads.clear();
        out.quads
            .reserve(self.quads.iter().map(Vec::len).sum::<usize>());
        for layer in Layer::ALL {
            let index = layer.index();
            let mut counts = [0; RANGES_PER_LAYER];
            let first = out.quads.len() as u32;
            for (slot, count) in counts.iter_mut().enumerate() {
                let quads = &self.quads[index * RANGES_PER_LAYER + slot];
                *count = quads.len() as u32;
                out.quads.extend_from_slice(quads);
            }
            let regular_bounds = self.bounds[index * 2].get();
            let wall_bounds = self.bounds[index * 2 + 1].get();
            out.layers[index] = LayerRanges {
                first,
                counts,
                bounds: CellBounds::union_of(regular_bounds, wall_bounds),
                regular_bounds,
                wall_bounds,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use cp_core::face::Orientation;

    use super::*;
    use crate::padded_index;

    /// Air and one opaque cube of material `material`.
    fn tables(material: u16) -> (Vec<BlockFlags>, Vec<BlockRender>, Vec<u16>) {
        let cube = BlockRender::oriented(
            RenderClass::Cube,
            [MaterialId(material); 6],
            Orientation::IDENTITY,
        );
        (
            vec![BlockFlags::empty(), BlockFlags::SOLID | BlockFlags::OPAQUE],
            vec![BlockRender::NONE, cube],
            vec![0, 0],
        )
    }

    #[test]
    fn a_wrapped_generation_does_not_revive_old_entries() {
        // One cube in air, meshed with material 1 and then with material 2. The second call
        // is made to wrap the generation counter back to the value of the first call; the
        // entry of the first call must not count as filled.
        let mut cells = vec![BlockStateId::AIR; PADDED_VOLUME];
        cells[padded_index(4, 5, 6)] = BlockStateId(1);
        let mut mesher = Mesher::new();
        for (material, generation) in [(1, 0), (2, u32::MAX)] {
            let (flags, render, cull_group) = tables(material);
            let input = MeshInput {
                cells: &cells,
                authored: None,
                level: 0,
                tables: MeshTables {
                    flags: &flags,
                    render: &render,
                    cull_group: &cull_group,
                },
            };
            mesher.states.generation = generation;
            let mesh = mesher.mesh(&input).expect("valid input");
            assert_eq!(
                mesher.states.generation, 1,
                "both calls run as generation 1"
            );
            assert_eq!(mesh.quads.len(), 6);
            assert!(
                mesh.quads
                    .iter()
                    .all(|quad| quad.material() == MaterialId(material)),
                "material {material}"
            );
        }
    }

    #[test]
    fn the_state_table_grows_but_never_serves_states_beyond_the_tables() {
        // Meshed with 300 states, then with the first two only: state 100 still has an entry
        // in the table, but it must be unknown to the call with the short tables.
        let mut cells = vec![BlockStateId::AIR; PADDED_VOLUME];
        cells[padded_index(1, 2, 3)] = BlockStateId(100);
        let (mut flags, mut render, mut cull_group) = tables(5);
        flags.resize(300, flags[1]);
        render.resize(300, render[1]);
        cull_group.resize(300, 0);
        let long = MeshInput {
            cells: &cells,
            authored: None,
            level: 0,
            tables: MeshTables {
                flags: &flags,
                render: &render,
                cull_group: &cull_group,
            },
        };
        let short = MeshInput {
            tables: MeshTables {
                flags: &flags[..2],
                render: &render[..2],
                cull_group: &cull_group[..2],
            },
            ..long
        };
        let mut mesher = Mesher::new();
        assert_eq!(mesher.mesh(&long).expect("valid input").quads.len(), 6);
        assert_eq!(
            mesher.mesh(&short),
            Err(MeshError::UnknownState {
                state: BlockStateId(100),
                cell: [1, 2, 3]
            })
        );
        assert_eq!(mesher.states.entries.len(), 300, "the table only grows");
        assert_eq!(mesher.mesh(&long).expect("valid input").quads.len(), 6);
        // The memory figure of the `Mesher` documentation.
        assert_eq!(size_of::<StateInfo>(), 64);
    }
}

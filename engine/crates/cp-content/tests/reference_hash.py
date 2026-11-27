"""Independent port of cp-content's content hash to Python.

Purpose: `Registry::content_hash` must be the same on every machine and in every run. The Rust
tests pin the hash of the built-in minimal content. This script recomputes that hash from the
TOML files and from the written rules alone (contract sections 2.3 and 4.4, the crate
documentation, and the encoding described below), in another language and runtime. Agreement
shows that the pinned number is a property of the content and of the rules, not of one
compiler, platform or hash-map order.

Not run by cargo. Run by hand after touching `registry.rs` (`hash_content`), `canon.rs`, the
state rules in `block.rs`, or the built-in content:

    python crates/cp-content/tests/reference_hash.py

It prints the hash of the built-in content and of `content/base`, and exits with code 0 only if
the built-in hash equals `BUILTIN_HASH` in `src/registry/tests.rs`.

It needs a TOML parser: `tomllib` (Python 3.11+), `tomli`, or the copy that ships inside pip.

The encoding, format 2 (all integers little-endian; `str` = u64 byte length + UTF-8;
`count` = u64; `bool` = u8 0 or 1):

    str "cp-content", u32 format (2)
    count textures, each (sorted by key):
        str key
        u8 0, str recipe, u64 seed, count params, each (sorted by name):
            str name, then  u8 0 + red, green, blue  for a colour ("#rrggbb" or "#rgb")
                         |  u8 1 + str               for a variant (any other string)
        or u8 1, str path
        u8 provenance (own 0, placeholder 1), bool has licence, str licence (or "")
    count materials, each (sorted by key):
        str key, str texture key, f32 roughness, f32 metallic, f32 emissive,
        u8 tint (none 0, grass 1, foliage 2, water 3), u8 flags (cutout 1, translucent 2)
    count blocks, each (sorted by key):
        str key, u8 orient (none 0, up6 1, axis3 2, rot24 3),
        bool solid, bool opaque, bool structural,
        u8 render (none 0, cube 1, cutout 2, fluid 3, model 4),
        bool has faces, then 6 x str material key (local faces +x -x +y -y +z -z),
        f32 hardness, f32 density, u8 light, count tags, each str (sorted),
        bool has cull group, str cull group (or ""),
        bool has lod proxy, str lod proxy (or ""),
        count ext namespaces, each (sorted): str namespace, table
    count cull groups, each str (sorted)
    count states, each (air first, then by key):
        str key, u16 flags, u8 render class,
        bool has faces, then 6 x (str material key, u8 uv_rot) per world face,
        u16 cull group (0 none, else 1 + index in the sorted names),
        u8 lod class (empty 0, fluid 1, solid 2), str key of the lod proxy

    table = count entries, each (sorted by key): str key, value
    value = u8 tag, then: 0 str | 1 u64 two's complement integer | 2 u64 float bits (every NaN
            as 0x7ff8000000000000) | 3 bool | 4 str datetime | 5 count + values | 6 table

    hash = hash_combine(fnv1a64(bytes), len(bytes))

Datetimes inside `[ext]` tables are not supported here (their text form is the TOML library's);
neither the built-in content nor the base pack has any.
"""

import math
import os
import re
import struct
import sys

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib

M = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
AIR = "cubeplanet:air"
FORMAT = 2

# --------------------------------------------------------------------------------------------
# cp_core::hash
# --------------------------------------------------------------------------------------------


def mix64(z):
    z &= M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)


def hash_combine(seed, value):
    return mix64(((seed + GOLDEN) & M) ^ mix64(value))


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & M
    return h


class Canon:
    def __init__(self):
        self.bytes = bytearray()

    def u8(self, value):
        self.bytes += struct.pack("<B", int(value))

    def u16(self, value):
        self.bytes += struct.pack("<H", value)

    def u32(self, value):
        self.bytes += struct.pack("<I", value)

    def u64(self, value):
        self.bytes += struct.pack("<Q", value & M)

    def f32(self, value):
        # Rounds the double to the nearest float, like Rust's `as f32`; +0.0 drops a minus zero.
        self.bytes += struct.pack("<f", float(value) + 0.0)

    def count(self, value):
        self.u64(value)

    def str(self, text):
        raw = text.encode("utf-8")
        self.count(len(raw))
        self.bytes += raw

    def finish(self):
        return hash_combine(fnv1a64(self.bytes), len(self.bytes))


def utf8_order(text):
    return text.encode("utf-8")


def toml_table(canon, table):
    canon.count(len(table))
    for key in sorted(table, key=utf8_order):
        canon.str(key)
        toml_value(canon, table[key])


def toml_value(canon, value):
    # `bool` before `int`: in Python a bool is an int.
    if isinstance(value, bool):
        canon.u8(3)
        canon.u8(value)
    elif isinstance(value, str):
        canon.u8(0)
        canon.str(value)
    elif isinstance(value, int):
        canon.u8(1)
        canon.u64(value)
    elif isinstance(value, float):
        canon.u8(2)
        bits = 0x7FF8000000000000 if math.isnan(value) else struct.unpack("<Q", struct.pack("<d", value))[0]
        canon.u64(bits)
    elif isinstance(value, list):
        canon.u8(5)
        canon.count(len(value))
        for item in value:
            toml_value(canon, item)
    elif isinstance(value, dict):
        canon.u8(6)
        toml_table(canon, value)
    else:
        raise NotImplementedError("datetimes in [ext] tables are not ported: %r" % (value,))


# --------------------------------------------------------------------------------------------
# Faces and orientation, from contract section 2.3
# --------------------------------------------------------------------------------------------

FACES = ["+x", "-x", "+y", "-y", "+z", "-z"]
NORMALS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
UNITS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def neg(v):
    return (-v[0], -v[1], -v[2])


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def t_up(up, v):
    """`T(up)` of the contract: block-local axes to world axes for a block whose top points
    to the face `up`, unturned."""
    x, y, z = v
    return {
        "+y": (x, y, z),
        "-y": (x, -y, -z),
        "+x": (y, -x, z),
        "-x": (-y, x, z),
        "+z": (x, -z, y),
        "-z": (x, z, -y),
    }[up]


def quarter_turn(n, v):
    """A quarter turn of `v` about the unit axis `n`, counter-clockwise seen from its tip."""
    along = n[0] * v[0] + n[1] * v[1] + n[2] * v[2]
    return (
        n[1] * v[2] - n[2] * v[1] + n[0] * along,
        n[2] * v[0] - n[0] * v[2] + n[1] * along,
        n[0] * v[1] - n[1] * v[0] + n[2] * along,
    )


def rotate(orientation, v):
    """`R(up_world, turn x 90 degrees) o T(up)` applied to `v`; `orientation` = (up, turn)."""
    up, turn = orientation
    w = t_up(up, v)
    axis = NORMALS[FACES.index(up)]
    for _ in range(turn):
        w = quarter_turn(axis, w)
    return w


def local_tex_axes(face):
    """(image right, image up) of a face of the unrotated block."""
    normal = NORMALS[face]
    if normal == (0, 1, 0):
        return ((1, 0, 0), (0, 0, -1))
    if normal == (0, -1, 0):
        return ((1, 0, 0), (0, 0, 1))
    tex_v = (0, 1, 0)
    return (cross(tex_v, normal), tex_v)


def quad_tex_axes(face, uv_rot):
    """(image right, image up) of a quad on `face` that carries `uv_rot`.

    The quad's tangent axes are u = (n + 1) % 3 and v = (n + 2) % 3. Seen from outside, (u, v)
    is right-handed on positive faces and mirrored on negative ones, so negative faces start
    from (v, u) (cp-core, `face.rs`). The pair is then rotated by uv_rot quarter turns:
    0:(a, b) 1:(b, -a) 2:(-a, -b) 3:(-b, a).
    """
    axis = face >> 1
    u = UNITS[(axis + 1) % 3]
    v = UNITS[(axis + 2) % 3]
    a, b = (v, u) if face & 1 else (u, v)
    return [(a, b), (b, neg(a)), (neg(a), neg(b)), (neg(b), a)][uv_rot]


def oriented(local_materials, orientation):
    """Per world face: (material of the local face that ends up there, uv_rot)."""
    faces = []
    for world in range(6):
        local = [face for face in range(6) if rotate(orientation, NORMALS[face]) == NORMALS[world]]
        assert len(local) == 1
        tex_u, tex_v = local_tex_axes(local[0])
        wanted = (rotate(orientation, tex_u), rotate(orientation, tex_v))
        rotations = [rot for rot in range(4) if quad_tex_axes(world, rot) == wanted]
        assert len(rotations) == 1
        faces.append((local_materials[local[0]], rotations[0]))
    return faces


# --------------------------------------------------------------------------------------------
# Packs, from contract section 4.4 and the crate documentation
# --------------------------------------------------------------------------------------------

KINDS = [("textures", "texture"), ("materials", "material"), ("blocks", "block")]
RENDER_CLASSES = {"none": 0, "cube": 1, "cutout": 2, "fluid": 3, "model": 4}
ORIENTS = {"none": 0, "up6": 1, "axis3": 2, "rot24": 3}


def read_file(path, table, into):
    with open(path, "rb") as file:
        for entry in tomllib.load(file).get(table, []):
            into[entry["key"]] = entry


def load_packs(directories):
    """Later packs replace whole definitions of earlier ones that have the same key."""
    merged = {"texture": {}, "material": {}, "block": {}}
    for directory in directories:
        for folder, table in KINDS:
            # Every .toml file below the folder, in sub-folders too.
            for root, _, names in os.walk(os.path.join(directory, folder)):
                for name in names:
                    if name.lower().endswith(".toml"):
                        read_file(os.path.join(root, name), table, merged[table])
    return merged


def by_key(definitions):
    return [definitions[key] for key in sorted(definitions, key=utf8_order)]


def local_faces(faces):
    if "all" in faces:
        return [faces["all"]] * 6
    if "top" in faces:
        side = faces["side"]
        return [side, side, faces["top"], faces["bottom"], side, side]
    return [faces[name] for name in FACES]


def state_key(block_key, properties):
    """Sorted properties, default values left out; the bare key for the default state."""
    if not properties:
        return block_key
    return "%s[%s]" % (block_key, ",".join("%s=%s" % pair for pair in sorted(properties)))


def states_of(block):
    """(state key, orientation) for every state of the block; orientation = (up, turn)."""
    key = block["key"]
    orient = block.get("orient", "none")
    if orient == "up6":
        return [(state_key(key, [] if up == "+y" else [("up", up)]), (up, 0)) for up in FACES]
    if orient == "axis3":
        return [
            (state_key(key, [] if axis == "y" else [("axis", axis)]), ("+" + axis, 0)) for axis in "xyz"
        ]
    if orient == "rot24":
        states = []
        for turn in range(4):
            for up in FACES:
                properties = []
                if turn != 0:
                    properties.append(("turn", str(turn)))
                if up != "+y":
                    properties.append(("up", up))
                states.append((state_key(key, properties), (up, turn)))
        return states
    return [(key, ("+y", 0))]


ORIENTATIONS = [(up, turn) for turn in range(4) for up in FACES]


def compose(outer, inner):
    """The orientation that applies `inner` first, then `outer` (`Orientation::compose`)."""
    wanted = [rotate(outer, rotate(inner, unit)) for unit in UNITS]
    found = [o for o in ORIENTATIONS if [rotate(o, unit) for unit in UNITS] == wanted]
    assert len(found) == 1
    return found[0]


def state_orientation(block, key):
    """The orientation of the state `key` of `block`."""
    for state, orientation in states_of(block):
        if state == key:
            return orientation
    raise KeyError(key)


def state_for(block, orientation):
    """The key of the state of `block` for `orientation`, projected onto what the block's
    `orient` can express: `up6` drops the turn, `axis3` keeps the axis of up, `none` has one
    state (`BlockDef::state_for`)."""
    up, turn = orientation
    orient = block.get("orient", "none")
    target = {
        "up6": (up, 0),
        "axis3": ("+" + up[1], 0),
        "rot24": (up, turn),
    }.get(orient, ("+y", 0))
    return [key for key, o in states_of(block) if o == target][0]


def lod_class(block):
    render = block.get("render", "cube")
    solid = block.get("solid", render in ("cube", "cutout", "model"))
    if render in ("cube", "cutout") and solid:
        return 2
    if render == "fluid":
        return 1
    return 0


def content_hash(merged):
    merged["block"].setdefault(AIR, {"key": AIR, "render": "none", "hardness": 0.0, "density": 0.0})
    textures = by_key(merged["texture"])
    materials = by_key(merged["material"])
    blocks = by_key(merged["block"])
    blocks_by_key = {block["key"]: block for block in blocks}
    cull_groups = sorted({block["cull_group"] for block in blocks if "cull_group" in block}, key=utf8_order)

    canon = Canon()
    canon.str("cp-content")
    canon.u32(FORMAT)

    canon.count(len(textures))
    for texture in textures:
        canon.str(texture["key"])
        if "procedural" in texture:
            procedural = texture["procedural"]
            canon.u8(0)
            canon.str(procedural["recipe"])
            canon.u64(procedural["seed"])
            params = procedural.get("params", {})
            canon.count(len(params))
            for name in sorted(params, key=utf8_order):
                value = params[name]
                canon.str(name)
                if value.startswith("#"):
                    digits = value[1:]
                    if len(digits) == 3:
                        digits = "".join(digit * 2 for digit in digits)
                    canon.u8(0)
                    canon.bytes += bytes.fromhex(digits)
                else:
                    canon.u8(1)
                    canon.str(value)
        else:
            canon.u8(1)
            canon.str(texture["file"])
        canon.u8({"own": 0, "placeholder": 1}[texture["provenance"]])
        canon.u8("licence" in texture)
        canon.str(texture.get("licence", ""))

    canon.count(len(materials))
    for material in materials:
        canon.str(material["key"])
        canon.str(material["texture"])
        canon.f32(material.get("roughness", 1.0))
        canon.f32(material.get("metallic", 0.0))
        canon.f32(material.get("emissive", 0.0))
        canon.u8({"none": 0, "grass": 1, "foliage": 2, "water": 3}[material.get("tint", "none")])
        canon.u8(sum({"cutout": 1, "translucent": 2}[flag] for flag in material.get("flags", [])))

    states = []
    canon.count(len(blocks))
    for block in blocks:
        render = block.get("render", "cube")
        solid = block.get("solid", render in ("cube", "cutout", "model"))
        opaque = block.get("opaque", render == "cube")
        structural = block.get("structural", False)
        light = block.get("light", 0)
        faces = local_faces(block["faces"]) if "faces" in block else None
        group = block.get("cull_group")
        proxy = block.get("lod_proxy")
        ext = block.get("ext", {})

        canon.str(block["key"])
        canon.u8(ORIENTS[block.get("orient", "none")])
        canon.u8(solid)
        canon.u8(opaque)
        canon.u8(structural)
        canon.u8(RENDER_CLASSES[render])
        canon.u8(faces is not None)
        for material in faces or []:
            canon.str(material)
        canon.f32(block["hardness"])
        canon.f32(block["density"])
        canon.u8(light)
        tags = sorted(block.get("tags", []), key=utf8_order)
        canon.count(len(tags))
        for tag in tags:
            canon.str(tag)
        canon.u8(group is not None)
        canon.str(group or "")
        canon.u8(proxy is not None)
        canon.str(proxy or "")
        canon.count(len(ext))
        for namespace in sorted(ext, key=utf8_order):
            canon.str(namespace)
            toml_table(canon, ext[namespace])

        # SOLID 1, OPAQUE 2, CUTOUT 4, FLUID 8, EMISSIVE 16, ENTITY 32 (never set), STRUCTURAL 64
        flags = 1 * solid + 2 * opaque + 16 * (light > 0) + 64 * structural
        flags += {"none": 0, "cube": 0, "cutout": 4, "fluid": 8, "model": 0}[render]
        group_number = 0 if group is None else 1 + cull_groups.index(group)
        for key, orientation in states_of(block):
            world_faces = None if faces is None else oriented(faces, orientation)
            if render != "model":
                lod = (lod_class(block), key)
            elif proxy is not None:
                # The proxy as written stands in for the unrotated model; a rotated state gets
                # it turned the same way.
                proxy_block = blocks_by_key[proxy.split("[")[0]]
                turned = compose(orientation, state_orientation(proxy_block, proxy))
                lod = (lod_class(proxy_block), state_for(proxy_block, turned))
            else:
                lod = (0, AIR)
            states.append((key, flags, RENDER_CLASSES[render], world_faces, group_number, lod))

    canon.count(len(cull_groups))
    for group in cull_groups:
        canon.str(group)

    states.sort(key=lambda state: (state[0] != AIR, utf8_order(state[0])))
    canon.count(len(states))
    for key, flags, render_class, world_faces, group_number, (lod_class_value, proxy_key) in states:
        canon.str(key)
        canon.u16(flags)
        canon.u8(render_class)
        canon.u8(world_faces is not None)
        for material, uv_rot in world_faces or []:
            canon.str(material)
            canon.u8(uv_rot)
        canon.u16(group_number)
        canon.u8(lod_class_value)
        canon.str(proxy_key)
    return canon.finish(), len(blocks), len(states)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    crate = os.path.dirname(here)

    # Vectors pinned by cp-core's own tests: the hash primitives above are the right ones.
    assert fnv1a64(b"foobar") == 0x85944171F73967E8
    assert hash_combine(1, 2) == 0x83B97BF58B9176C8
    # The unrotated block and two rows of cp-core's pinned uv_rot table (turn 0).
    assert [rot for _, rot in oriented(list(range(6)), ("+y", 0))] == [3, 0, 1, 0, 0, 3]
    assert [rot for _, rot in oriented(list(range(6)), ("-z", 0))] == [2, 1, 1, 2, 0, 1]
    # A turn about +y carries the front (+z) to +x, -z, -x (contract section 2.3).
    assert [rotate(("+y", turn), (0, 0, 1)) for turn in range(4)] == [(0, 0, 1), (1, 0, 0), (0, 0, -1), (-1, 0, 0)]
    assert len({key for key, _ in states_of({"key": "t:m", "orient": "rot24"})}) == 24
    # The LOD proxy of a model turns with the model: a shaft laid along x or z is a log along x
    # or z (the case pinned by `rotated_models_turn_their_lod_proxy` in `src/loader/tests.rs`).
    log = {"key": "t:log", "orient": "axis3"}
    furnace = {"key": "t:furnace", "orient": "rot24"}
    upright = state_orientation(log, "t:log")
    assert state_for(log, compose(("+x", 0), upright)) == "t:log[axis=x]"
    assert state_for(log, compose(("-x", 3), upright)) == "t:log[axis=x]"
    assert state_for(log, compose(("-z", 2), upright)) == "t:log[axis=z]"
    assert state_for(log, compose(("+y", 1), upright)) == "t:log"
    assert state_for(furnace, compose(("-y", 0), state_orientation(furnace, "t:furnace"))) == "t:furnace[up=-y]"

    builtin = {"texture": {}, "material": {}, "block": {}}
    for folder, table in KINDS:
        read_file(os.path.join(crate, "src", "builtin", folder + ".toml"), table, builtin[table])
    builtin_hash, blocks, states = content_hash(builtin)
    print("builtin minimal: %2d blocks, %2d states, hash 0x%016x" % (blocks, states, builtin_hash))

    base = os.path.join(crate, "..", "..", "content", "base")
    base_hash, blocks, states = content_hash(load_packs([base]))
    print("content/base:    %2d blocks, %2d states, hash 0x%016x" % (blocks, states, base_hash))

    with open(os.path.join(crate, "src", "registry", "tests.rs"), encoding="utf-8") as file:
        pinned = re.search(r"const BUILTIN_HASH: u64 = (0x[0-9a-fA-F_]+);", file.read())
    if pinned is None:
        print("MISMATCH: BUILTIN_HASH not found in src/registry/tests.rs")
        return 1
    expected = int(pinned.group(1).replace("_", ""), 16)
    if expected != builtin_hash:
        print("MISMATCH: src/registry/tests.rs pins 0x%016x" % expected)
        return 1
    print("OK: BUILTIN_HASH in src/registry/tests.rs matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())

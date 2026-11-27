"""Independent port of cp-core's hashing, random numbers and noise to Python.

Purpose: the Rust tests pin exact output bits (hashes of sample grids, single values). This
script recomputes a representative part of those pins from the algorithm descriptions alone,
in another language and runtime. Agreement shows that the pinned numbers are a property of the
algorithms (IEEE 754 arithmetic in a fixed order), not of one compiler or platform.

Not run by cargo. Run by hand after touching `hash.rs` or `noise/`:

    python crates/cp-core/tests/reference_port.py

Exit code 0 and only "OK" lines mean every recomputed value matches the constants in
`src/hash.rs` and `src/noise/tests.rs`. Python floats are IEEE doubles and `math.sqrt` is
correctly rounded, so every operation below is bit-exact; the order of operations mirrors the
Rust source.
"""

import math
import struct
import sys

M = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
failures = 0


def check(name, got, expected):
    global failures
    ok = got == expected
    failures += 0 if ok else 1
    print("%-22s %s" % (name, "OK" if ok else "MISMATCH: got %r, expected %r" % (got, expected)))


# --------------------------------------------------------------------------------------------
# hash.rs
# --------------------------------------------------------------------------------------------


def mix64(z):
    z &= M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)


def hash_combine(seed, value):
    return mix64(((seed + GOLDEN) & M) ^ mix64(value))


def hash_ivec3(seed, x, y, z):
    xy = (x & 0xFFFFFFFF) | ((y & 0xFFFFFFFF) << 32)
    return hash_combine(hash_combine(seed, xy), z & 0xFFFFFFFF)


def hash_bytes(data):
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & M
    return h


class Rng:
    """PCG-XSH-RR 64/32."""

    def __init__(self, seed, stream):
        self.state = 0
        self.inc = ((stream << 1) | 1) & M
        self.step()
        self.state = (self.state + seed) & M
        self.step()

    def step(self):
        self.state = (self.state * 6364136223846793005 + self.inc) & M

    def next_u32(self):
        old = self.state
        self.step()
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = old >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def fork(self, stream):
        return Rng(hash_combine(self.state, stream), hash_combine(self.inc, stream))

    def range_u32(self, lo, hi):
        span = hi - lo
        product = self.next_u32() * span
        if (product & 0xFFFFFFFF) < span:
            threshold = ((1 << 32) - span) % span
            while (product & 0xFFFFFFFF) < threshold:
                product = self.next_u32() * span
        return lo + (product >> 32)


rng = Rng(42, 54)
check(
    "pcg32 vector",
    [rng.next_u32() for _ in range(6)],
    [0xA15C02B7, 0x7B47F409, 0xBA1D3330, 0x83D2F293, 0xBFA4784B, 0xCBED606E],
)
check(
    "mix64 (splitmix64)",
    [mix64(GOLDEN), mix64(GOLDEN * 2), mix64(GOLDEN * 3)],
    [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F],
)
check(
    "combined hashes",
    [
        hash_combine(1, 2),
        hash_combine(M, 0),
        hash_ivec3(0, 0, 0, 0),
        hash_ivec3(42, 1, -2, 3),
        hash_ivec3(42, -(2**31), 2**31 - 1, -1),
    ],
    [
        0x83B97BF58B9176C8,
        0xE4D971771B652C20,
        0xA706DD2F4D197E6F,
        0x8F346C92F6B71AB1,
        0x2FE7FA971B268458,
    ],
)
child = Rng(42, 54).fork(7)
check("fork", [child.next_u32() for _ in range(3)], [0xB80BD308, 0x2296EEFC, 0x4E3A5791])
rng = Rng(42, 54)
check(
    "range_u32",
    [rng.range_u32(0, 6), rng.range_u32(10, 20), rng.range_u32(0, 1000000)],
    [3, 14, 727008],
)
check(
    "fnv-1a",
    [hash_bytes(b""), hash_bytes(b"a"), hash_bytes(b"foobar")],
    [0xCBF29CE484222325, 0xAF63DC4C8601EC8C, 0x85944171F73967E8],
)

# --------------------------------------------------------------------------------------------
# noise/lattice.rs
# --------------------------------------------------------------------------------------------

SALT_VALUE = 0x56414C5545A73D19
SALT_GRADIENT = 0x475241449C0B62E5
SALT_VALUE_GRADIENT = 0x564752417B36D40F
SALT_WORLEY = 0x574F524CE1F358B7
SALT_OCTAVE = 0x4F4354412D8EA6C3


def seed_state(seed, salt):
    return mix64(seed ^ salt)


def fold(state, coordinate):
    x = ((state ^ (coordinate & M)) * 0xD6E8FEB86659FD93) & M
    return x ^ (x >> 32)


finish = mix64


def split(x):
    truncated = int(x)
    floor = truncated - (1 if x < float(truncated) else 0)
    return floor, x - float(floor)


def fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    return a + (b - a) * t


def signed_unit(h):
    return float(h >> 11) * (1.0 / 4503599627370496.0) - 1.0


def clamp(v):
    return max(-1.0, min(1.0, v))


def const_sqrt(x):
    y = x if x > 1.0 else 1.0
    for _ in range(48):
        y = 0.5 * (y + x / y)
    return y


FRAC_PI_2 = 1.5707963267948966


def sin_cos_turns(turns):
    quarter_turns = turns * 4.0
    quadrant = int(quarter_turns)
    r = (quarter_turns - float(quadrant)) * FRAC_PI_2
    r2 = r * r
    sin, cos, sin_term, cos_term = r, 1.0, r, 1.0
    for k in range(1, 15):
        n = float(2 * k)
        cos_term = -cos_term * r2 / ((n - 1.0) * n)
        sin_term = -sin_term * r2 / (n * (n + 1.0))
        cos += cos_term
        sin += sin_term
    return [(sin, cos), (cos, -sin), (-sin, -cos), (-cos, sin)][quadrant & 3]


GOLDEN_TURN = 0.6180339887498949
GRADIENTS2 = [None] * 256
for i in range(128):
    s, c = sin_cos_turns((float(i) + 0.5) / 256.0)
    length = const_sqrt(s * s + c * c)
    x, y = c / length, s / length
    GRADIENTS2[i] = (x, y)
    GRADIENTS2[i + 128] = (-x, -y)
GRADIENTS3 = [None] * 256
for i in range(128):
    z = (float(i) + 0.5) / 128.0
    radius = const_sqrt(1.0 - z * z)
    turns = float(i) * GOLDEN_TURN
    turns = turns - float(int(turns))
    s, c = sin_cos_turns(turns)
    x, y = radius * c, radius * s
    length = const_sqrt(x * x + y * y + z * z)
    x, y, z = x / length, y / length, z / length
    GRADIENTS3[2 * i] = (x, y, z)
    GRADIENTS3[2 * i + 1] = (-x, -y, -z)
SQRT_2 = 1.4142135623730951
GRADIENT2_SCALE = SQRT_2
GRADIENT3_SCALE = 2.0 / const_sqrt(3.0)

# --------------------------------------------------------------------------------------------
# noise/basis.rs
# --------------------------------------------------------------------------------------------


def blend2(state, p, corner):
    (ix, fx), (iy, fy) = split(p[0]), split(p[1])
    x0, x1 = fold(state, ix), fold(state, ix + 1)
    y0, y1 = iy, iy + 1

    def at(x, y, dx, dy):
        return corner(finish(fold(x, y)), dx, dy)

    gx, gy = fx - 1.0, fy - 1.0
    u = fade(fx)
    low = lerp(at(x0, y0, fx, fy), at(x1, y0, gx, fy), u)
    high = lerp(at(x0, y1, fx, gy), at(x1, y1, gx, gy), u)
    return lerp(low, high, fade(fy))


def blend3(state, p, corner):
    (ix, fx), (iy, fy), (iz, fz) = split(p[0]), split(p[1]), split(p[2])
    x0, x1 = fold(state, ix), fold(state, ix + 1)
    y0, y1, z0, z1 = iy, iy + 1, iz, iz + 1
    x0y0, x1y0, x0y1, x1y1 = fold(x0, y0), fold(x1, y0), fold(x0, y1), fold(x1, y1)

    def at(xy, z, dx, dy, dz):
        return corner(finish(fold(xy, z)), dx, dy, dz)

    gx, gy, gz = fx - 1.0, fy - 1.0, fz - 1.0
    u, v = fade(fx), fade(fy)
    near = lerp(
        lerp(at(x0y0, z0, fx, fy, fz), at(x1y0, z0, gx, fy, fz), u),
        lerp(at(x0y1, z0, fx, gy, fz), at(x1y1, z0, gx, gy, fz), u),
        v,
    )
    far = lerp(
        lerp(at(x0y0, z1, fx, fy, gz), at(x1y0, z1, gx, fy, gz), u),
        lerp(at(x0y1, z1, fx, gy, gz), at(x1y1, z1, gx, gy, gz), u),
        v,
    )
    return lerp(near, far, fade(fz))


def slope2(h, dx, dy):
    g = GRADIENTS2[h >> 56]
    return g[0] * dx + g[1] * dy


def slope3(h, dx, dy, dz):
    g = GRADIENTS3[h >> 56]
    return g[0] * dx + g[1] * dy + g[2] * dz


def height(h):
    return signed_unit((h << 8) & M)


def value2(seed, p):
    return clamp(blend2(seed_state(seed, SALT_VALUE), p, lambda h, dx, dy: signed_unit(h)))


def value3(seed, p):
    return clamp(blend3(seed_state(seed, SALT_VALUE), p, lambda h, dx, dy, dz: signed_unit(h)))


def gradient2(seed, p):
    return clamp(blend2(seed_state(seed, SALT_GRADIENT), p, slope2) * GRADIENT2_SCALE)


def gradient3(seed, p):
    return clamp(blend3(seed_state(seed, SALT_GRADIENT), p, slope3) * GRADIENT3_SCALE)


VALUE_WEIGHT2 = SQRT_2 - 1.0
GRADIENT_WEIGHT2 = (1.0 - VALUE_WEIGHT2) * GRADIENT2_SCALE
VALUE_WEIGHT3 = 1.0 / (const_sqrt(7.0) + 1.0)
GRADIENT_WEIGHT3 = (1.0 - VALUE_WEIGHT3) * GRADIENT3_SCALE


def value_gradient2(seed, p):
    def corner(h, dx, dy):
        return VALUE_WEIGHT2 * height(h) + GRADIENT_WEIGHT2 * slope2(h, dx, dy)

    return clamp(blend2(seed_state(seed, SALT_VALUE_GRADIENT), p, corner))


def value_gradient3(seed, p):
    def corner(h, dx, dy, dz):
        return VALUE_WEIGHT3 * height(h) + GRADIENT_WEIGHT3 * slope3(h, dx, dy, dz)

    return clamp(blend3(seed_state(seed, SALT_VALUE_GRADIENT), p, corner))


# --------------------------------------------------------------------------------------------
# noise/worley.rs (exhaustive search over a 7x7x7 block instead of the pruned search)
# --------------------------------------------------------------------------------------------


def worley3(seed, p):
    state = seed_state(seed, SALT_WORLEY)
    (cx, fx), (cy, fy), (cz, fz) = split(p[0]), split(p[1]), split(p[2])
    scale = 1.0 / 2097152.0
    squared = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                h = finish(fold(fold(fold(state, cx + dx), cy + dy), cz + dz))
                jx = float(h >> 43) * scale
                jy = float((h >> 22) & 0x1FFFFF) * scale
                jz = float((h >> 1) & 0x1FFFFF) * scale
                ox = (float(dx) + jx) - fx
                oy = (float(dy) + jy) - fy
                oz = (float(dz) + jz) - fz
                squared.append(ox * ox + oy * oy + oz * oz)
    squared.sort()
    return math.sqrt(squared[0]), math.sqrt(squared[1])


# --------------------------------------------------------------------------------------------
# noise/fbm.rs (plain shape, default parameters)
# --------------------------------------------------------------------------------------------

GOLDEN_FRACTION = 1.0 - 0.6180339887498949
_sin, _cos = sin_cos_turns(0.25 * GOLDEN_FRACTION)
ROTATION2 = [[_cos, -_sin], [_sin, _cos]]
_length = const_sqrt(14.0)
_x, _y, _z = 1.0 / _length, 2.0 / _length, 3.0 / _length
_sin, _cos = sin_cos_turns(GOLDEN_FRACTION)
_k = 1.0 - _cos
ROTATION3 = [
    [_cos + _k * _x * _x, _k * _x * _y - _sin * _z, _k * _x * _z + _sin * _y],
    [_k * _y * _x + _sin * _z, _cos + _k * _y * _y, _k * _y * _z - _sin * _x],
    [_k * _z * _x - _sin * _y, _k * _z * _y + _sin * _x, _cos + _k * _z * _z],
]


def octave_seed(seed, octave):
    return (seed + (octave + 1) * 0xA24BAED4963EE407) & M


def offset2(seed):
    h = mix64(seed ^ SALT_OCTAVE)
    scale = 1.0 / 4294967296.0
    return (float(h >> 32) * scale, float(h & 0xFFFFFFFF) * scale)


def offset3(seed):
    h = mix64(seed ^ SALT_OCTAVE)
    scale = 1.0 / 2097152.0
    return (
        float(h >> 43) * scale,
        float((h >> 22) & 0x1FFFFF) * scale,
        float((h >> 1) & 0x1FFFFF) * scale,
    )


def expand2(n):
    n2 = n * n
    return n * (15.0 + n2 * (3.0 * n2 - 10.0)) * (1.0 / 8.0)


def expand3(n):
    n2 = n * n
    return n * (315.0 + n2 * (n2 * (378.0 + n2 * (35.0 * n2 - 180.0)) - 420.0)) * (1.0 / 128.0)


def fbm2(seed, p, octaves=5, lacunarity=2.0, gain=0.5):
    q = list(p)
    total, total_amplitude, amplitude = 0.0, 0.0, 1.0
    for octave in range(octaves):
        s = octave_seed(seed, octave)
        m = ROTATION2
        q = [m[0][0] * q[0] + m[0][1] * q[1], m[1][0] * q[0] + m[1][1] * q[1]]
        o = offset2(s)
        n = expand2(value_gradient2(s, (q[0] + o[0], q[1] + o[1])))
        q = [q[0] * lacunarity, q[1] * lacunarity]
        total += amplitude * n
        total_amplitude += abs(amplitude)
        amplitude *= gain
    return clamp(total / total_amplitude)


def fbm3(seed, p, octaves=5, lacunarity=2.0, gain=0.5):
    q = list(p)
    total, total_amplitude, amplitude = 0.0, 0.0, 1.0
    for octave in range(octaves):
        s = octave_seed(seed, octave)
        m = ROTATION3
        q = [m[i][0] * q[0] + m[i][1] * q[1] + m[i][2] * q[2] for i in range(3)]
        o = offset3(s)
        n = expand3(value_gradient3(s, (q[0] + o[0], q[1] + o[1], q[2] + o[2])))
        q = [c * lacunarity for c in q]
        total += amplitude * n
        total_amplitude += abs(amplitude)
        amplitude *= gain
    return clamp(total / total_amplitude)


# --------------------------------------------------------------------------------------------
# noise/tests.rs: PINNED_SAMPLES and a selection of PINNED
# --------------------------------------------------------------------------------------------

SEED = 0x00C0FFEE


def bits(values):
    return [struct.pack("<d", v) for v in values]


p2 = (0.3, 0.7)
p3 = (0.3, 0.7, -1.6)
f1, f2 = worley3(SEED, p3)
check(
    "sample values",
    bits(
        [
            value2(SEED, p2),
            gradient2(SEED, p2),
            value3(SEED, p3),
            gradient3(SEED, p3),
            f1,
            f2,
            fbm2(SEED, p2),
            fbm3(SEED, p3),
        ]
    ),
    bits(
        [
            0.267306142564215,
            -0.33976159198059913,
            -0.011498072430835185,
            -0.14642775927369894,
            0.7163968408275815,
            0.8393281908231011,
            0.37846472210099635,
            -0.36237597355757356,
        ]
    ),
)


def grid2():
    points = []
    for iy in range(-32, 32):
        for ix in range(-32, 32):
            points.append((float(ix) * 0.173 + 0.05, float(iy) * 0.211 - 0.03))
    for iy in range(16):
        for ix in range(16):
            points.append((123456.7 + float(ix) * 0.31, -7654321.1 + float(iy) * 0.29))
    return points


def grid3():
    points = []
    for iz in range(-10, 10):
        for iy in range(-10, 10):
            for ix in range(-10, 10):
                points.append(
                    (float(ix) * 0.29 + 0.05, float(iy) * 0.31 - 0.03, float(iz) * 0.37 + 0.11)
                )
    for iz in range(8):
        for iy in range(8):
            for ix in range(8):
                points.append(
                    (
                        123456.7 + float(ix) * 0.31,
                        -7654321.1 + float(iy) * 0.29,
                        33554431.9 + float(iz) * 0.27,
                    )
                )
    return points


def pin(values):
    return hash_bytes(b"".join(struct.pack("<d", v) for v in values))


g2, g3 = grid2(), grid3()
check("value2 grid", pin(value2(SEED, p) for p in g2), 0x006D9F082ED7574F)
check("value3 grid", pin(value3(SEED, p) for p in g3), 0x348373A49EE72194)
check("gradient2 grid", pin(gradient2(SEED, p) for p in g2), 0xE6CC1CB29BE4581D)
check("gradient3 grid", pin(gradient3(SEED, p) for p in g3), 0x56E5D27F5E1F5EFD)
check("worley3 grid", pin(v for p in g3 for v in worley3(SEED, p)), 0x5BF4B9B95988E440)
check("fbm.sample2 grid", pin(fbm2(SEED, p) for p in g2), 0xC15998DF5FCCE484)
check("fbm.sample3 grid", pin(fbm3(SEED, p) for p in g3), 0x056674E09153507A)

sys.exit(1 if failures else 0)

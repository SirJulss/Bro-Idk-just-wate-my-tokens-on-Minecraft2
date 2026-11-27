// Value noise for shading: smooth pseudo-random values between the points of an integer
// lattice, from integer hashing alone (no textures), so every GPU computes the same values. It
// only shades surfaces and the sky; no simulation reads it.

// A value in 0..1 for the lattice point `p`.
fn lattice_value(p: vec2<i32>) -> f32 {
    var h = (bitcast<u32>(p.x) * 0x8da6b343u) ^ (bitcast<u32>(p.y) * 0xd8163841u);
    h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
    h = h ^ (h >> 12u);
    return f32(h & 0xffffu) * (1.0 / 65535.0);
}

// Smooth noise in 0..1 with one lattice cell per unit of `p`: the lattice values blended with
// the smoothstep of the position inside the cell.
fn value_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let w = f * f * (3.0 - 2.0 * f);
    let i = vec2<i32>(cell);
    let a = lattice_value(i);
    let b = lattice_value(i + vec2<i32>(1, 0));
    let c = lattice_value(i + vec2<i32>(0, 1));
    let d = lattice_value(i + vec2<i32>(1, 1));
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

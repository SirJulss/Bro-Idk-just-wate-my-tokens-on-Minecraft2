// Octahedral encoding of unit vectors in two values of 0..1 (Cigolle et al., "A Survey of
// Efficient Representations for Independent Unit Vectors", 2014). The G-buffer stores its
// normals this way (contract §5.3).

fn octahedral_wrap(v: vec2<f32>) -> vec2<f32> {
    let signs = select(vec2<f32>(-1.0), vec2<f32>(1.0), v >= vec2<f32>(0.0));
    return (vec2<f32>(1.0) - abs(v.yx)) * signs;
}

// `n` must have unit length.
fn octahedral_encode(n: vec3<f32>) -> vec2<f32> {
    var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if n.z < 0.0 {
        p = octahedral_wrap(p);
    }
    return p * 0.5 + vec2<f32>(0.5);
}

fn octahedral_decode(encoded: vec2<f32>) -> vec3<f32> {
    let f = encoded * 2.0 - vec2<f32>(1.0);
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = clamp(-n.z, 0.0, 1.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

// AgX tone mapping (Troy Sobotka) in the polynomial approximation of Benjamin Wrensch (2023):
// scene-linear Rec.709 in, display-linear Rec.709 out. The sRGB transfer function is applied
// by the target format or by the caller.

// The log2 range the curve spans: 9 stops below and 5.5 stops above middle grey (0.18). The
// reference range (10 below, 6.5 above) keeps a sunlit white card at about 190 sRGB and turns
// a sunny day grey and matt; with one stop less headroom a white card in the sun reaches about
// 225 and sunlit cloud tops about 245, while middle grey stays near 120.
const AGX_MIN_EV: f32 = -11.47393;
const AGX_MAX_EV: f32 = 3.026069;

// The look applied in AgX's display-encoded space (the ASC CDL step of the reference
// implementation): a little contrast (power), then vibrance instead of a global saturation,
// so that pale colours (sky, haze, sand) gain up to VIBRANCE of their colourfulness while
// colours that are already strong (sunlit grass, leaves) keep theirs. The darker tones lose
// colour (by up to SHADOW_DESATURATION, fading out towards SHADOW_LUMA), as film and the eye
// do: the toe of the curve would otherwise stretch the ratios of dark channels into neon green
// and black, and a dusk meadow, lit by a sun with little blue left, into mud olive. Sunlit
// noon grass lies above SHADOW_LUMA and keeps its colour.
const AGX_LOOK_POWER: f32 = 1.1;
const AGX_LOOK_VIBRANCE: f32 = 0.3;
const AGX_LOOK_SHADOW_DESATURATION: f32 = 0.7;
const AGX_LOOK_SHADOW_LUMA: f32 = 0.55;

fn agx_look(v: vec3<f32>) -> vec3<f32> {
    let graded = pow(max(v, vec3<f32>(0.0)), vec3<f32>(AGX_LOOK_POWER));
    let luma = dot(graded, vec3<f32>(0.2126, 0.7152, 0.0722));
    let chroma = max(graded.r, max(graded.g, graded.b)) - min(graded.r, min(graded.g, graded.b));
    var gain = 1.0 + AGX_LOOK_VIBRANCE * (1.0 - smoothstep(0.15, 0.45, chroma));
    gain *= 1.0 - AGX_LOOK_SHADOW_DESATURATION
        * (1.0 - smoothstep(0.0, AGX_LOOK_SHADOW_LUMA, luma));
    return vec3<f32>(luma) + gain * (graded - vec3<f32>(luma));
}

// The sixth-order fit of the AgX base contrast curve on 0..1.
fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2
        + 0.1191 * x - vec3<f32>(0.00232);
}

fn agx_tonemap(color: vec3<f32>) -> vec3<f32> {
    // Column-major: the inset and outset matrices of the reference implementation.
    let inset = mat3x3<f32>(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104,
    );
    let outset = mat3x3<f32>(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116,
    );
    var v = inset * max(color, vec3<f32>(1e-10));
    v = clamp(log2(v), vec3<f32>(AGX_MIN_EV), vec3<f32>(AGX_MAX_EV));
    v = (v - vec3<f32>(AGX_MIN_EV)) / (AGX_MAX_EV - AGX_MIN_EV);
    v = agx_look(agx_contrast(v));
    v = outset * v;
    return pow(max(v, vec3<f32>(0.0)), vec3<f32>(2.2));
}

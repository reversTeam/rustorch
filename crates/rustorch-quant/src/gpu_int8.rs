//! Quantized int8 GPU GEMM via WGSL — Phase 3 task `377d069d`.
//!
//! WGSL doesn't have native dp4a (4× int8 dot product in one
//! instruction); we emulate it with packed-i32 unpacking + 4 mul-
//! adds. The shader works on int32 buffers where each 32-bit word
//! holds 4 packed int8 values.
//!
//! ## Layout
//!
//! - `a`: i32 buffer, length = (M * K) / 4 (each i32 packs 4 i8s)
//! - `b`: i32 buffer, length = (K * N) / 4
//! - `c`: f32 buffer, length = M * N (dequantized output)
//!
//! `K` must be a multiple of 4 (the packing unit).

/// WGSL source for the int8 GEMM with dp4a-style packed accumulation.
pub const INT8_GEMM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<i32>;
@group(0) @binding(1) var<storage, read>       b: array<i32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             params: array<vec4<u32>, 1>;

// params[0] = (m, k, n, scale_bits)
// scale = bitcast<f32>(params[0].w)

fn dp4a_i8(a_packed: i32, b_packed: i32) -> i32 {
    // Emulate vpdpbusd / sdot: unpack two 4×i8 vectors and accumulate
    // into i32. Each lane is sign-extended.
    var acc: i32 = 0;
    let av_arr = vec4<i32>(
        ((a_packed << 24) >> 24),
        ((a_packed << 16) >> 24),
        ((a_packed << 8)  >> 24),
        ((a_packed)       >> 24),
    );
    let bv_arr = vec4<i32>(
        ((b_packed << 24) >> 24),
        ((b_packed << 16) >> 24),
        ((b_packed << 8)  >> 24),
        ((b_packed)       >> 24),
    );
    acc = av_arr.x * bv_arr.x
        + av_arr.y * bv_arr.y
        + av_arr.z * bv_arr.z
        + av_arr.w * bv_arr.w;
    return acc;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = params[0].x;
    let k = params[0].y;
    let n = params[0].z;
    let scale = bitcast<f32>(params[0].w);
    let row = gid.y;
    let col = gid.x;
    if (row >= m || col >= n) { return; }

    // Each i32 holds 4 packed i8s along the K axis.
    let k_packed = k / 4u;
    var acc: i32 = 0;
    for (var kp: u32 = 0u; kp < k_packed; kp = kp + 1u) {
        let a_packed = a[row * k_packed + kp];
        // b is laid out [k, n] row-major in i8 → packed along K
        // with the same packing factor; offset = (kp*4 * n + col) / 4.
        // For simplicity (and matching the CPU scalar impl), the
        // host-side dispatcher transposes b before upload so the
        // inner dot uses the same packing as a.
        let b_packed = b[col * k_packed + kp];
        acc = acc + dp4a_i8(a_packed, b_packed);
    }
    c[row * n + col] = f32(acc) * scale;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_contains_dp4a_emulation() {
        assert!(INT8_GEMM_WGSL.contains("dp4a_i8"));
        assert!(INT8_GEMM_WGSL.contains("vec4<i32>"));
    }

    #[test]
    fn wgsl_uses_packed_i32_buffers() {
        assert!(INT8_GEMM_WGSL.contains("array<i32>"));
        assert!(INT8_GEMM_WGSL.contains("array<f32>")); // output is f32
    }

    #[test]
    fn wgsl_has_2d_workgroup() {
        assert!(INT8_GEMM_WGSL.contains("workgroup_size(8, 8)"));
    }
}

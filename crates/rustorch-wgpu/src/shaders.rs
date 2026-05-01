//! WGSL shader sources for element-wise ops.
//!
//! v1 ships F32 only. The shaders read from `lhs` (and `rhs` for
//! binary) storage buffers and write to `out`. Bind groups follow the
//! convention:
//! - binding 0: lhs (read-only storage buffer)
//! - binding 1: rhs (read-only storage buffer)   [binary only]
//! - binding 2: out (read-write storage buffer)
//! - binding 3: meta (uniform u32 array; `[0]` = numel)              [scalar metadata]
//!
//! Workgroup size is 64 (`@workgroup_size(64)`), tunable in a later
//! slice.

/// Binary kernel template — `<OP>` is one of `+`, `-`, `*`, `/`.
fn binary_template(op: &str) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> meta: array<vec4<u32>, 1>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = meta[0].x;
    if (i >= n) {{ return; }}
    out[i] = lhs[i] {op} rhs[i];
}}
"#
    )
}

/// Unary kernel template — `<EXPR>` is a WGSL expression in `x`.
fn unary_template(expr: &str) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> meta: array<vec4<u32>, 1>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let n = meta[0].x;
    if (i >= n) {{ return; }}
    let x = inp[i];
    out[i] = {expr};
}}
"#
    )
}

/// WGSL source for `add(lhs, rhs) -> out`.
pub fn add_f32() -> String {
    binary_template("+")
}
/// WGSL source for `sub`.
pub fn sub_f32() -> String {
    binary_template("-")
}
/// WGSL source for `mul`.
pub fn mul_f32() -> String {
    binary_template("*")
}
/// WGSL source for `div`.
pub fn div_f32() -> String {
    binary_template("/")
}

/// WGSL source for `relu` (`max(0, x)`).
pub fn relu_f32() -> String {
    unary_template("max(0.0, x)")
}
/// WGSL source for `neg` (`-x`).
pub fn neg_f32() -> String {
    unary_template("-x")
}
/// WGSL source for `sigmoid`.
pub fn sigmoid_f32() -> String {
    unary_template("1.0 / (1.0 + exp(-x))")
}
/// WGSL source for `tanh` (uses WGSL's built-in `tanh`).
pub fn tanh_f32() -> String {
    unary_template("tanh(x)")
}
/// WGSL source for `silu` (`x * sigmoid(x)`).
pub fn silu_f32() -> String {
    unary_template("x * (1.0 / (1.0 + exp(-x)))")
}

/// Whether the kernel is binary (true) or unary (false). Drives the
/// dispatch path bind-group layout.
pub fn is_binary(op: &'static str) -> bool {
    matches!(op, "add" | "sub" | "mul" | "div")
}

/// Lookup WGSL source by op name (panics on unknown op).
pub fn source_for(op: &'static str) -> String {
    match op {
        "add" => add_f32(),
        "sub" => sub_f32(),
        "mul" => mul_f32(),
        "div" => div_f32(),
        "relu" => relu_f32(),
        "neg" => neg_f32(),
        "sigmoid" => sigmoid_f32(),
        "tanh" => tanh_f32(),
        "silu" => silu_f32(),
        other => panic!("unknown wgpu kernel: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_template_contains_plus() {
        assert!(add_f32().contains("lhs[i] + rhs[i]"));
    }

    #[test]
    fn unary_template_uses_x() {
        let s = relu_f32();
        assert!(s.contains("max(0.0, x)"));
        assert!(s.contains("inp"));
        assert!(s.contains("out"));
    }

    #[test]
    fn binary_classification() {
        assert!(is_binary("add"));
        assert!(is_binary("sub"));
        assert!(is_binary("mul"));
        assert!(is_binary("div"));
        assert!(!is_binary("relu"));
        assert!(!is_binary("sigmoid"));
    }

    #[test]
    fn source_dispatch_returns_non_empty() {
        for op in &[
            "add", "sub", "mul", "div", "relu", "neg", "sigmoid", "tanh", "silu",
        ] {
            assert!(!source_for(op).is_empty(), "empty source for {op}");
        }
    }
}

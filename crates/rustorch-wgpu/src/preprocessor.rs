//! Minimal WGSL preprocessor: `#include "fragment.wgsl"` and macro
//! substitution `{{KEY}}` → value.
//!
//! WGSL itself has no preprocessor, but kernel authors often want to
//! share a chunk of code (e.g. an online-softmax inner loop) across
//! several shaders. This module supplies the primitives without
//! pulling in a heavy templating engine:
//!
//! - [`expand_includes`] resolves `#include "name"` directives by
//!   looking the name up in the supplied `fragments` table. Includes
//!   are expanded recursively up to a configurable depth.
//! - [`expand_macros`] substitutes `{{KEY}}` → `value` for every key
//!   in the supplied table. Useful for workgroup-size and dtype-tag
//!   substitution in templates.
//! - [`PreprocessError`] surfaces include-not-found / depth-exceeded
//!   so kernel authors get a precise diagnostic.

use std::collections::HashMap;

/// Errors raised by the preprocessor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreprocessError {
    /// `#include "X"` referenced a name not in the fragments table.
    IncludeNotFound(String),
    /// Recursive includes exceeded the depth cap (default 32) — likely
    /// an infinite loop.
    IncludeDepthExceeded,
}

impl core::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PreprocessError::IncludeNotFound(n) => {
                write!(f, "WGSL preprocess: include not found: {}", n)
            },
            PreprocessError::IncludeDepthExceeded => {
                write!(
                    f,
                    "WGSL preprocess: include depth exceeded (cyclic include?)"
                )
            },
        }
    }
}

impl std::error::Error for PreprocessError {}

const MAX_INCLUDE_DEPTH: usize = 32;

/// Resolve every `#include "name"` directive in `source` by looking
/// `name` up in `fragments` and substituting the fragment text in
/// place. Recursive: included fragments may themselves include other
/// fragments, up to [`MAX_INCLUDE_DEPTH`].
///
/// Lines whose first non-whitespace token is `#include` are matched.
/// All other input is passed through verbatim.
pub fn expand_includes(
    source: &str,
    fragments: &HashMap<&str, &str>,
) -> Result<String, PreprocessError> {
    expand_inner(source, fragments, 0)
}

fn expand_inner(
    source: &str,
    fragments: &HashMap<&str, &str>,
    depth: usize,
) -> Result<String, PreprocessError> {
    if depth > MAX_INCLUDE_DEPTH {
        return Err(PreprocessError::IncludeDepthExceeded);
    }
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("#include") {
            let name = parse_include_arg(rest)?;
            let body = fragments
                .get(name.as_str())
                .ok_or_else(|| PreprocessError::IncludeNotFound(name.clone()))?;
            out.push_str(&expand_inner(body, fragments, depth + 1)?);
            if !out.ends_with('\n') {
                out.push('\n');
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    Ok(out)
}

fn parse_include_arg(rest: &str) -> Result<String, PreprocessError> {
    // Expect `<whitespace>"<name>"<rest>` — find the first quoted
    // string. Anything else surfaces as IncludeNotFound("<malformed>").
    let trimmed = rest.trim_start();
    if let Some(stripped) = trimmed.strip_prefix('"') {
        if let Some(end) = stripped.find('"') {
            return Ok(stripped[..end].to_string());
        }
    }
    Err(PreprocessError::IncludeNotFound(format!(
        "<malformed: {}>",
        trimmed.trim()
    )))
}

/// Replace every `{{KEY}}` token in `source` with the value pinned
/// to that key in `vars`. Untouched if the table is empty. Use for
/// workgroup-size and dtype-tag substitution before submitting WGSL
/// to the device.
pub fn expand_macros(source: &str, vars: &HashMap<&str, &str>) -> String {
    let mut out = source.to_string();
    for (key, val) in vars {
        let placeholder = format!("{{{{{}}}}}", key); // {{KEY}}
        out = out.replace(&placeholder, val);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_no_directives() {
        let src = "fn main() { return; }";
        let frags = HashMap::new();
        let out = expand_includes(src, &frags).unwrap();
        assert!(out.contains("fn main()"));
    }

    #[test]
    fn include_substitutes_named_fragment() {
        let mut frags: HashMap<&str, &str> = HashMap::new();
        frags.insert("hello.wgsl", "let greeting: i32 = 42;");
        let src = "#include \"hello.wgsl\"\nfn main() {{}}";
        let out = expand_includes(src, &frags).unwrap();
        assert!(out.contains("greeting: i32 = 42"));
    }

    #[test]
    fn include_unknown_fragment_errors() {
        let frags: HashMap<&str, &str> = HashMap::new();
        let src = "#include \"missing.wgsl\"";
        let err = expand_includes(src, &frags).unwrap_err();
        assert_eq!(err, PreprocessError::IncludeNotFound("missing.wgsl".into()));
    }

    #[test]
    fn cyclic_include_hits_depth_cap() {
        let mut frags: HashMap<&str, &str> = HashMap::new();
        // Fragment A includes B, B includes A → infinite loop.
        frags.insert("a", "#include \"b\"");
        frags.insert("b", "#include \"a\"");
        let src = "#include \"a\"";
        let err = expand_includes(src, &frags).unwrap_err();
        assert_eq!(err, PreprocessError::IncludeDepthExceeded);
    }

    #[test]
    fn macro_substitution_replaces_all_occurrences() {
        let mut vars: HashMap<&str, &str> = HashMap::new();
        vars.insert("WG", "64");
        vars.insert("DTYPE", "f32");
        let src = "@workgroup_size({{WG}}) fn main(x: {{DTYPE}}) -> {{DTYPE}} {}";
        let out = expand_macros(src, &vars);
        assert!(out.contains("@workgroup_size(64)"));
        assert!(out.contains("x: f32"));
        assert!(out.contains("-> f32"));
        assert!(!out.contains("{{"));
    }

    #[test]
    fn include_then_macros_chain() {
        // `#include` must be on its own line (post-trim_start). The
        // chain here writes the include on a dedicated line, then
        // applies macro substitution to the expanded source.
        let mut frags: HashMap<&str, &str> = HashMap::new();
        frags.insert("body", "let v: {{DTYPE}} = 0.0;");
        let src = "fn main() {\n  #include \"body\"\n}";
        let out = expand_includes(src, &frags).unwrap();
        let mut vars: HashMap<&str, &str> = HashMap::new();
        vars.insert("DTYPE", "f32");
        let out = expand_macros(&out, &vars);
        assert!(out.contains("let v: f32 = 0.0"));
    }
}

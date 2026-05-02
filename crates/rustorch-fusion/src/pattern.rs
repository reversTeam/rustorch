//! Pattern DSL — describes the op-chain shapes the matcher looks for.

/// Kinds of ops the matcher knows about. Concrete enough for the
/// 15-pattern library; extensible by adding variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Matrix multiplication.
    Matmul,
    /// Bias add (broadcast over batch).
    BiasAdd,
    /// ReLU activation.
    Relu,
    /// GELU activation (tanh approximation).
    Gelu,
    /// SiLU / swish activation.
    Silu,
    /// LayerNorm.
    LayerNorm,
    /// RMSNorm.
    RmsNorm,
    /// Linear (fused matmul+bias).
    Linear,
    /// Element-wise add.
    Add,
    /// Element-wise mul.
    Mul,
    /// Element-wise sub.
    Sub,
    /// Element-wise div.
    Div,
    /// Sigmoid.
    Sigmoid,
    /// Tanh.
    Tanh,
    /// Generic (used for "any op" wildcards).
    Any,
}

/// A pattern is a linear chain of `OpKind`s. Matched in DAG order:
/// each consecutive op must consume the previous op's output as its
/// primary input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern {
    /// Human-readable label (e.g. "matmul+bias+relu").
    pub label: String,
    /// Sequence of ops; first one is the chain's root.
    pub chain: Vec<OpKind>,
}

impl Pattern {
    /// Build a pattern from a label + chain.
    pub fn new(label: &str, chain: Vec<OpKind>) -> Self {
        Self {
            label: label.to_string(),
            chain,
        }
    }

    /// Length of the chain.
    pub fn len(&self) -> usize {
        self.chain.len()
    }

    /// Is the pattern empty? (illegal, but we expose the predicate
    /// for completeness)
    pub fn is_empty(&self) -> bool {
        self.chain.is_empty()
    }
}

/// The default 15-pattern library covering ~80% of transformer/CNN
/// hot paths.
pub fn default_library() -> Vec<Pattern> {
    vec![
        // matmul + bias + activation family
        Pattern::new("matmul+bias", vec![OpKind::Matmul, OpKind::BiasAdd]),
        Pattern::new(
            "matmul+bias+relu",
            vec![OpKind::Matmul, OpKind::BiasAdd, OpKind::Relu],
        ),
        Pattern::new(
            "matmul+bias+gelu",
            vec![OpKind::Matmul, OpKind::BiasAdd, OpKind::Gelu],
        ),
        Pattern::new(
            "matmul+bias+silu",
            vec![OpKind::Matmul, OpKind::BiasAdd, OpKind::Silu],
        ),
        Pattern::new("linear+relu", vec![OpKind::Linear, OpKind::Relu]),
        // norm + linear family
        Pattern::new("layernorm+linear", vec![OpKind::LayerNorm, OpKind::Linear]),
        Pattern::new("rmsnorm+linear", vec![OpKind::RmsNorm, OpKind::Linear]),
        // residual / elementwise
        Pattern::new("residual_add", vec![OpKind::Add]),
        Pattern::new("add+mul", vec![OpKind::Add, OpKind::Mul]),
        Pattern::new(
            "add+mul+sigmoid",
            vec![OpKind::Add, OpKind::Mul, OpKind::Sigmoid],
        ),
        Pattern::new("sub+mul", vec![OpKind::Sub, OpKind::Mul]),
        Pattern::new("mul+tanh", vec![OpKind::Mul, OpKind::Tanh]),
        Pattern::new("div+add+relu", vec![OpKind::Div, OpKind::Add, OpKind::Relu]),
        Pattern::new(
            "matmul+bias+silu+mul",
            vec![OpKind::Matmul, OpKind::BiasAdd, OpKind::Silu, OpKind::Mul],
        ),
        Pattern::new(
            "layernorm+matmul+bias",
            vec![OpKind::LayerNorm, OpKind::Matmul, OpKind::BiasAdd],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_round_trips_via_debug() {
        let p = Pattern::new(
            "matmul+bias+relu",
            vec![OpKind::Matmul, OpKind::BiasAdd, OpKind::Relu],
        );
        assert_eq!(p.len(), 3);
        assert!(!p.is_empty());
        // Debug-print should work.
        let s = format!("{:?}", p);
        assert!(s.contains("matmul"));
    }

    #[test]
    fn default_library_has_15_patterns() {
        let lib = default_library();
        assert_eq!(lib.len(), 15);
        // No two patterns should share a label.
        let mut labels: Vec<&str> = lib.iter().map(|p| p.label.as_str()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), 15);
    }

    #[test]
    fn opkind_any_serves_as_wildcard_in_future_extension() {
        // Sanity: Any variant exists and is hashable / comparable.
        let any = OpKind::Any;
        assert_eq!(any, OpKind::Any);
        assert_ne!(any, OpKind::Matmul);
    }
}

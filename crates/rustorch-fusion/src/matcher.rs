//! DAG matcher engine — walks an op trace and finds chain matches
//! against a registered Pattern list.
//!
//! Trace representation: `Vec<TraceNode>` where each node has an
//! `op_kind` and a list of input node indices. The matcher scans
//! linearly, attempting each pattern at each node, emitting `Match`
//! records for hits.

use crate::pattern::{OpKind, Pattern};

/// One op in the trace.
#[derive(Debug, Clone)]
pub struct TraceNode {
    /// Op kind (matmul, bias, relu, ...).
    pub op_kind: OpKind,
    /// Indices of upstream nodes this op consumes. The first input
    /// is treated as the "primary" (chain-following) input.
    pub inputs: Vec<usize>,
}

impl TraceNode {
    /// Build a node from kind + primary input(s).
    pub fn new(op_kind: OpKind, inputs: Vec<usize>) -> Self {
        Self { op_kind, inputs }
    }
}

/// A matched chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match {
    /// Pattern that matched.
    pub pattern_label: String,
    /// Trace node indices in chain order (root first).
    pub node_indices: Vec<usize>,
}

/// Scan `trace` and emit one `Match` per pattern hit. Matches that
/// overlap with a previously-emitted longer match are suppressed
/// (longest-chain-first preference).
pub fn find_matches(trace: &[TraceNode], patterns: &[Pattern]) -> Vec<Match> {
    if trace.is_empty() || patterns.is_empty() {
        return Vec::new();
    }
    let mut matches: Vec<Match> = Vec::new();
    let mut covered: Vec<bool> = vec![false; trace.len()];
    // Sort patterns by length desc — longer matches win tie-breaks.
    let mut sorted_patterns: Vec<&Pattern> = patterns.iter().collect();
    sorted_patterns.sort_by_key(|b| std::cmp::Reverse(b.len()));
    for pattern in sorted_patterns {
        if pattern.is_empty() {
            continue;
        }
        for start in 0..trace.len() {
            if covered[start] {
                continue;
            }
            if let Some(chain) = try_match_chain(trace, start, &pattern.chain) {
                if chain.iter().any(|&i| covered[i]) {
                    continue; // overlap with longer match
                }
                for &i in &chain {
                    covered[i] = true;
                }
                matches.push(Match {
                    pattern_label: pattern.label.clone(),
                    node_indices: chain,
                });
            }
        }
    }
    // Sort matches by first node index for deterministic output.
    matches.sort_by_key(|m| m.node_indices[0]);
    matches
}

/// Try to match `chain` starting at `start`. Returns the chain of
/// trace indices on success, `None` on failure.
fn try_match_chain(trace: &[TraceNode], start: usize, chain: &[OpKind]) -> Option<Vec<usize>> {
    let mut indices: Vec<usize> = Vec::with_capacity(chain.len());
    let mut current = start;
    for (depth, expected) in chain.iter().enumerate() {
        let node = trace.get(current)?;
        if !op_kind_matches(node.op_kind, *expected) {
            return None;
        }
        indices.push(current);
        if depth + 1 == chain.len() {
            // Last link — done.
            return Some(indices);
        }
        // Find the next chain link: a successor whose first input is
        // the current node.
        let next = trace
            .iter()
            .enumerate()
            .find(|(_, n)| n.inputs.first() == Some(&current))?;
        current = next.0;
    }
    Some(indices)
}

#[inline]
fn op_kind_matches(actual: OpKind, expected: OpKind) -> bool {
    expected == OpKind::Any || actual == expected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::default_library;

    fn trace_3node_chain() -> Vec<TraceNode> {
        vec![
            TraceNode::new(OpKind::Matmul, vec![]),
            TraceNode::new(OpKind::BiasAdd, vec![0]),
            TraceNode::new(OpKind::Relu, vec![1]),
        ]
    }

    #[test]
    fn matches_matmul_bias_relu_3node_chain() {
        let trace = trace_3node_chain();
        let matches = find_matches(&trace, &default_library());
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_label, "matmul+bias+relu");
        assert_eq!(matches[0].node_indices, vec![0, 1, 2]);
    }

    #[test]
    fn empty_trace_returns_empty_matches() {
        let trace: Vec<TraceNode> = Vec::new();
        let matches = find_matches(&trace, &default_library());
        assert!(matches.is_empty());
    }

    #[test]
    fn empty_patterns_returns_empty_matches() {
        let trace = trace_3node_chain();
        let matches = find_matches(&trace, &[]);
        assert!(matches.is_empty());
    }

    #[test]
    fn longer_pattern_wins_over_shorter_overlap() {
        // matmul+bias+relu (length 3) should match instead of
        // matmul+bias (length 2) on the 3-node chain.
        let trace = trace_3node_chain();
        let matches = find_matches(&trace, &default_library());
        assert_eq!(matches.len(), 1);
        assert!(matches[0].pattern_label.contains("relu"));
    }

    #[test]
    fn no_match_when_chain_is_broken() {
        // matmul → relu (skipping bias) — won't match matmul+bias+relu
        // and won't match matmul+bias either.
        let trace = vec![
            TraceNode::new(OpKind::Matmul, vec![]),
            TraceNode::new(OpKind::Relu, vec![0]),
        ];
        let matches = find_matches(&trace, &default_library());
        // `linear+relu` doesn't match (matmul ≠ linear). matmul+bias
        // doesn't match (relu ≠ bias). No matches.
        assert!(matches.is_empty(), "got {:?}", matches);
    }

    #[test]
    fn matches_residual_add_pattern() {
        let trace = vec![
            TraceNode::new(OpKind::Matmul, vec![]),
            TraceNode::new(OpKind::Add, vec![0]),
        ];
        let matches = find_matches(&trace, &default_library());
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_label, "residual_add");
    }

    #[test]
    fn dag_dependencies_respected() {
        // Match must not return matches whose nodes violate the
        // dependency order (produces before consumes).
        let trace = trace_3node_chain();
        let matches = find_matches(&trace, &default_library());
        for m in &matches {
            for win in m.node_indices.windows(2) {
                let producer = win[0];
                let consumer = win[1];
                // Consumer must depend (transitively or directly) on
                // producer. For our linear chain, the consumer's
                // primary input must be the producer.
                assert_eq!(
                    trace[consumer].inputs.first().copied(),
                    Some(producer),
                    "consumer {consumer} doesn't take producer {producer} as input"
                );
            }
        }
    }
}

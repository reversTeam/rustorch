//! Range generators — sugar for the doc v0.7.2 examples
//! `logspace(1e-5, 1e-2, 6)` and `linspace(0.1, 0.9, 9)`.
//!
//! Returns plain `Vec<f64>` rather than the more elaborate iterator
//! style of numpy because the values land in JSON arrays anyway.

/// Linear space — `n` points evenly spaced between `start` and `end`
/// inclusive. `n == 0` returns `[]`; `n == 1` returns `[start]` so
/// the function is total over the natural-numbers domain.
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

/// Logarithmic space — `n` points geometrically spaced between
/// `start` and `end` inclusive. Both endpoints must be positive.
/// Falls back to `linspace` if either is non-positive (returning the
/// linear schedule rather than NaNs).
pub fn logspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if start <= 0.0 || end <= 0.0 {
        return linspace(start, end, n);
    }
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let log_start = start.ln();
    let log_end = end.ln();
    let step = (log_end - log_start) / (n - 1) as f64;
    (0..n)
        .map(|i| (log_start + step * i as f64).exp())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn linspace_endpoints_and_step() {
        let v = linspace(0.0, 1.0, 5);
        assert_eq!(v.len(), 5);
        assert!(approx_eq(v[0], 0.0));
        assert!(approx_eq(v[4], 1.0));
        assert!(approx_eq(v[2], 0.5));
    }

    #[test]
    fn linspace_edge_cases() {
        assert!(linspace(0.0, 1.0, 0).is_empty());
        assert_eq!(linspace(7.0, 9.0, 1), vec![7.0]);
    }

    #[test]
    fn logspace_six_points_match_numpy() {
        // np.logspace via numpy:
        //   logspace(log10(1e-5), log10(1e-2), 6) → [1e-5, 1e-4.4, ..., 1e-2]
        // Our function takes (start, end), not log endpoints.
        let v = logspace(1e-5, 1e-2, 6);
        assert_eq!(v.len(), 6);
        assert!(approx_eq(v[0], 1e-5));
        assert!(approx_eq(v[5], 1e-2));
        // Each step multiplies by the same factor.
        let r1 = v[1] / v[0];
        let r2 = v[2] / v[1];
        assert!((r1 - r2).abs() / r1 < 1e-6);
    }

    #[test]
    fn logspace_falls_back_on_zero() {
        let v = logspace(0.0, 1.0, 3);
        assert_eq!(v, linspace(0.0, 1.0, 3));
    }
}

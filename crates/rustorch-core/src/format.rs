//! PyTorch-style `Display` and `Debug` for `Tensor`.

use std::fmt;

use crate::Tensor;

/// Above this element count we elide with "..." per axis to keep output
/// readable. Same threshold as PyTorch's default.
const TRUNCATE_THRESHOLD: usize = 1000;

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor(")?;
        format_axis(
            f,
            self.shape(),
            self.data(),
            0,
            self.len() > TRUNCATE_THRESHOLD,
        )?;
        write!(f, ")")
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Same formatter as Display, plus shape footer.
        write!(f, "tensor(")?;
        format_axis(
            f,
            self.shape(),
            self.data(),
            0,
            self.len() > TRUNCATE_THRESHOLD,
        )?;
        write!(f, ", shape={:?})", self.shape())
    }
}

/// Recursive axis printer.
fn format_axis(
    f: &mut fmt::Formatter<'_>,
    shape: &[usize],
    data: &[f32],
    axis: usize,
    truncate: bool,
) -> fmt::Result {
    // Scalar / 0-d
    if shape.is_empty() {
        return format_scalar(f, data[0]);
    }
    // 1-D leaf
    if shape.len() == 1 {
        let n = shape[0];
        write!(f, "[")?;
        if truncate && n > 6 {
            for (idx, v) in data.iter().take(3).enumerate() {
                if idx > 0 {
                    write!(f, ", ")?;
                }
                format_scalar(f, *v)?;
            }
            write!(f, ", ..., ")?;
            for (idx, v) in data.iter().skip(n - 3).take(3).enumerate() {
                if idx > 0 {
                    write!(f, ", ")?;
                }
                format_scalar(f, *v)?;
            }
        } else {
            for (idx, v) in data.iter().enumerate() {
                if idx > 0 {
                    write!(f, ", ")?;
                }
                format_scalar(f, *v)?;
            }
        }
        write!(f, "]")?;
        return Ok(());
    }
    // N-D recursive
    let stride: usize = shape[1..].iter().product();
    write!(f, "[")?;
    let n = shape[0];
    let visible: Vec<usize> = if truncate && n > 6 {
        (0..3).chain((n - 3)..n).collect()
    } else {
        (0..n).collect()
    };
    let mut prev = None::<usize>;
    for i in &visible {
        if let Some(p) = prev {
            write!(f, ",")?;
            // Visible gap = elision marker
            if *i != p + 1 {
                writeln!(f)?;
                indent(f, axis + 1)?;
                writeln!(f, "...,")?;
                indent(f, axis + 1)?;
            } else {
                writeln!(f)?;
                indent(f, axis + 1)?;
            }
        }
        let chunk = &data[i * stride..(i + 1) * stride];
        format_axis(f, &shape[1..], chunk, axis + 1, truncate)?;
        prev = Some(*i);
    }
    write!(f, "]")
}

/// Pretty-print a single f32 (PyTorch-style: `1.` for integers, `1e-7`
/// for small values, `nan` / `inf` for special).
fn format_scalar(f: &mut fmt::Formatter<'_>, v: f32) -> fmt::Result {
    if v.is_nan() {
        return write!(f, "nan");
    }
    if v.is_infinite() {
        return write!(f, "{}", if v > 0.0 { "inf" } else { "-inf" });
    }
    let abs = v.abs();
    if abs != 0.0 && !(1e-4..1e6).contains(&abs) {
        // Scientific
        write!(f, "{:.4e}", v)
    } else if v == v.trunc() {
        // Whole number → trailing dot
        write!(f, "{}.", v as i64)
    } else {
        write!(f, "{:.4}", v)
    }
}

fn indent(f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
    for _ in 0..(depth + "tensor(".len()) {
        write!(f, " ")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_display() {
        let t = Tensor::scalar(2.71);
        assert_eq!(format!("{}", t), "tensor(2.7100)");
    }

    #[test]
    fn matrix_truncates_when_huge() {
        // Build a tensor with 1001 elements so the truncation path runs.
        let n = 1001usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t = Tensor::from_vec([n], data).unwrap();
        let s = format!("{}", t);
        assert!(s.contains("..."), "expected elision in {s}");
    }

    #[test]
    fn vector_display_integers() {
        let t = Tensor::from_vec([3usize], vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(format!("{}", t), "tensor([1., 2., 3.])");
    }

    #[test]
    fn matrix_display() {
        let t = Tensor::from_vec([2usize, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let s = format!("{}", t);
        assert!(s.starts_with("tensor([[1., 2.],"));
        assert!(s.contains("[3., 4.]]"));
    }

    #[test]
    fn empty_display() {
        let t = Tensor::zeros([0usize]);
        assert_eq!(format!("{}", t), "tensor([])");
    }

    #[test]
    fn nan_display() {
        let t = Tensor::from_vec([1usize], vec![f32::NAN]).unwrap();
        assert_eq!(format!("{}", t), "tensor([nan])");
    }

    #[test]
    fn neg_inf_display() {
        let t = Tensor::from_vec([1usize], vec![f32::NEG_INFINITY]).unwrap();
        assert_eq!(format!("{}", t), "tensor([-inf])");
    }

    #[test]
    fn small_float_scientific() {
        let t = Tensor::from_vec([1usize], vec![1e-7]).unwrap();
        let s = format!("{}", t);
        // Either 1.0000e-7 or similar — at minimum, no trailing dot
        assert!(s.contains("e"), "expected scientific, got {s}");
    }

    #[test]
    fn debug_includes_shape() {
        let t = Tensor::from_vec([2usize, 3], vec![0.0; 6]).unwrap();
        let s = format!("{:?}", t);
        assert!(s.contains("shape=[2, 3]"), "got {s}");
    }
}

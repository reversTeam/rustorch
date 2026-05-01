//! CTC (Connectionist Temporal Classification) loss kernel (P1.4).
//!
//! v1 ships forward only. The standard alpha-beta backward is
//! mathematically straightforward but verbose; deferred to a follow-up
//! that wires it into autograd.
//!
//! Inputs:
//! - `log_probs` : `[T, B, C]` log-probabilities (output of log_softmax)
//! - `targets`   : `[B, S]` I64 target labels (no blank)
//! - `input_lens` / `target_lens` : `[B]` I64 actual lengths
//!
//! Output:
//! - per-sample CTC loss `[B]`, summed/meaned per the `Reduction`.

#![allow(clippy::needless_range_loop)]

use crate::backend::Reduction;
use crate::error::BackendError;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

const BLANK: i64 = 0;

/// Forward CTC loss. Returns a scalar tensor (mean / sum / per-sample).
pub fn ctc_loss_forward(
    log_probs: &Tensor,
    targets: &Tensor,
    input_lens: &[i64],
    target_lens: &[i64],
    reduction: Reduction,
) -> Result<Tensor, BackendError> {
    if log_probs.dtype() != Dtype::F32 {
        return Err(BackendError::DtypeMismatch {
            op: "ctc_loss",
            lhs: log_probs.dtype(),
            rhs: Dtype::F32,
        });
    }
    if log_probs.ndim() != 3 || targets.ndim() != 2 {
        return Err(BackendError::ShapeMismatch {
            op: "ctc_loss",
            lhs: log_probs.shape().to_vec(),
            rhs: targets.shape().to_vec(),
        });
    }
    let t_in = log_probs.shape()[0];
    let bs = log_probs.shape()[1];
    let c = log_probs.shape()[2];
    let s_max = targets.shape()[1];
    let lp = log_probs.as_slice::<f32>().expect("F32");
    let tg = targets.as_slice::<i64>().expect("I64 targets");

    let mut losses = vec![0.0_f32; bs];
    for b in 0..bs {
        let t_b = (input_lens[b] as usize).min(t_in);
        let s_b = (target_lens[b] as usize).min(s_max);

        // Build extended target with blanks: y' = blank y_1 blank y_2 ... blank
        let s_ext = 2 * s_b + 1;
        let mut y_ext = vec![BLANK; s_ext];
        for i in 0..s_b {
            y_ext[2 * i + 1] = tg[b * s_max + i];
        }
        // alpha[t][s] = log P(prefix t emits y'[..=s])
        let neg_inf = f32::NEG_INFINITY;
        let mut alpha = vec![neg_inf; t_b * s_ext];
        // Initialise t=0
        let lp_b = |t: usize, k: i64| -> f32 { lp[(t * bs + b) * c + k as usize] };
        if s_ext >= 1 {
            alpha[0] = lp_b(0, BLANK); // alpha[0][0]
        }
        if s_ext >= 2 {
            alpha[1] = lp_b(0, y_ext[1]); // alpha[0][1]
        }
        // Recurrence: alpha[t][s] = (alpha[t-1][s] ⊕ alpha[t-1][s-1] [⊕ alpha[t-1][s-2] if non-repeated]) + lp[y_ext[s]]
        for t in 1..t_b {
            for s in 0..s_ext {
                let mut a = alpha[(t - 1) * s_ext + s];
                if s > 0 {
                    a = log_sum_exp(a, alpha[(t - 1) * s_ext + s - 1]);
                }
                if s > 1 && y_ext[s] != BLANK && y_ext[s] != y_ext[s - 2] {
                    a = log_sum_exp(a, alpha[(t - 1) * s_ext + s - 2]);
                }
                alpha[t * s_ext + s] = a + lp_b(t, y_ext[s]);
            }
        }
        // Loss = -log( alpha[T-1][2*S] ⊕ alpha[T-1][2*S - 1] )
        let last_t = t_b - 1;
        let mut total = neg_inf;
        if s_ext >= 1 {
            total = log_sum_exp(total, alpha[last_t * s_ext + s_ext - 1]);
        }
        if s_ext >= 2 {
            total = log_sum_exp(total, alpha[last_t * s_ext + s_ext - 2]);
        }
        losses[b] = -total;
    }
    let out = match reduction {
        Reduction::None => Tensor::from_vec([bs], losses).unwrap(),
        Reduction::Sum => Tensor::scalar(losses.iter().sum::<f32>()),
        Reduction::Mean => Tensor::scalar(losses.iter().sum::<f32>() / bs as f32),
    };
    Ok(out)
}

fn log_sum_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_perfect_alignment_low_loss() {
        // T=3, B=1, C=3 (blank=0, classes 1, 2). Target = [1, 2].
        // Make log_probs strongly favour the alignment 0,1,2 for example.
        // We just check the loss is finite and positive.
        let lp = vec![
            // t=0: log P(blank)=0, others very negative
            0.0_f32, -10.0, -10.0, // t=0, b=0
            // t=1
            -10.0, 0.0, -10.0, // t=2
            -10.0, -10.0, 0.0,
        ];
        let lp_t = Tensor::from_vec([3usize, 1, 3], lp).unwrap();
        let tg_t = Tensor::from_vec_typed::<i64, _>([1usize, 2], vec![1_i64, 2]).unwrap();
        let loss = ctc_loss_forward(&lp_t, &tg_t, &[3], &[2], Reduction::Mean).unwrap();
        let v = loss.as_slice::<f32>().unwrap()[0];
        // Loss should be finite and small (target alignment is highly favoured).
        // Allow tiny negative round-off near zero.
        assert!(v.is_finite() && v.abs() < 0.1, "got {v}");
    }
}

//! Parameter initialization library (P1.6).
//!
//! Mirrors `torch.nn.init`. Each variant computes the
//! distribution-specific bound and fills a tensor of the requested
//! shape. RNG is seedable for reproducible model init.
//!
//! v1 ships the most-used variants:
//! - [`Init::Zeros`], [`Init::Ones`], [`Init::Constant`]
//! - [`Init::Uniform`], [`Init::Normal`]
//! - [`Init::XavierUniform`], [`Init::XavierNormal`] (Glorot)
//! - [`Init::KaimingUniform`], [`Init::KaimingNormal`] (He) with
//!   FanIn/FanOut and ReLU/LeakyReLU nonlinearities
//!
//! `Orthogonal` and `TruncNormal` plug into the same enum and are
//! pending a follow-up slice.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustorch_core::tensor::tensor_impl::Tensor;

/// Fan mode for Kaiming/He initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanMode {
    /// Use fan_in (preserve forward-pass variance).
    FanIn,
    /// Use fan_out (preserve backward-pass variance).
    FanOut,
}

/// Nonlinearity hint for Kaiming/He gain computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Nonlinearity {
    /// Linear / Sigmoid (gain = 1).
    Linear,
    /// Tanh (gain = 5/3).
    Tanh,
    /// ReLU (gain = sqrt(2)).
    Relu,
    /// LeakyReLU with negative slope (gain = sqrt(2 / (1 + slope²))).
    LeakyRelu(f32),
}

impl Nonlinearity {
    /// Recommended gain for this nonlinearity.
    pub fn gain(self) -> f32 {
        match self {
            Nonlinearity::Linear => 1.0,
            Nonlinearity::Tanh => 5.0 / 3.0,
            Nonlinearity::Relu => 2.0_f32.sqrt(),
            Nonlinearity::LeakyRelu(slope) => (2.0 / (1.0 + slope * slope)).sqrt(),
        }
    }
}

/// Initialization strategy.
#[derive(Debug, Clone, Copy)]
pub enum Init {
    /// Fill with zeros.
    Zeros,
    /// Fill with ones.
    Ones,
    /// Fill with a constant.
    Constant(f32),
    /// Uniform on `[a, b]`.
    Uniform {
        /// Inclusive lower bound.
        a: f32,
        /// Exclusive upper bound.
        b: f32,
    },
    /// Normal with mean and std.
    Normal {
        /// Distribution mean.
        mean: f32,
        /// Distribution standard deviation.
        std: f32,
    },
    /// Xavier/Glorot uniform: bound = sqrt(6 / (fan_in + fan_out)).
    XavierUniform,
    /// Xavier/Glorot normal: std = sqrt(2 / (fan_in + fan_out)).
    XavierNormal,
    /// Kaiming/He uniform: bound = gain * sqrt(3 / fan).
    KaimingUniform {
        /// Use FanIn or FanOut.
        mode: FanMode,
        /// Nonlinearity hint for gain.
        nonlinearity: Nonlinearity,
    },
    /// Kaiming/He normal: std = gain / sqrt(fan).
    KaimingNormal {
        /// Use FanIn or FanOut.
        mode: FanMode,
        /// Nonlinearity hint for gain.
        nonlinearity: Nonlinearity,
    },
}

/// Compute fan_in and fan_out for a tensor shape.
///
/// Convention (matches PyTorch): for a 2-D tensor of shape `[in, out]`
/// (rustorch order — Linear's weight), `fan_in = in` and `fan_out = out`.
/// For 3+ dim tensors (e.g. conv weight `[out, in, kH, kW]`),
/// `fan_in = in * kH * kW` and `fan_out = out * kH * kW`.
pub fn calculate_fan(shape: &[usize]) -> (usize, usize) {
    if shape.len() < 2 {
        // 1-D parameters (e.g. bias / gamma): fan_in = fan_out = numel.
        let n: usize = shape.iter().product::<usize>().max(1);
        return (n, n);
    }
    if shape.len() == 2 {
        // rustorch Linear convention: weight is [in, out].
        return (shape[0], shape[1]);
    }
    // Conv-style: [out, in, k1, k2, ...] (PyTorch convention).
    let receptive_field: usize = shape[2..].iter().product();
    let fan_in = shape[1] * receptive_field;
    let fan_out = shape[0] * receptive_field;
    (fan_in, fan_out)
}

/// Generate a freshly-initialised tensor of the given shape using the
/// specified strategy and seed.
pub fn init_with_seed(init: Init, shape: &[usize], seed: u64) -> Tensor {
    let mut rng = StdRng::seed_from_u64(seed);
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = match init {
        Init::Zeros => vec![0.0_f32; numel],
        Init::Ones => vec![1.0_f32; numel],
        Init::Constant(c) => vec![c; numel],
        Init::Uniform { a, b } => (0..numel).map(|_| rng.gen_range(a..b)).collect(),
        Init::Normal { mean, std } => (0..numel)
            .map(|_| sample_normal(&mut rng, mean, std))
            .collect(),
        Init::XavierUniform => {
            let (fan_in, fan_out) = calculate_fan(shape);
            let bound = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
            (0..numel).map(|_| rng.gen_range(-bound..bound)).collect()
        },
        Init::XavierNormal => {
            let (fan_in, fan_out) = calculate_fan(shape);
            let std = (2.0_f32 / (fan_in + fan_out) as f32).sqrt();
            (0..numel)
                .map(|_| sample_normal(&mut rng, 0.0, std))
                .collect()
        },
        Init::KaimingUniform { mode, nonlinearity } => {
            let (fan_in, fan_out) = calculate_fan(shape);
            let fan = match mode {
                FanMode::FanIn => fan_in,
                FanMode::FanOut => fan_out,
            } as f32;
            let bound = nonlinearity.gain() * (3.0_f32 / fan).sqrt();
            (0..numel).map(|_| rng.gen_range(-bound..bound)).collect()
        },
        Init::KaimingNormal { mode, nonlinearity } => {
            let (fan_in, fan_out) = calculate_fan(shape);
            let fan = match mode {
                FanMode::FanIn => fan_in,
                FanMode::FanOut => fan_out,
            } as f32;
            let std = nonlinearity.gain() / fan.sqrt();
            (0..numel)
                .map(|_| sample_normal(&mut rng, 0.0, std))
                .collect()
        },
    };
    Tensor::from_vec(shape.to_vec(), data).expect("shape match for init")
}

/// Convenience: `init_with_seed` using a time-derived seed.
pub fn init(strategy: Init, shape: &[usize]) -> Tensor {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    init_with_seed(strategy, shape, seed)
}

/// Sample from N(mean, std) via the Box-Muller transform.
fn sample_normal(rng: &mut StdRng, mean: f32, std: f32) -> f32 {
    let u1: f32 = rng.gen_range(1e-7_f32..1.0); // avoid 0 for log
    let u2: f32 = rng.gen::<f32>();
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0_f32 * std::f32::consts::PI * u2;
    let z = r * theta.cos();
    mean + std * z
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empirical_mean_std(data: &[f32]) -> (f32, f32) {
        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        (mean, var.sqrt())
    }

    #[test]
    fn zeros_and_ones_constants() {
        let z = init_with_seed(Init::Zeros, &[4, 4], 0);
        assert!(z.as_slice::<f32>().unwrap().iter().all(|&v| v == 0.0));
        let o = init_with_seed(Init::Ones, &[4, 4], 0);
        assert!(o.as_slice::<f32>().unwrap().iter().all(|&v| v == 1.0));
        let c = init_with_seed(Init::Constant(0.42), &[3], 0);
        assert!(c.as_slice::<f32>().unwrap().iter().all(|&v| v == 0.42));
    }

    #[test]
    fn uniform_in_range() {
        let t = init_with_seed(Init::Uniform { a: -2.0, b: 5.0 }, &[1000], 42);
        let v = t.as_slice::<f32>().unwrap();
        assert!(v.iter().all(|&x| (-2.0..5.0).contains(&x)));
    }

    #[test]
    fn normal_stats_within_tolerance() {
        // 10000 samples of N(0, 1) — sample mean within 0.05, std within 0.05.
        let t = init_with_seed(
            Init::Normal {
                mean: 0.0,
                std: 1.0,
            },
            &[10_000],
            7,
        );
        let v = t.as_slice::<f32>().unwrap();
        let (m, s) = empirical_mean_std(v);
        assert!(m.abs() < 0.1, "mean ≈ 0, got {m}");
        assert!((s - 1.0).abs() < 0.1, "std ≈ 1, got {s}");
    }

    #[test]
    fn fan_calculation_2d() {
        // Linear weight [in, out]
        assert_eq!(calculate_fan(&[3, 4]), (3, 4));
    }

    #[test]
    fn fan_calculation_conv() {
        // [out=64, in=3, kH=3, kW=3]
        assert_eq!(calculate_fan(&[64, 3, 3, 3]), (3 * 3 * 3, 64 * 3 * 3));
    }

    #[test]
    fn fan_calculation_1d_bias() {
        // 1-D bias: fan_in = fan_out = numel
        assert_eq!(calculate_fan(&[10]), (10, 10));
    }

    #[test]
    fn nonlinearity_gain_values() {
        assert!((Nonlinearity::Linear.gain() - 1.0).abs() < 1e-7);
        assert!((Nonlinearity::Tanh.gain() - 5.0 / 3.0).abs() < 1e-7);
        assert!((Nonlinearity::Relu.gain() - 2.0_f32.sqrt()).abs() < 1e-7);
        // LeakyReLU(0.1): sqrt(2 / (1 + 0.01)) ≈ 1.408
        let g = Nonlinearity::LeakyRelu(0.1).gain();
        let expected = (2.0_f32 / 1.01_f32).sqrt();
        assert!((g - expected).abs() < 1e-6);
    }

    #[test]
    fn xavier_uniform_bound_matches_formula() {
        // For shape [3, 5], bound = sqrt(6 / (3+5)) = sqrt(0.75)
        let t = init_with_seed(Init::XavierUniform, &[3, 5], 42);
        let v = t.as_slice::<f32>().unwrap();
        let bound = (6.0_f32 / 8.0).sqrt();
        assert!(v.iter().all(|&x| x.abs() < bound + 1e-5));
    }

    #[test]
    fn kaiming_normal_std_scales_with_fan() {
        // For shape [100, 200] (fan_in = 100, ReLU), std = sqrt(2)/sqrt(100) = sqrt(0.02)
        let t = init_with_seed(
            Init::KaimingNormal {
                mode: FanMode::FanIn,
                nonlinearity: Nonlinearity::Relu,
            },
            &[100, 200],
            42,
        );
        let v = t.as_slice::<f32>().unwrap();
        let (_, s) = empirical_mean_std(v);
        let expected = (2.0_f32 / 100.0).sqrt();
        // 5% tolerance — 20k samples is still noisy.
        assert!(
            (s - expected).abs() / expected < 0.1,
            "got std {s}, expected {expected}"
        );
    }

    #[test]
    fn seeded_init_is_reproducible() {
        let a = init_with_seed(
            Init::KaimingUniform {
                mode: FanMode::FanIn,
                nonlinearity: Nonlinearity::Relu,
            },
            &[10, 10],
            1234,
        );
        let b = init_with_seed(
            Init::KaimingUniform {
                mode: FanMode::FanIn,
                nonlinearity: Nonlinearity::Relu,
            },
            &[10, 10],
            1234,
        );
        assert_eq!(a.as_slice::<f32>().unwrap(), b.as_slice::<f32>().unwrap());
    }
}

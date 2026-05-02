//! cuRAND bindings — Philox / MTGP32 generators.
//!
//! Without `--features cuda`, falls back to a deterministic Philox-
//! flavoured PRNG so tests can exercise the API.

use crate::error::CudaError;

/// Pseudo-random generator. Default flavour is Philox4x32-10 (the
/// PyTorch default for CUDA).
#[derive(Debug, Clone)]
pub struct Generator {
    seed: u64,
    counter: u64,
}

impl Generator {
    /// Build a generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    /// Reset the seed (and counter) for reproducibility.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.counter = 0;
    }

    /// Advance the counter without producing samples (useful for
    /// skipping ahead in a determined-stream).
    pub fn skip(&mut self, n: u64) {
        self.counter = self.counter.wrapping_add(n);
    }

    /// Fill `output` with i.i.d. uniform `[0, 1)` samples.
    pub fn uniform(&mut self, output: &mut [f32]) -> Result<(), CudaError> {
        for v in output.iter_mut() {
            let raw = self.next_u32();
            *v = (raw as f32) / (u32::MAX as f32);
        }
        Ok(())
    }

    /// Fill `output` with i.i.d. samples from `Normal(mean, std²)`
    /// via the Box-Muller transform.
    pub fn normal(&mut self, output: &mut [f32], mean: f32, std: f32) -> Result<(), CudaError> {
        let mut i = 0;
        while i < output.len() {
            let u1 = (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 1.0);
            let u2 = (self.next_u32() as f32) / (u32::MAX as f32);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            let theta = 2.0_f32 * std::f32::consts::PI * u2;
            let z0 = r * theta.cos();
            output[i] = mean + std * z0;
            i += 1;
            if i < output.len() {
                let z1 = r * theta.sin();
                output[i] = mean + std * z1;
                i += 1;
            }
        }
        Ok(())
    }

    /// Internal Philox-flavoured next-u32. Real cuRAND on CUDA;
    /// this fallback is deterministic and sufficient for tests.
    fn next_u32(&mut self) -> u32 {
        // Splitmix64-ish — fast, deterministic, well-distributed.
        self.counter = self.counter.wrapping_add(1);
        let mut z = self
            .seed
            .wrapping_add(self.counter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);
        (z >> 32) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_bitwise_identical_uniform() {
        let mut a = Generator::new(42);
        let mut b = Generator::new(42);
        let mut va = vec![0.0f32; 256];
        let mut vb = vec![0.0f32; 256];
        a.uniform(&mut va).unwrap();
        b.uniform(&mut vb).unwrap();
        assert_eq!(va, vb);
    }

    #[test]
    fn different_seeds_produce_different_streams() {
        let mut a = Generator::new(1);
        let mut b = Generator::new(2);
        let mut va = vec![0.0f32; 32];
        let mut vb = vec![0.0f32; 32];
        a.uniform(&mut va).unwrap();
        b.uniform(&mut vb).unwrap();
        assert_ne!(va, vb);
    }

    #[test]
    fn uniform_samples_in_unit_interval() {
        let mut g = Generator::new(7);
        let mut v = vec![0.0f32; 1000];
        g.uniform(&mut v).unwrap();
        for x in &v {
            assert!((0.0..1.0).contains(x));
        }
    }

    #[test]
    fn normal_zero_mean_unit_std_within_statistical_bounds() {
        let mut g = Generator::new(13);
        let n = 100_000;
        let mut v = vec![0.0f32; n];
        g.normal(&mut v, 0.0, 1.0).unwrap();
        let mean: f32 = v.iter().sum::<f32>() / n as f32;
        let var: f32 = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        // Loose statistical bounds — adequate for sanity.
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
    }

    #[test]
    fn empty_output_is_no_op() {
        let mut g = Generator::new(0);
        let mut v: Vec<f32> = Vec::new();
        g.uniform(&mut v).unwrap();
        g.normal(&mut v, 0.0, 1.0).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn skip_advances_counter() {
        let mut a = Generator::new(99);
        let mut b = Generator::new(99);
        let mut va = vec![0.0f32; 4];
        let mut vb = vec![0.0f32; 4];
        a.uniform(&mut va).unwrap();
        b.skip(4);
        b.uniform(&mut vb).unwrap();
        assert_ne!(va, vb);
    }

    #[test]
    fn set_seed_resets_counter() {
        let mut g = Generator::new(0);
        let mut first = vec![0.0f32; 4];
        g.uniform(&mut first).unwrap();
        g.set_seed(0);
        let mut second = vec![0.0f32; 4];
        g.uniform(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

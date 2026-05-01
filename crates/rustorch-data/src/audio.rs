//! Audio augmentations / spectrogram primitives (P1.9 minimal slice).
//!
//! v1 ships:
//! - [`Resample`] — linear-interpolation resampling between sample rates
//! - [`Spectrogram`] — magnitude spectrogram via real FFT (naive DFT v1)
//! - [`SpecAugment`] — time + frequency masking (Park et al. 2019)
//!
//! MelSpectrogram / MFCC pending follow-ups that need a mel filter
//! bank constructor.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustorch_core::tensor::tensor_impl::Tensor;

/// Linear-interpolation resampling. Audio is `[N, T]` F32; output is
/// `[N, T_out]` where `T_out = T * sr_out / sr_in`.
pub struct Resample {
    /// Source sample rate.
    pub sr_in: u32,
    /// Target sample rate.
    pub sr_out: u32,
}

impl Resample {
    /// Build with input/output sample rates.
    pub fn new(sr_in: u32, sr_out: u32) -> Self {
        Resample { sr_in, sr_out }
    }

    /// Apply: produces a new tensor with the resampled signal.
    pub fn apply(&self, x: &Tensor) -> Tensor {
        let s = x.shape().to_vec();
        if s.len() != 2 {
            panic!("Resample expects [N, T] input, got {s:?}");
        }
        let n = s[0];
        let t = s[1];
        let t_out = (t as f64 * self.sr_out as f64 / self.sr_in as f64) as usize;
        let data = x.as_slice::<f32>().expect("F32");
        let mut out = vec![0.0_f32; n * t_out];
        for ni in 0..n {
            for j in 0..t_out {
                let src_pos = j as f64 * self.sr_in as f64 / self.sr_out as f64;
                let i = src_pos.floor() as usize;
                let frac = (src_pos - i as f64) as f32;
                let i_next = (i + 1).min(t - 1);
                let v0 = data[ni * t + i];
                let v1 = data[ni * t + i_next];
                out[ni * t_out + j] = v0 * (1.0 - frac) + v1 * frac;
            }
        }
        Tensor::from_vec([n, t_out], out).expect("resample shape")
    }
}

/// Magnitude spectrogram via naive DFT (v1). Real FFT lands in a
/// follow-up that pulls in a small FFT crate.
pub struct Spectrogram {
    /// FFT window size.
    pub n_fft: usize,
    /// Hop length between successive frames.
    pub hop: usize,
}

impl Spectrogram {
    /// Build with window size and hop.
    pub fn new(n_fft: usize, hop: usize) -> Self {
        Spectrogram { n_fft, hop }
    }

    /// Apply: input `[T]` audio → `[F, T_out]` magnitude spectrogram
    /// with `F = n_fft/2 + 1`.
    pub fn apply(&self, x: &Tensor) -> Tensor {
        let s = x.shape().to_vec();
        if s.len() != 1 {
            panic!("Spectrogram expects 1-D audio, got {s:?}");
        }
        let t = s[0];
        let f_bins = self.n_fft / 2 + 1;
        let t_frames = if t < self.n_fft {
            0
        } else {
            (t - self.n_fft) / self.hop + 1
        };
        let data = x.as_slice::<f32>().expect("F32");
        let mut out = vec![0.0_f32; f_bins * t_frames];
        for frame in 0..t_frames {
            let start = frame * self.hop;
            for k in 0..f_bins {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for nn in 0..self.n_fft {
                    let theta = -2.0 * std::f64::consts::PI * (k as f64) * (nn as f64)
                        / (self.n_fft as f64);
                    let v = data[start + nn] as f64;
                    re += v * theta.cos();
                    im += v * theta.sin();
                }
                let mag = (re * re + im * im).sqrt() as f32;
                out[k * t_frames + frame] = mag;
            }
        }
        Tensor::from_vec([f_bins, t_frames], out).expect("spectrogram shape")
    }
}

/// SpecAugment — random time + frequency masking on a `[F, T]`
/// spectrogram. Replaces masked regions with zeros.
pub struct SpecAugment {
    /// Maximum time-mask width.
    pub time_mask: usize,
    /// Maximum frequency-mask width.
    pub freq_mask: usize,
    rng: StdRng,
}

impl SpecAugment {
    /// Build with the given mask widths.
    pub fn new(time_mask: usize, freq_mask: usize) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        SpecAugment {
            time_mask,
            freq_mask,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Reproducible build with a fixed seed.
    pub fn with_seed(time_mask: usize, freq_mask: usize, seed: u64) -> Self {
        SpecAugment {
            time_mask,
            freq_mask,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Apply masking in-place style (returns a new tensor).
    pub fn apply(&mut self, x: &Tensor) -> Tensor {
        let s = x.shape().to_vec();
        if s.len() != 2 {
            panic!("SpecAugment expects [F, T] input, got {s:?}");
        }
        let f = s[0];
        let t = s[1];
        let mut data: Vec<f32> = x.as_slice::<f32>().expect("F32").to_vec();
        // Frequency mask: pick a starting bin and width.
        if self.freq_mask > 0 && f > 0 {
            let width = self.rng.gen_range(0..=self.freq_mask.min(f));
            if width > 0 {
                let start = self.rng.gen_range(0..=f - width);
                for fi in start..start + width {
                    for ti in 0..t {
                        data[fi * t + ti] = 0.0;
                    }
                }
            }
        }
        // Time mask
        if self.time_mask > 0 && t > 0 {
            let width = self.rng.gen_range(0..=self.time_mask.min(t));
            if width > 0 {
                let start = self.rng.gen_range(0..=t - width);
                for fi in 0..f {
                    for ti in start..start + width {
                        data[fi * t + ti] = 0.0;
                    }
                }
            }
        }
        Tensor::from_vec([f, t], data).expect("specaug shape")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_doubles_samples() {
        // 1 batch of 4 samples; 16k → 32k → 8 output samples.
        let x = Tensor::from_vec([1usize, 4], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let r = Resample::new(16_000, 32_000);
        let y = r.apply(&x);
        assert_eq!(y.shape(), &[1, 8]);
    }

    #[test]
    fn spectrogram_shape_correct() {
        // T=8, n_fft=4, hop=2 → t_frames = (8-4)/2 + 1 = 3, f_bins = 3.
        let x = Tensor::from_vec([8usize], vec![1.0_f32; 8]).unwrap();
        let sp = Spectrogram::new(4, 2);
        let y = sp.apply(&x);
        assert_eq!(y.shape(), &[3, 3]);
    }

    #[test]
    fn spec_augment_zero_masks_no_change() {
        let x = Tensor::from_vec([4usize, 4], (0..16).map(|i| i as f32).collect()).unwrap();
        let mut aug = SpecAugment::with_seed(0, 0, 42);
        let y = aug.apply(&x);
        // Both mask widths are 0 → output unchanged.
        assert_eq!(y.as_slice::<f32>().unwrap(), x.as_slice::<f32>().unwrap());
    }

    #[test]
    fn spec_augment_with_masks_zero_some_cells() {
        let x = Tensor::from_vec([4usize, 4], vec![1.0_f32; 16]).unwrap();
        let mut aug = SpecAugment::with_seed(2, 2, 0xC0DE);
        let y = aug.apply(&x);
        let v = y.as_slice::<f32>().unwrap();
        // At least some cells should be zeroed.
        assert!(v.contains(&0.0));
    }
}

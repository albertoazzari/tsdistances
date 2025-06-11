use rustfft::{Fft, FftDirection, algorithm::Radix4, num_complex::Complex};

use crate::Float;

pub fn next_multiple_of_n(x: usize, n: usize) -> usize {
    (x + n - 1) / n * n
}

pub fn derivate(x: &Vec<Vec<Float>>) -> Vec<Vec<Float>> {
    let mut x_d = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        x_d.push(vec![0.0; x[i].len() - 2]);
    }
    for i in 0..x.len() {
        for j in 1..x[i].len() - 1 {
            x_d[i][j - 1] = ((x[i][j] - x[i][j - 1]) + (x[i][j + 1] - x[i][j - 1]) / 2.0) / 2.0;
        }
    }
    x_d
}

const WEIGHT_MAX: Float = 1.0;
pub fn dtw_weights(len: usize, g: Float) -> Vec<Float> {
    let mut weights = vec![0.0; len];
    let half_len = len as Float / 2.0;
    for i in 0..len {
        weights[i] =
            WEIGHT_MAX / (1.0 + (std::f64::consts::E as Float).powf(-g * (i as Float - half_len as Float)));
    }
    weights
}
// [1 / (1 + np.exp(-g * (i - max_size / 2))) for i in range(0, max_size)]

const MSM_C: Float = 1.0;
#[inline(always)]
pub fn msm_cost_function(x: Float, y: Float, z: Float) -> Float {
    MSM_C + (y.min(z) - x).max(x - y.max(z)).max(0.0)
}

pub fn cross_correlation(a: &[Float], b: &[Float]) -> Vec<Float> {
    // zero-pad the input signals a and b (add zeros to the end of each. The zero padding should fill the vectors until they reach a size of at least N = size(a)+size(b)-1
    let fft_len = (a.len() + b.len() - 1).next_power_of_two();
    let fft = Radix4::new(fft_len, FftDirection::Forward);

    let mut a_fft = vec![Complex::new(0.0, 0.0); fft_len];
    let mut b_fft = vec![Complex::new(0.0, 0.0); fft_len];
    for (i, val) in a.iter().enumerate() {
        a_fft[i] = Complex::new(*val, 0.0);
    }
    for (i, val) in b.iter().enumerate() {
        b_fft[i] = Complex::new(*val, 0.0);
    }

    fft.process(&mut a_fft);
    fft.process(&mut b_fft);

    let mut c_fft = vec![Complex::new(0.0, 0.0); fft_len];
    for i in 0..fft_len {
        c_fft[i] = a_fft[i].conj() * b_fft[i];
    }

    let mut c = vec![0.0; fft_len];
    let ifft = Radix4::new(fft_len, FftDirection::Inverse);
    ifft.process(&mut c_fft);
    for i in 0..fft_len {
        c[i] = c_fft[i].re / fft_len as Float;
    }
    c
}

pub fn zscore(x: &[Float]) -> Vec<Float> {
    let mean = x.iter().sum::<Float>() / x.len() as Float;
    let std = (x.iter().map(|val| (val - mean).powi(2)).sum::<Float>() / x.len() as Float).sqrt();
    x.iter().map(|val| (val - mean) / std).collect()
}

pub fn l2_norm(x: &[Float]) -> Float {
    x.iter().map(|val| val.powi(2)).sum::<Float>().sqrt()
}

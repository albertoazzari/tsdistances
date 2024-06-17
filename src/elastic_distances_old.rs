use crate::utils::FloatVecEq;
use dashmap::DashMap;
use lazy_static::lazy_static;
use std::{
    cmp::{max, min},
    mem::swap,
};
//use rustfft::{FftPlanner, num_complex::Complex};

lazy_static! {
    pub static ref EUCLIDEAN_CACHE: DashMap<(FloatVecEq, FloatVecEq), f64> = DashMap::new();
}
pub fn euclidean(x1: &[f64], x2: &[f64], cached: bool) -> f64 {
    let mut key_cache = (FloatVecEq(vec![]), FloatVecEq(vec![]));
    if cached {
        let x1_cache = FloatVecEq(x1.to_vec());
        let x2_cache = FloatVecEq(x2.to_vec());

        key_cache = (x1_cache, x2_cache);

        if let Some(value) = EUCLIDEAN_CACHE.get(&key_cache) {
            return *value.value();
        }
    }

    let distance = x1
        .iter()
        .zip(x2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    if cached {
        if EUCLIDEAN_CACHE.len() > 1e6 as usize {
            return distance;
        }
        EUCLIDEAN_CACHE.insert(key_cache.clone(), distance);
        swap(&mut key_cache.0, &mut key_cache.1);
        EUCLIDEAN_CACHE.insert(key_cache, distance);
    }

    distance
}

lazy_static! {
    pub static ref ERP_CACHE: DashMap<(FloatVecEq, FloatVecEq), f64> = DashMap::new();
}

pub fn erp(x1: &[f64], x2: &[f64], gap_penalty: f64, band: f64, cached: bool) -> f64 {
    let mut key_cache = (FloatVecEq(vec![]), FloatVecEq(vec![]));
    if cached {
        let x1_cache = FloatVecEq(x1.to_vec());
        let x2_cache = FloatVecEq(x2.to_vec());
        key_cache = (x1_cache, x2_cache);

        if let Some(value) = ERP_CACHE.get(&key_cache) {
            return *value.value();
        }
    }

    let n = x1.len();
    let m = x2.len();

    let sakoe_chiba_window_radius = (n as f64 + 1.0) * band;

    let alpha = ((m) as f64) / ((n) as f64);

    let mut current = vec![f64::INFINITY; m + 1];
    let mut previous = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;

    let g_x1 = x1
        .iter()
        .map(|x| (x - gap_penalty).abs())
        .collect::<Vec<_>>();
    let g_x2 = x2
        .iter()
        .map(|x| (x - gap_penalty).abs())
        .collect::<Vec<_>>();

    for i in 1..=n {
        let lower = alpha * (i as f64) - sakoe_chiba_window_radius;
        let upper = alpha * (i as f64) + sakoe_chiba_window_radius;

        let lower = max(1, lower.ceil() as usize);
        let upper = min(m, upper.floor() as usize);

        current[..].fill(f64::INFINITY);

        let x1 = |i| if i == 0 { g_x1[i] } else { x1[i - 1] };
        let x2 = |i| if i == 0 { g_x2[i] } else { x2[i - 1] };

        for j in lower..=upper {
            let cost = (x1(i - 1) - x2(j - 1)).abs();
            current[j] = (previous[j - 1] + cost)
                .min(previous[j] + g_x1[i - 1])
                .min(current[j - 1] + g_x2[j - 1]);
        }
        swap(&mut previous, &mut current);
    }

    let distance = previous[m];

    if cached {
        if ERP_CACHE.len() > 1e6 as usize {
            return distance;
        }
        ERP_CACHE.insert(key_cache.clone(), distance);
        swap(&mut key_cache.0, &mut key_cache.1);
        ERP_CACHE.insert(key_cache, distance);
    }

    distance
}

lazy_static! {
    pub static ref LCSS_CACHE: DashMap<(FloatVecEq, FloatVecEq), f64> = DashMap::new();
}
pub fn lcss(x1: &[f64], x2: &[f64], epsilon: f64, band: f64, cached: bool) -> f64 {
    let mut key_cache = (FloatVecEq(vec![]), FloatVecEq(vec![]));
    if cached {
        let x1_cache = FloatVecEq(x1.to_vec());
        let x2_cache = FloatVecEq(x2.to_vec());
        key_cache = (x1_cache, x2_cache);

        if let Some(value) = LCSS_CACHE.get(&key_cache) {
            return *value.value();
        }
    }

    let n = x1.len();
    let m = x2.len();

    let sakoe_chiba_window_radius = (n as f64 + 1.0) * band;

    let alpha = ((m) as f64) / ((n) as f64);

    let mut current = vec![0.0; m + 1];
    let mut previous = vec![0.0; m + 1];
    previous[0] = 0.0;

    for i in 1..=n {
        let lower = alpha * (i as f64) - sakoe_chiba_window_radius;
        let upper = alpha * (i as f64) + sakoe_chiba_window_radius;

        let lower = max(1, lower.ceil() as usize);
        let upper = min(m, upper.floor() as usize);

        current[..].fill(0.0);

        // closure to handle the insertion of g_x_sum as padding in the first element of the timeseries
        let x1 = |i| if i == 0 { 0.0 } else { x1[i - 1] };
        let x2 = |i| if i == 0 { 0.0 } else { x2[i - 1] };

        for j in lower..=upper {
            let cost = (x1(i - 1) - x2(j - 1)).abs();
            if cost <= epsilon {
                current[j] = previous[j - 1] + 1.0;
            } else {
                current[j] = f64::max(previous[j], current[j - 1]);
            }
        }
        swap(&mut previous, &mut current);
    }

    let distance = (min(n, m) as f64 - previous[m]) / min(n, m) as f64;

    if cached {
        if LCSS_CACHE.len() > 1e6 as usize {
            return distance;
        }
        LCSS_CACHE.insert(key_cache.clone(), distance);
        swap(&mut key_cache.0, &mut key_cache.1);
        LCSS_CACHE.insert(key_cache, distance);
    }

    return distance;
}

lazy_static! {
    pub static ref TWE_CACHE: DashMap<(FloatVecEq, FloatVecEq), f64> = DashMap::new();
}
pub fn twe(x1: &[f64], x2: &[f64], nu: f64, lambda: f64, band: f64, cached: bool) -> f64 {
    let mut key_cache = (FloatVecEq(vec![]), FloatVecEq(vec![]));
    if cached {
        let x1_cache = FloatVecEq(x1.to_vec());
        let x2_cache = FloatVecEq(x2.to_vec());
        key_cache = (x1_cache, x2_cache);

        if let Some(value) = TWE_CACHE.get(&key_cache) {
            return *value.value();
        }
    }

    let n = x1.len();
    let m = x2.len();

    let delete_addition = nu + lambda;
    let sakoe_chiba_window_radius = (n as f64 + 1.0) * band;

    let alpha = ((m) as f64) / ((n) as f64);

    let mut current = vec![f64::INFINITY; m + 1];
    let mut previous = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;

    for i in 1..=n {
        let lower = alpha * (i as f64) - sakoe_chiba_window_radius;
        let upper = alpha * (i as f64) + sakoe_chiba_window_radius;

        let lower = max(1, lower.ceil() as usize);
        let upper = min(m, upper.floor() as usize);

        current[..].fill(f64::INFINITY);

        // closure to handle the insertion of zeros as padding in the first element of the timeseries
        let x1 = |i| if i == 0 { 0.0 } else { x1[i - 1] };
        let x2 = |i| if i == 0 { 0.0 } else { x2[i - 1] };

        for j in lower..=upper {
            // deletion in x1
            let deletion_x1_euclidean_dist = (x1(i - 1) - x2(i)).abs();
            let del_x1: f64 = previous[j] + deletion_x1_euclidean_dist + delete_addition;

            // deletion in x2
            let deletion_x2_euclidean_dist = (x1(j - 1) - x2(j)).abs();
            let del_x2 = current[j - 1] + deletion_x2_euclidean_dist + delete_addition;

            // match
            let match_same_euclid_dist = (x1(i) - x2(j)).abs();
            let match_previous_euclid_dist = (x1(i - 1) - x2(j - 1)).abs();

            let match_x1_x2 = previous[j - 1]
                + match_same_euclid_dist
                + match_previous_euclid_dist
                + (nu * (2.0 * (i as isize - j as isize).abs() as f64));

            current[j] = del_x1.min(del_x2.min(match_x1_x2));
        }
        swap(&mut previous, &mut current);
    }
    let distance = previous[m];

    if cached {
        if TWE_CACHE.len() > 1e6 as usize {
            return distance;
        }
        TWE_CACHE.insert(key_cache.clone(), distance);
        swap(&mut key_cache.0, &mut key_cache.1);
        TWE_CACHE.insert(key_cache, distance);
    }

    distance
}

lazy_static! {
    pub static ref DTW_CACHE: DashMap<(FloatVecEq, FloatVecEq), f64> = DashMap::new();
}
pub fn dtw(x1: &[f64], x2: &[f64], band: f64, cached: bool) -> f64 {
    let mut key_cache = (FloatVecEq(vec![]), FloatVecEq(vec![]));
    if cached {
        let x1_cache = FloatVecEq(x1.to_vec());
        let x2_cache = FloatVecEq(x2.to_vec());
        key_cache = (x1_cache, x2_cache);

        if let Some(value) = DTW_CACHE.get(&key_cache) {
            return *value.value();
        }
    }

    let n = x1.len();
    let m = x2.len();

    let sakoe_chiba_window_radius = (n as f64 + 1.0) * band;

    let alpha = ((m) as f64) / ((n) as f64);

    let mut current = vec![f64::INFINITY; m + 1];
    let mut previous = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;

    for i in 1..=n {
        let lower = alpha * (i as f64) - sakoe_chiba_window_radius;
        let upper = alpha * (i as f64) + sakoe_chiba_window_radius;

        let lower = max(1, lower.ceil() as usize);
        let upper = min(m, upper.floor() as usize);

        current[..].fill(f64::INFINITY);

        // closure to handle the insertion of zeros as padding in the first element of the timeseries
        let x1 = |i| if i == 0 { 0.0 } else { x1[i - 1] };
        let x2 = |i| if i == 0 { 0.0 } else { x2[i - 1] };

        for j in lower..=upper {
            let dist = (x1(i) - x2(j)).powi(2);
            current[j] = dist + previous[j].min(current[j - 1].min(previous[j - 1]));
        }
        swap(&mut previous, &mut current);
    }

    let distance = previous[m];

    if cached {
        if DTW_CACHE.len() > 1e6 as usize {
            return distance;
        }
        DTW_CACHE.insert(key_cache.clone(), distance);
        swap(&mut key_cache.0, &mut key_cache.1);
        DTW_CACHE.insert(key_cache, distance);
    }

    distance
}

lazy_static! {
    pub static ref DDTW_CACHE: DashMap<(FloatVecEq, FloatVecEq), f64> = DashMap::new();
}
pub fn ddtw(x1: &[f64], x2: &[f64], band: f64, cached: bool) -> f64 {
    let mut key_cache = (FloatVecEq(vec![]), FloatVecEq(vec![]));
    if cached {
        let x1_cache = FloatVecEq(x1.to_vec());
        let x2_cache = FloatVecEq(x2.to_vec());
        key_cache = (x1_cache, x2_cache);

        if let Some(value) = DDTW_CACHE.get(&key_cache) {
            return *value.value();
        }
    }

    let x1 = derivate(x1);
    let x2 = derivate(x2);

    let n = x1.len();
    let m = x2.len();

    let sakoe_chiba_window_radius = (n as f64 + 1.0) * band;

    let alpha = ((m) as f64) / ((n) as f64);

    let mut current = vec![f64::INFINITY; m + 1];
    let mut previous = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;

    for i in 1..=n {
        let lower = alpha * (i as f64) - sakoe_chiba_window_radius;
        let upper = alpha * (i as f64) + sakoe_chiba_window_radius;

        let lower = max(1, lower.ceil() as usize);
        let upper = min(m, upper.floor() as usize);

        current[..].fill(f64::INFINITY);

        // closure to handle the insertion of zeros as padding in the first element of the timeseries
        let x1 = |i| if i == 0 { 0.0 } else { x1[i - 1] };
        let x2 = |i| if i == 0 { 0.0 } else { x2[i - 1] };

        for j in lower..=upper {
            let dist = (x1(i) - x2(j)).powi(2);
            current[j] = dist + previous[j].min(current[j - 1].min(previous[j - 1]));
        }
        swap(&mut previous, &mut current);
    }

    let distance = previous[m];

    if cached {
        if DDTW_CACHE.len() > 1e6 as usize {
            return distance;
        }
        DDTW_CACHE.insert(key_cache.clone(), distance);
        swap(&mut key_cache.0, &mut key_cache.1);
        DDTW_CACHE.insert(key_cache, distance);
    }

    distance
}

fn derivate(x: &[f64]) -> Vec<f64> {
    let mut x_d = vec![0.0; x.len()];
    for i in 1..x.len() - 1 {
        x_d[i] = ((x[i] - x[i - 1]) + (x[i + 1] - x[i - 1]) / 2.0) / 2.0;
    }
    x_d[0] = x_d[1];
    x_d[x.len() - 1] = x_d[x.len() - 2];
    x_d
}

lazy_static! {
    pub static ref WDTW_CACHE: DashMap<(FloatVecEq, FloatVecEq), f64> = DashMap::new();
}
pub fn wdtw(x1: &[f64], x2: &[f64], band: f64, cached: bool) -> f64 {
    let mut key_cache = (FloatVecEq(vec![]), FloatVecEq(vec![]));
    if cached {
        let x1_cache = FloatVecEq(x1.to_vec());
        let x2_cache = FloatVecEq(x2.to_vec());
        key_cache = (x1_cache, x2_cache);

        if let Some(value) = WDTW_CACHE.get(&key_cache) {
            return *value.value();
        }
    }

    let n = x1.len();
    let m = x2.len();

    let weights = init_weights(max(n, m), 0.05);

    let sakoe_chiba_window_radius = (n as f64 + 1.0) * band;

    let alpha = ((m) as f64) / ((n) as f64);

    let mut current = vec![f64::INFINITY; m + 1];
    let mut previous = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;

    for i in 1..=n {
        let lower = alpha * (i as f64) - sakoe_chiba_window_radius;
        let upper = alpha * (i as f64) + sakoe_chiba_window_radius;

        let lower = max(1, lower.ceil() as usize);
        let upper = min(m, upper.floor() as usize);

        current[..].fill(f64::INFINITY);

        // closure to handle the insertion of zeros as padding in the first element of the timeseries
        let x1 = |i| if i == 0 { 0.0 } else { x1[i - 1] };
        let x2 = |i| if i == 0 { 0.0 } else { x2[i - 1] };

        for j in lower..=upper {
            let dist = (x1(i) - x2(j)).powi(2) * weights[(i as i32 - j as i32).abs() as usize];
            current[j] = dist + previous[j].min(current[j - 1].min(previous[j - 1]));
        }
        swap(&mut previous, &mut current);
    }

    let distance = previous[m];

    if cached {
        if WDTW_CACHE.len() > 1e6 as usize {
            return distance;
        }
        WDTW_CACHE.insert(key_cache.clone(), distance);
        swap(&mut key_cache.0, &mut key_cache.1);
        WDTW_CACHE.insert(key_cache, distance);
    }

    distance
}
const WEIGHT_MAX: f64 = 1.0;
fn init_weights(len: usize, g: f64) -> Vec<f64> {
    let mut weights = vec![0.0; len];
    let half_len = len as f64 / 2.0;
    for i in 0..len {
        weights[i] = WEIGHT_MAX / (1.0 + std::f64::consts::E.powf(-g * (i as f64 - half_len)));
    }
    weights
}

lazy_static! {
    pub static ref WDDTW_CACHE: DashMap<(FloatVecEq, FloatVecEq), f64> = DashMap::new();
}
pub fn wddtw(x1: &[f64], x2: &[f64], band: f64, cached: bool) -> f64 {
    let mut key_cache = (FloatVecEq(vec![]), FloatVecEq(vec![]));
    if cached {
        let x1_cache = FloatVecEq(x1.to_vec());
        let x2_cache = FloatVecEq(x2.to_vec());
        key_cache = (x1_cache, x2_cache);

        if let Some(value) = WDDTW_CACHE.get(&key_cache) {
            return *value.value();
        }
    }

    let x1 = derivate(x1);
    let x2 = derivate(x2);

    let distance = wdtw(&x1, &x2, band, false);

    if cached {
        if WDDTW_CACHE.len() > 1e6 as usize {
            return distance;
        }
        WDDTW_CACHE.insert(key_cache.clone(), distance);
        swap(&mut key_cache.0, &mut key_cache.1);
        WDDTW_CACHE.insert(key_cache, distance);
    }

    distance
}

lazy_static! {
    pub static ref MSM_CACHE: DashMap<(FloatVecEq, FloatVecEq), f64> = DashMap::new();
}
pub fn msm(x1: &[f64], x2: &[f64], band: f64, cached: bool) -> f64 {
    let mut key_cache = (FloatVecEq(vec![]), FloatVecEq(vec![]));
    if cached {
        let x1_cache = FloatVecEq(x1.to_vec());
        let x2_cache = FloatVecEq(x2.to_vec());
        key_cache = (x1_cache, x2_cache);

        if let Some(value) = MSM_CACHE.get(&key_cache) {
            return *value.value();
        }
    }

    let n = x1.len();
    let m = x2.len();

    let sakoe_chiba_window_radius = (n as f64 + 1.0) * band;

    let alpha = ((m) as f64) / ((n) as f64);

    let mut previous = vec![0.0; m];
    let mut current = vec![0.0; m];
    previous[0] = (x1[0] - x2[0]).abs();
    for j in 1..m {
        previous[j] = previous[j - 1] + msm_cost_function(x2[j], x1[0], x2[j - 1]);
    }

    for i in 1..n {
        let lower = alpha * (i as f64) - sakoe_chiba_window_radius;
        let upper = alpha * (i as f64) + sakoe_chiba_window_radius;

        let lower = max(1, lower.ceil() as usize);
        let upper = min(m, upper.floor() as usize);

        current[..].fill(f64::INFINITY);

        current[0] = previous[0] + msm_cost_function(x1[i], x1[i - 1], x2[0]);
        for j in lower..upper {
            current[j] = (previous[j - 1] + (x1[i] - x2[j]).abs())
                .min(previous[j] + msm_cost_function(x1[i], x1[i - 1], x2[j]))
                .min(current[j - 1] + msm_cost_function(x2[j], x1[i], x2[j - 1]));
        }
        swap(&mut previous, &mut current);
    }
    let distance = previous[m - 1];

    if cached {
        if MSM_CACHE.len() > 1e6 as usize {
            return distance;
        }
        MSM_CACHE.insert(key_cache.clone(), distance);
        swap(&mut key_cache.0, &mut key_cache.1);
        MSM_CACHE.insert(key_cache, distance);
    }

    distance
}

const MSM_C: f64 = 1.0;
fn msm_cost_function(x_i: f64, x_i_1: f64, y_j: f64) -> f64 {
    if (x_i >= x_i_1 && x_i <= y_j) || (x_i_1 >= x_i && x_i >= y_j) {
        MSM_C
    } else {
        MSM_C + (x_i - x_i_1).abs().min((x_i - y_j).abs())
    }
}

lazy_static! {
    pub static ref ADTW_CACHE: DashMap<(FloatVecEq, FloatVecEq), f64> = DashMap::new();
}
pub fn adtw(x1: &[f64], x2: &[f64], w: f64, band: f64, cached: bool) -> f64 {
    let mut key_cache = (FloatVecEq(vec![]), FloatVecEq(vec![]));
    if cached {
        let x1_cache = FloatVecEq(x1.to_vec());
        let x2_cache = FloatVecEq(x2.to_vec());
        key_cache = (x1_cache, x2_cache);

        if let Some(value) = ADTW_CACHE.get(&key_cache) {
            return *value.value();
        }
    }

    let n = x1.len();
    let m = x2.len();

    let sakoe_chiba_window_radius = (n as f64 + 1.0) * band;

    let alpha = ((m) as f64) / ((n) as f64);

    let mut current = vec![f64::INFINITY; m + 1];
    let mut previous = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;

    for i in 1..=n {
        let lower = alpha * (i as f64) - sakoe_chiba_window_radius;
        let upper = alpha * (i as f64) + sakoe_chiba_window_radius;

        let lower = max(1, lower.ceil() as usize);
        let upper = min(m, upper.floor() as usize);

        current[..].fill(f64::INFINITY);

        // closure to handle the insertion of zeros as padding in the first element of the timeseries
        let x1 = |i| if i == 0 { 0.0 } else { x1[i - 1] };
        let x2 = |i| if i == 0 { 0.0 } else { x2[i - 1] };

        for j in lower..=upper {
            let dist = (x1(i) - x2(j)).powi(2);
            current[j] = dist + (previous[j] + w).min((current[j - 1] + w).min(previous[j - 1]));
        }
        swap(&mut previous, &mut current);
    }

    let distance = previous[m];

    if cached {
        if ADTW_CACHE.len() > 1e6 as usize {
            return distance;
        }
        ADTW_CACHE.insert(key_cache.clone(), distance);
        swap(&mut key_cache.0, &mut key_cache.1);
        ADTW_CACHE.insert(key_cache, distance);
    }

    distance
}

// lazy_static! {
//     pub static ref SBD_CACHE: DashMap<(FloatVecEq, FloatVecEq), f64> = DashMap::new();
// }

// pub fn sbd(x1: &[f64], x2: &[f64], cached: bool) -> f64 {
//     let mut key_cache = (FloatVecEq(vec![]), FloatVecEq(vec![]));
//     if cached {
//         let x1_cache = FloatVecEq(x1.to_vec());
//         let x2_cache = FloatVecEq(x2.to_vec());
//         key_cache = (x1_cache, x2_cache);

//         if let Some(value) = SBD_CACHE.get(&key_cache) {
//             return *value.value();
//         }
//     }

//     let n = x1.len();

//     let mut planner = FftPlanner::new();
//     let fft = planner.plan_fft_forward(n);

//     let mut x1_complex = x1.iter().map(|x| Complex::new(*x, 0.0)).collect::<Vec<_>>();
//     let mut x2_complex = x2.iter().map(|x| Complex::new(*x, 0.0)).collect::<Vec<_>>();

//     fft.process(&mut x1_complex);
//     fft.process(&mut x2_complex);

//     let cc = x1_complex.iter()
//         .zip(x2_complex.iter())
//         .map(|(a, b)| a * b.conj())
//         .collect::<Vec<_>>();

//     let mut ifft = planner.plan_fft_inverse(n);
//     let mut cc_ifft = cc.clone();
//     ifft.process(&mut cc_ifft);

//     let norm = norm(x1, x2);
//     let ncc = cc_ifft.iter()
//         .map(|x| x.re / norm)
//         .collect::<Vec<_>>();

//     1.0 - ncc.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()

// }

// fn norm(x: &[f64], y: &[f64]) -> f64 {
//     let x_ = x.iter().map(|x| x.powi(2)).sum::<f64>();
//     let y_ = y.iter().map(|y| y.powi(2)).sum::<f64>();

//     (x_ * y_).sqrt()
// }

#![allow(dead_code)]
use crate::{
    diagonal,
    matrix::DiagonalMatrix,
    utils::{cross_correlation, derivate, dtw_weights, l2_norm, msm_cost_function, zscore},
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::max;
use tsdistances_gpu::device::get_best_gpu;

const MIN_CHUNK_SIZE: usize = 16;
const CHUNKS_PER_THREAD: usize = 8;

/// Computes the pairwise distance between two sets of timeseries.
///
/// This function computes the distance between each pair of timeseries (one from each set) using the
/// provided distance function. The computation is parallelized across multiple threads to improve
/// performance. The number of threads used can be controlled via the `n_jobs` parameter.
///
fn compute_distance(
    distance: impl (Fn(&[f64], &[f64]) -> f64) + Sync + Send,
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    n_jobs: i32,
) -> Vec<Vec<f64>> {
    let n_jobs = if n_jobs == -1 {
        rayon::current_num_threads() as usize
    } else {
        n_jobs.max(1) as usize
    };

    let semaphore = parking_lot::Mutex::new(n_jobs);
    let cond_var = parking_lot::Condvar::new();
    let x1 = x1.into_iter().enumerate().collect::<Vec<_>>();
    let distance_matrix = x1
        .par_chunks(max(MIN_CHUNK_SIZE, x1.len() / n_jobs / CHUNKS_PER_THREAD))
        .map(|a| {
            let mut guard = semaphore.lock();
            while *guard == 0 {
                cond_var.wait(&mut guard);
            }
            *guard -= 1;
            drop(guard);
            let result = a
                .iter()
                .map(|(i, a)| {
                    if let Some(x2) = &x2 {
                        x2.iter()
                            .map(|b| {
                                let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
                                distance(a, b)
                            })
                            .collect::<Vec<_>>()
                    } else {
                        x1.iter()
                            .take(*i)
                            .map(|(_, b)| {
                                let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
                                distance(a, b)
                            })
                            .collect::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        })
        .flatten()
        .collect::<Vec<_>>();
    if x2.is_none() {
        let mut distance_matrix = distance_matrix;
        for i in 0..distance_matrix.len() {
            distance_matrix[i].push(0.0);
            for j in i + 1..distance_matrix.len() {
                let d = distance_matrix[j][i];
                distance_matrix[i].push(d);
            }
        }
        distance_matrix
    } else {
        distance_matrix
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, n_jobs=-1))]
pub fn euclidean(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {
    let distance_matrix = compute_distance(
        |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, gap_penalty=0.0, n_jobs=-1, device="cpu"))]
pub fn erp(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    gap_penalty: f64,
    n_jobs: i32,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    if gap_penalty < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Gap penalty must be non-negative",
        ));
    }
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;
    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        diagonal::diagonal_distance::<DiagonalMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            |a, b, i, j, x, y, z| {
                                (y + (a[i] - b[j]).abs()).min(
                                    (z + (a[i] - gap_penalty).abs())
                                        .min(x + (b[j] - gap_penalty).abs()),
                                )
                            },
                        )
                    },
                    x1,
                    x2,
                    n_jobs,
                ));
            }
            "gpu" => {
                let device_gpu = get_best_gpu();
                distance_matrix = Some(compute_distance(
                    |a, b| tsdistances_gpu::erp(device_gpu.clone(), a, b, gap_penalty),
                    x1,
                    x2,
                    1,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }
    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing ERP distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, epsilon=1.0, n_jobs=-1, device="cpu"))]
pub fn lcss(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    epsilon: f64,
    n_jobs: i32,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    if epsilon < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Epsilon must be non-negative",
        ));
    }
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;

    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        let similarity = diagonal::diagonal_distance::<DiagonalMatrix>(
                            a,
                            b,
                            0.0,
                            sakoe_chiba_band,
                            |a, b, i, j, x, y, z| {
                                let dist = (a[i] - b[j]).abs();
                                if dist <= epsilon {
                                    y + 1.0
                                } else {
                                    x.max(z)
                                }
                            },
                        );
                        let min_len = a.len().min(b.len()) as f64;
                        1.0 - similarity / min_len
                    },
                    x1,
                    x2,
                    n_jobs,
                ));
            }
            "gpu" => {
                let device_gpu = get_best_gpu();
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        let similarity = tsdistances_gpu::lcss(device_gpu.clone(), a, b, epsilon);
                        let min_len = a.len().min(b.len()) as f64;
                        1.0 - similarity / min_len
                    },
                    x1,
                    x2,
                    1,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }
    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing LCSS distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, n_jobs=-1, device="cpu"))]
pub fn dtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    n_jobs: i32,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;

    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        diagonal::diagonal_distance::<DiagonalMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            |a, b, i, j, x, y, z| {
                                let dist = (a[i] - b[j]).powi(2);
                                dist + z.min(x.min(y))
                            },
                        )
                    },
                    x1,
                    x2,
                    n_jobs,
                ));
            }
            "gpu" => {
                let device_gpu = get_best_gpu();
                distance_matrix = Some(compute_distance(
                    |a, b| tsdistances_gpu::dtw(device_gpu.clone(), a, b),
                    x1,
                    x2,
                    1,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }

    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing DTW distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, n_jobs=-1, device="cpu"))]
pub fn ddtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    n_jobs: i32,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let x1_d = derivate(&x1);
    let x2_d = if let Some(x2) = &x2 {
        Some(derivate(&x2))
    } else {
        None
    };
    dtw(x1_d, x2_d, sakoe_chiba_band, n_jobs, device)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, g=0.05, n_jobs=-1, device="cpu"))]
pub fn wdtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    g: f64, //constant that controls the curvature (slope) of the function
    n_jobs: i32,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;

    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        let weights = dtw_weights(a.len().max(b.len()), g);
                        diagonal::diagonal_distance::<DiagonalMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            |a, b, i, j, x, y, z| {
                                let dist = (a[i] - b[j]).powi(2)
                                    * weights[(i as i32 - j as i32).abs() as usize];
                                dist + z.min(x.min(y))
                            },
                        )
                    },
                    x1,
                    x2,
                    n_jobs,
                ));
            }
            "gpu" => {
                let device_gpu = get_best_gpu();
                distance_matrix = Some(compute_distance(
                    |a, b| tsdistances_gpu::wdtw(device_gpu.clone(), a, b, g),
                    x1,
                    x2,
                    1,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }

    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing WDTW distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, g=0.05, n_jobs=-1, device="cpu"))]
pub fn wddtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    g: f64,
    n_jobs: i32,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let x1_d = derivate(&x1);
    let x2_d = if let Some(x2) = &x2 {
        Some(derivate(&x2))
    } else {
        None
    };
    wdtw(x1_d, x2_d, sakoe_chiba_band, g, n_jobs, device)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, n_jobs=-1, device="cpu"))]
pub fn msm(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    n_jobs: i32,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;

    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        diagonal::diagonal_distance::<DiagonalMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            |a, b, i, j, x, y, z| {
                                (y + (a[i] - b[j]).abs())
                                    .min(
                                        z + msm_cost_function(
                                            a[i],
                                            a.get(i - 1).copied().unwrap_or(0.0),
                                            b[j],
                                        ),
                                    )
                                    .min(
                                        x + msm_cost_function(
                                            b[j],
                                            a[i],
                                            b.get(j - 1).copied().unwrap_or(0.0),
                                        ),
                                    )
                            },
                        )
                    },
                    x1,
                    x2,
                    n_jobs,
                ));
            }
            "gpu" => {
                let device_gpu = get_best_gpu();
                distance_matrix = Some(compute_distance(
                    |a, b| tsdistances_gpu::msm(device_gpu.clone(), a, b),
                    x1,
                    x2,
                    1,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }

    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing MSM distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, stiffness=0.001, penalty=1.0, n_jobs=-1, device="cpu"))]
pub fn twe(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    stiffness: f64,
    penalty: f64,
    n_jobs: i32,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    if stiffness < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Stiffness (nu) must be non-negative",
        ));
    }
    if penalty < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Penalty (lambda) must be non-negative",
        ));
    }
    let delete_addition = stiffness + penalty;

    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;

    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        diagonal::diagonal_distance::<DiagonalMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            |a, b, i, j, x, y, z| {
                                // deletion in a
                                let del_a: f64 = z
                                    + (a.get(i - 1).copied().unwrap_or(0.0) - a[i]).abs()
                                    + delete_addition;

                                // deletion in b
                                let del_b = x
                                    + (b.get(j - 1).copied().unwrap_or(0.0) - b[j]).abs()
                                    + delete_addition;

                                // match
                                let match_current = (a[i] - b[j]).abs();
                                let match_previous = (a.get(i - 1).copied().unwrap_or(0.0)
                                    - b.get(j - 1).copied().unwrap_or(0.0))
                                .abs();
                                let match_a_b = y
                                    + match_current
                                    + match_previous
                                    + stiffness * (2.0 * (i as isize - j as isize).abs() as f64);

                                del_a.min(del_b.min(match_a_b))
                            },
                        )
                    },
                    x1,
                    x2,
                    n_jobs,
                ));
            }
            "gpu" => {
                let device_gpu = get_best_gpu();
                distance_matrix = Some(compute_distance(
                    |a, b| tsdistances_gpu::twe(device_gpu.clone(), a, b, stiffness, penalty),
                    x1,
                    x2,
                    1,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }

    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing TWE distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, warp_penalty=0.1, n_jobs=-1, device="cpu"))]
pub fn adtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    warp_penalty: f64,
    n_jobs: i32,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    if warp_penalty < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Weight must be non-negative",
        ));
    }
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;
    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        diagonal::diagonal_distance::<DiagonalMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            |a, b, i, j, x, y, z| {
                                let dist = (a[i] - b[j]).powi(2);
                                dist + (z + warp_penalty).min((x + warp_penalty).min(y))
                            },
                        )
                    },
                    x1,
                    x2,
                    n_jobs,
                ));
            }
            "gpu" => {
                let device_gpu = get_best_gpu();
                distance_matrix = Some(compute_distance(
                    |a, b| tsdistances_gpu::adtw(device_gpu.clone(), a, b, warp_penalty),
                    x1,
                    x2,
                    1,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }

    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing ADTW distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, n_jobs=-1))]
pub fn sbd(x1: Vec<Vec<f64>>, x2: Option<Vec<Vec<f64>>>, n_jobs: i32) -> PyResult<Vec<Vec<f64>>> {
    let distance_matrix = compute_distance(
        |a, b| {
            let a = zscore(&a);
            let b = zscore(&b);
            let cc = cross_correlation(&a, &b);
            1.0 - cc.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
                / (l2_norm(&a) * l2_norm(&b))
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

#![allow(dead_code)]
use std::cmp::max;
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::{diagonal, utils::{cross_correlation, derivate, dtw_weights, l2_norm, msm_cost_function, zscore}};

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
                        x2.iter().map(|b| distance(a, b)).collect::<Vec<_>>()
                    } else {
                        x1.iter()
                            .take(*i)
                            .map(|(_, b)| distance(a, b))
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
            diagonal::diagonal_distance(a, b, f64::INFINITY, |i, j, _x, y, _z| {
                y + (a[i] - b[j]).powi(2)
            }).sqrt()
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, gap_penalty=0.0, n_jobs=-1))]
pub fn erp(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    gap_penalty: f64,
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {
    if gap_penalty < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Gap penalty must be non-negative",
        ));
    }
    let distance_matrix = compute_distance(
        |a, b| {
            diagonal::diagonal_distance(a, b, f64::INFINITY, |i, j, x, y, z| {
                (y + (a[i] - b[j]).abs())
                    .min((z + (a[i] - gap_penalty).abs()).min(x + (b[j] - gap_penalty).abs()))
            })
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, epsilon=1.0, n_jobs=-1))]
pub fn lcss(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    epsilon: f64,
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {
    if epsilon < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Epsilon must be non-negative",
        ));
    }
    let distance_matrix = compute_distance(
        |a, b| {
            let similarity = diagonal::diagonal_distance(a, b, 0.0, |i, j, x, y, z| {
                let dist = (a.get(i - 1).copied().unwrap_or(0.0) - b.get(j - 1).copied().unwrap_or(0.0)).abs();
                if dist <= epsilon {
                    y + 1.0
                } else {
                    x.max(z)
                }
            });
            let min_len = a.len().min(b.len()) as f64;
            (min_len - similarity) / min_len
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, n_jobs=-1))]
pub fn dtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {
    let distance_matrix = compute_distance(
        |a, b| {
            diagonal::diagonal_distance(a, b, f64::INFINITY, |i, j, x, y, z| {
                let dist = (a[i] - b[j]).powi(2);
                dist + z.min(x.min(y))
            })
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, n_jobs=-1))]
pub fn ddtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {
    let x1_d = derivate(&x1);
    let x2_d = if let Some(x2) = &x2 {
        Some(derivate(&x2))
    } else {
        None
    };
    dtw(x1_d, x2_d, n_jobs)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, g=0.0, n_jobs=-1))]
pub fn wdtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    g: f64, //constant that controls the curvature (slope) of the function
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {
    let distance_matrix = compute_distance(
        |a, b| {
            let weights = dtw_weights(a.len().max(b.len()), g);
            diagonal::diagonal_distance(a, b, f64::INFINITY, |i, j, x, y, z| {
                let dist = (a[i] - b[j]).powi(2) * weights[(i as i32 - j as i32).abs() as usize];
                dist + x.min(y.min(z))
            })
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, g=0.0, n_jobs=-1))]
pub fn wddtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    g: f64,
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {
    let x1_d = derivate(&x1);
    let x2_d = if let Some(x2) = &x2 {
        Some(derivate(&x2))
    } else {
        None
    };
    wdtw(x1_d, x2_d, g, n_jobs)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, n_jobs=-1))]
pub fn msm(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {
    let distance_matrix = compute_distance(
        |a, b| {
            diagonal::diagonal_distance(a, b, f64::INFINITY, |i, j, x, y, z| {
                (y + (a[i] - b[j]).abs())
                .min(z + msm_cost_function(a[i], a.get(i - 1).copied().unwrap_or(0.0), b[j]))
                .min(x + msm_cost_function(b[j], a[i], b.get(j - 1).copied().unwrap_or(0.0)))
            })
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, stiffness=0.001, penalty=1.0, n_jobs=-1))]
pub fn twe(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    stiffness: f64,
    penalty: f64,
    n_jobs: i32,
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

    let distance_matrix = compute_distance(
        |a, b| {
            diagonal::diagonal_distance(a, b, f64::INFINITY, |i, j, x, y, z| {
                // deletion in x1
                let deletion_x1_euclidean_dist = (a.get(i - 1).copied().unwrap_or(0.0) - b[i]).abs();
                let del_x1: f64 = z + deletion_x1_euclidean_dist + delete_addition;

                // deletion in x2
                let deletion_x2_euclidean_dist = (a.get(j - 1).copied().unwrap_or(0.0) - b[j]).abs();
                let del_x2 = x + deletion_x2_euclidean_dist + delete_addition;

                // match
                let match_same_euclid_dist = (a[i] - b[j]).abs();
                let match_previous_euclid_dist = (a.get(i - 1).copied().unwrap_or(0.0) - b.get(j - 1).copied().unwrap_or(0.0)).abs();

                let match_x1_x2 = y
                    + match_same_euclid_dist
                    + match_previous_euclid_dist
                    + (stiffness * (2.0 * (i as isize - j as isize).abs() as f64));

                del_x1.min(del_x2.min(match_x1_x2))
            })
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, w=0.1, n_jobs=-1))]
pub fn adtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    w: f64,
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {
    if w < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Weight must be non-negative",
        ));
    }
    let distance_matrix = compute_distance(
        |a, b| {
            diagonal::diagonal_distance(a, b, f64::INFINITY, |i, j, x, y, z| {
                let dist = (a[i] - b[j]).powi(2);
                dist + (z + w).min((x + w).min(y))
            })
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, n_jobs=-1))]
pub fn sbd( 
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {

    let distance_matrix = compute_distance(
        |a, b| {
            let a = zscore(&a);
            let b = zscore(&b);
            let cc = cross_correlation(&a, &b);
            1.0 - cc.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() / (l2_norm(&a) * l2_norm(&b))
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

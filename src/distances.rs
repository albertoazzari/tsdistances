#![allow(dead_code)]
use crate::{
    diagonal,
    matrix::DiagonalMatrix,
    utils::{cross_correlation, derivate, dtw_weights, l2_norm, msm_cost_function, zscore},
};
use core::f64;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::{max, min};
use tsdistances_gpu::device::get_best_gpu;
use tsdistances_gpu::GpuBatchMode;
use catch22;

const MIN_CHUNK_SIZE: usize = 16;
const CHUNKS_PER_THREAD: usize = 8;

fn compute_distance_batched(
    distance: impl (Fn(&[Vec<f64>], &[Vec<f64>], bool) -> Vec<Vec<f64>>) + Sync + Send,
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    chunk_size: usize,
) -> Vec<Vec<f64>> {
    let mut result = Vec::with_capacity(x1.len());

    let mut x1_offset = 0;
    for x1_part in x1.chunks(chunk_size) {
        result.resize_with(x1_offset + x1_part.len(), || {
            Vec::with_capacity(x2.as_ref().map_or(x1.len(), |x| x.len()))
        });
        for x2_part in x2.as_ref().unwrap_or(&x1).chunks(chunk_size) {
            let distance_matrix = distance(x1_part, x2_part, x2.is_none());
            for (x1_idx, row) in distance_matrix.iter().enumerate() {
                result[x1_offset + x1_idx].extend_from_slice(&row[..]);
            }
        }
        x1_offset += x1_part.len();
    }
    result
}

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
            let row_len = distance_matrix.len();
            distance_matrix[i].reserve(row_len - i);
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

fn check_same_length(x: &[Vec<f64>]) -> bool {
    if x.len() == 0 {
        return false;
    }

    let len = x[0].len();
    x.iter().all(|a| a.len() == len)
}

fn compute_max_group(
    count1: usize,
    count2: usize,
    len1: usize,
    len2: usize,
    max_threads: usize,
) -> usize {
    let threads_per_instance = len1.min(len2) + 1;
    let warps_per_instance = threads_per_instance.div_ceil(64);
    let max_warps = max_threads / 64;
    let max_instances = max_warps / warps_per_instance;

    let max_group = if count1 * count2 <= max_instances {
        count1.max(count2)
    } else {
        let max_sqrt = (max_instances as f64).sqrt().floor() as usize;
        if count1 < max_sqrt {
            count2 / count1
        } else if count2 < max_sqrt {
            count1 / count2
        } else {
            max_sqrt
        }
    };

    max_group.max(1)
}

macro_rules! gpu_call {
    (
        device_gpu($device_gpu:expr),
        $distance_matrix:ident = |$x1:ident($a:ident), $x2:ident($b:ident), $BatchMode:ident| {
        $($body:tt)*
    }) => {
        let max_threads = $device_gpu
            .info()
            .map(|i| i.max_groups() as usize * i.max_threads() as usize)
            .unwrap_or(65536);

        $distance_matrix = Some(
            if check_same_length(&$x1) && $x2.as_ref().map(|x2| check_same_length(&x2)).unwrap_or(true) {
                type $BatchMode = tsdistances_gpu::MultiBatchMode;

                let batch_size = compute_max_group(
                    $x1.len(),
                    $x2.as_ref().map_or($x1.len(), |x| x.len()),
                    $x1[0].len(),
                    $x2.as_ref().map_or($x1[0].len(), |x| x[0].len()),
                    max_threads,
                );

                compute_distance_batched(
                    |$a, $b, _| {
                        $($body)*
                    },
                    $x1,
                    $x2,
                    batch_size,
                )
            } else {
                type $BatchMode = tsdistances_gpu::SingleBatchMode;
                compute_distance(
                    |$a, $b| {
                        $($body)*
                    },
                    $x1,
                    $x2,
                    1,
                )
            }
        );
    };
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
#[pyo3(signature = (x1, x2=None, n_jobs=-1))]
pub fn catch_euclidean(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {
    let x1 =  x1.iter().map(|x| {
        let mut transformed_x = Vec::with_capacity(catch22::N_CATCH22);
        for i in 0..catch22::N_CATCH22{
            let value = catch22::compute(&x, i);
            if value.is_nan(){
                transformed_x.push(0.0);
            } else {
                transformed_x.push(value);
            }
        }
        return transformed_x;
    }).collect::<Vec<Vec<_>>>();
    let x2 = if let Some(x2) = x2 {
        Some(x2.iter().map(|x| {
            let mut transformed_x = Vec::with_capacity(catch22::N_CATCH22);
            for i in 0..catch22::N_CATCH22{
                let value = catch22::compute(&x, i);
                if value.is_finite(){
                    transformed_x.push(value);
                } else {
                    transformed_x.push(0.0);
                }
            }
            return transformed_x;
        }).collect::<Vec<Vec<_>>>())
    } else {
        None
    };
    // Z-Normalize on the column-wise
    let mean_x1 = (0..catch22::N_CATCH22).map(|i| {
        let sum = x1.iter().map(|x| x[i]).sum::<f64>();
        sum / x1.len() as f64
    }).collect::<Vec<f64>>();
    let std_x1 = (0..catch22::N_CATCH22).map(|i| {
        let sum = x1.iter().map(|x| (x[i] - mean_x1[i]).powi(2)).sum::<f64>();
        (sum / x1.len() as f64).sqrt()
    }).collect::<Vec<f64>>();
    let x1 = x1.iter().map(|x| {
        x.iter().enumerate().map(|(i, val)| (val - mean_x1[i]) / if std_x1[i].abs() < f64::EPSILON {1.0} else {std_x1[i]}).collect::<Vec<f64>>()
    }).collect::<Vec<Vec<f64>>>();
    
    let x2 = if let Some(x2) = x2 {
        let mean_x2 = (0..catch22::N_CATCH22).map(|i| {
            let sum = x2.iter().map(|x| x[i]).sum::<f64>();
            sum / x2.len() as f64
        }).collect::<Vec<f64>>();
        let std_x2 = (0..catch22::N_CATCH22).map(|i| {
            let sum = x2.iter().map(|x| (x[i] - mean_x2[i]).powi(2)).sum::<f64>();
            (sum / x2.len() as f64).sqrt()
        }).collect::<Vec<f64>>();
        Some(x2.iter().map(|x| {
            x.iter().enumerate().map(|(i, val)| (val - mean_x2[i]) / if std_x2[i].abs() < f64::EPSILON {1.0} else {std_x2[i]}).collect::<Vec<f64>>()
        }).collect::<Vec<Vec<f64>>>())
    } else {
        None
    };
    euclidean(x1, x2, n_jobs)
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
                gpu_call!(
                    device_gpu(device_gpu),
                    distance_matrix = |x1(a), x2(b), BatchMode| {
                        tsdistances_gpu::erp::<BatchMode>(device_gpu.clone(), a, b, gap_penalty)
                    }
                );
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
                                (dist <= epsilon) as i32 as f64 * (y + 1.0)
                                    + (dist > epsilon) as i32 as f64 * x.max(z)
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
                gpu_call!(
                    device_gpu(device_gpu),
                    distance_matrix = |x1(a), x2(b), BatchMode| {
                        let similarity =
                            tsdistances_gpu::lcss::<BatchMode>(device_gpu.clone(), a, b, epsilon);
                        let min_len =
                            BatchMode::input_seq_len(&a).min(BatchMode::input_seq_len(&b)) as f64;
                        BatchMode::apply_fn(similarity, |s| 1.0 - s / min_len)
                    }
                );
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
                gpu_call!(
                    device_gpu(device_gpu),
                    distance_matrix = |x1(a), x2(b), BatchMode| {
                        tsdistances_gpu::dtw::<BatchMode>(device_gpu.clone(), a, b)
                    }
                );
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
                gpu_call!(
                    device_gpu(device_gpu),
                    distance_matrix = |x1(a), x2(b), BatchMode| {
                        let weights = dtw_weights(
                            BatchMode::input_seq_len(&a).max(BatchMode::input_seq_len(&b)),
                            g,
                        );
                        tsdistances_gpu::wdtw::<BatchMode>(device_gpu.clone(), a, b, &weights)
                    }
                );
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
                                            a.get(i - 1).copied().unwrap_or_default(),
                                            b[j],
                                        ),
                                    )
                                    .min(
                                        x + msm_cost_function(
                                            b[j],
                                            a[i],
                                            b.get(j - 1).copied().unwrap_or_default(),
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
                gpu_call!(
                    device_gpu(device_gpu),
                    distance_matrix = |x1(a), x2(b), BatchMode| {
                        tsdistances_gpu::msm::<BatchMode>(device_gpu.clone(), a, b)
                    }
                );
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
                gpu_call!(
                    device_gpu(device_gpu),
                    distance_matrix = |x1(a), x2(b), BatchMode| {
                        tsdistances_gpu::twe::<BatchMode>(
                            device_gpu.clone(),
                            a,
                            b,
                            stiffness,
                            penalty,
                        )
                    }
                );
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
                gpu_call!(
                    device_gpu(device_gpu),
                    distance_matrix = |x1(a), x2(b), BatchMode| {
                        tsdistances_gpu::adtw::<BatchMode>(device_gpu.clone(), a, b, warp_penalty)
                    }
                );
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
pub fn sb(x1: Vec<Vec<f64>>, x2: Option<Vec<Vec<f64>>>, n_jobs: i32) -> PyResult<Vec<Vec<f64>>> {
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

#[pyfunction]
#[pyo3(signature = (x1, window, x2=None, n_jobs=-1))]
pub fn mp(
    x1: Vec<Vec<f64>>,
    window: i32,
    x2: Option<Vec<Vec<f64>>>,
    n_jobs: i32,
) -> PyResult<Vec<Vec<f64>>> {
    let threshold = 0.05;
    let window = window as usize;
    let distance_matrix = compute_distance(
        |a, b| {
            let n_a = a.len();
            let n_b = b.len();
            let mut p_abba = mp_(&a, &b, window as usize);
            let n = min(
                (threshold * (n_a + n_b) as f64).ceil() as usize,
                n_a - window + 1 + n_b - window + 1 - 1,
            );
            *p_abba
                .select_nth_unstable_by(n, |x, y| x.partial_cmp(y).unwrap())
                .1
        },
        x1,
        x2,
        n_jobs,
    );
    Ok(distance_matrix)
}

fn mp_(a: &[f64], b: &[f64], window: usize) -> Vec<f64> {
    let n_a = a.len();
    let n_b = b.len();

    let window = window.min(n_a).min(n_b);

    let mut p_ab = vec![f64::INFINITY; n_a - window + 1];
    let mut p_ba = vec![f64::INFINITY; n_b - window + 1];

    let (mean_a, std_a) = mean_std_per_windows(&a, window);
    let (mean_b, std_b) = mean_std_per_windows(&b, window);

    for (i, sw_a) in a.windows(window).enumerate() {
        for (j, sw_b) in b.windows(window).enumerate() {
            let mut dist = 0.0;
            for (x, y) in sw_a.iter().zip(sw_b.iter()) {
                dist += (((x - mean_a[i]) / std_a[i]) - ((y - mean_b[j]) / std_b[j])).powi(2);
            }
            dist = dist.sqrt();
            p_ab[i] = p_ab[i].min(dist);
            p_ba[j] = p_ba[j].min(dist);
        }
    }

    if p_ab.len() > p_ba.len() {
        p_ab.extend(p_ba);
        p_ab
    } else {
        p_ba.extend(p_ab);
        p_ba
    }
}

// fn mean_std_per_windows(a: &[f64], window: i32) -> (Vec<f64>, Vec<f64>) {
//     let mut means = Vec::with_capacity(a.len() - window + 1);
//     let mut stds = Vec::with_capacity(a.len() - window + 1);

//     for (i, sw_a) in a.windows(window).enumerate() {
//         means[i] = mean(sw_a);
//         stds[i] = std(sw_a);
//     }

//     (means, stds)
// }
fn mean_std_per_windows(a: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let n = a.len();

    let mut means = Vec::with_capacity(n - window + 1);
    let mut stds = Vec::with_capacity(n - window + 1);

    let mut sum: f64 = a[0..window].iter().sum();
    let mut sum_squares: f64 = a[0..window].iter().map(|&x| x * x).sum();

    means.push(sum / window as f64);
    let var = (sum_squares / window as f64) - (means[0] * means[0]);
    stds.push(var.sqrt());

    for i in window..n {

        sum += a[i] - a[i - window];
        sum_squares += a[i] * a[i] - a[i - window] * a[i - window];


        let mean = sum / window as f64;
        means.push(mean);

        let var = (sum_squares / window as f64) - (mean * mean);
        stds.push(var.sqrt());
    }

    (means, stds)
}

#[test]
pub fn test_mean() {
    use rand::random;
    let a = (0..100).map(|x| random::<f64>()).collect::<Vec<_>>();
    let mut mean_a = Vec::new();
    let mut std_a = Vec::new();
    for (i, sw_a) in a.windows(10).enumerate() {
        mean_a.push(sw_a.iter().sum::<f64>() / sw_a.len() as f64);
        std_a.push((sw_a.iter().map(|val| (val - mean_a[i]).powi(2)).sum::<f64>() / sw_a.len() as f64).sqrt());
    }

    let (mean_a_, std_a_) = mean_std_per_windows(&a, 10);

    // Define a tolerance for floating-point comparison
    let tolerance = 1e-8;

    // Check lengths
    assert_eq!(mean_a.len(), mean_a_.len());
    assert_eq!(std_a.len(), std_a_.len());

    // Check each value with tolerance
    for (m1, m2) in mean_a.iter().zip(mean_a_) {
        assert!((m1 - m2).abs() < tolerance, "Mean values differ: {} vs {}", m1, m2);
    }
    for (s1, s2) in std_a.iter().zip(std_a_) {
        assert!((s1 - s2).abs() < tolerance, "Std values differ: {} vs {}", s1, s2);
    }
}

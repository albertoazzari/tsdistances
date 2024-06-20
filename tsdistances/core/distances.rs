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
/// Computes the pairwise Euclidean distances between two sets of timeseries.
///
/// Given two sets of timeseries `x1` and `x2`, this function computes the Euclidean distance
/// between each pair of timeseries (one from each set). The computation is parallelized across
/// multiple threads to improve performance. The number of threads used can be controlled
/// via the `n_jobs` parameter.
///
/// # Parameters
/// - `x1` (Vec<Vec<f64>>): The first set of timeseries with shape (n, m) where `n` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `x2` (Vec<Vec<f64>>): The second set of timeseries with shape (k, m) where `k` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `cached` (bool): Whether to use cached computations if available. This can be used to speed up
///   repeated distance calculations with the same input data.
/// - `n_jobs` (i32): The number of threads to use for parallel computation. If set to `-1`,
///   all available CPU cores will be used.
///
/// # Returns
/// Vec<Vec<f64>>: A matrix of shape (n, k) where each element [i][j] represents the Euclidean
/// distance between the i-th timeseries of `x1` and the j-th timeseries of `x2`.
///
/// # Examples
/// ```python
/// import tsdistances
///
/// x1 = [
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ]
///
/// x2 = [
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ]
///
/// # Calculate the distances using 4 threads
/// result = tsdistances.euclidean(x1, x2, cached=False, n_jobs=4)
/// print(result)  # Output: [[10.392304845413264, 15.588457268119896], [5.196152422706632, 10.392304845413264]]
/// ```
///
/// # Notes
/// - The function uses a mutex and condition variable to manage thread synchronization.
/// - The computation splits the input data into chunks to balance the load across the available threads.
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
/// Computes the pairwise Edit Distance with Real Penalty (ERP) between two sets of timeseries.
///
/// Given two sets of timeseries `x1` and `x2`, this function computes the ERP distance
/// between each pair of timeseries (one from each set). The computation is parallelized across
/// multiple threads to improve performance. The number of threads used can be controlled
/// via the `n_jobs` parameter.
///
/// # Parameters
/// - `x1` (Vec<Vec<f64>>): The first set of timeseries with shape (n, m) where `n` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `x2` (Vec<Vec<f64>>): The second set of timeseries with shape (k, m) where `k` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `gap_penalty` (f64): The gap penalty to use in the ERP calculation.
/// - `band` (f64): The Sakoe-Chiba band width, used to constrain the warping window.
/// - `cached` (bool): Whether to use cached computations if available. This can be used to speed up
///   repeated distance calculations with the same input data.
/// - `n_jobs` (i32): The number of threads to use for parallel computation. If set to `-1`,
///   all available CPU cores will be used.
///
/// # Returns
/// Vec<Vec<f64>>: A matrix of shape (n, k) where each element [i][j] represents the ERP distance
/// between the i-th timeseries of `x1` and the j-th timeseries of `x2`.
///
/// # Examples
/// ```python
/// import tsdistances
///
/// x1 = [
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ]
///
/// x2 = [
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ]
///
/// # Calculate the distances using a gap penalty of 1.0 and 4 threads
/// result = tsdistances.erp(x1, x2, gap_penalty=1.0, band=1.0, cached=False, n_jobs=4)
/// print(result)  # Output: [[18.0, 27.0], [9.0, 18.0]]
/// ```
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
/// Computes the pairwise Longest Common Subsequence (LCSS) distance between two sets of timeseries.
///
/// Given two sets of timeseries `x1` and `x2`, this function computes the LCSS distance
/// between each pair of timeseries (one from each set). The computation is parallelized across
/// multiple threads to improve performance. The number of threads used can be controlled
/// via the `n_jobs` parameter.
///
/// # Parameters
/// - `x1` (Vec<Vec<f64>>): The first set of timeseries with shape (n, m) where `n` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `x2` (Vec<Vec<f64>>): The second set of timeseries with shape (k, m) where `k` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `epsilon` (f64): The maximum distance between matching points in the LCSS calculation.
/// - `band` (f64): The Sakoe-Chiba band width, used to constrain the warping window.
/// - `cached` (bool): Whether to use cached computations if available. This can be used to speed up
///   repeated distance calculations with the same input data.
/// - `n_jobs` (i32): The number of threads to use for parallel computation. If set to `-1`,
///   all available CPU cores will be used.
///
/// # Returns
/// Vec<Vec<f64>>: A matrix of shape (n, k) where each element [i][j] represents the LCSS distance
/// between the i-th timeseries of `x1` and the j-th timeseries of `x2`.
///
/// # Examples
/// ```python
/// import tsdistances
///
/// x1 = [
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ]
///
/// x2 = [
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ]
///
/// # Calculate the distances using epsilon of 0.5 and 4 threads
/// result = tsdistances.lcss(x1, x2, epsilon=2.5, band=1.0, cached=False, n_jobs=4)
/// print(result)  # Output: [[0.6666, 0.6666], [0.3333, 0.6666]]
/// ```
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
/// Computes the pairwise Dynamic Time Warping (DTW) distance between two sets of timeseries.
///
/// Given two sets of timeseries `x1` and `x2`, this function computes the DTW distance
/// between each pair of timeseries (one from each set). The computation is parallelized across
/// multiple threads to improve performance. The number of threads used can be controlled
/// via the `n_jobs` parameter.
///
/// # Parameters
/// - `x1` (Vec<Vec<f64>>): The first set of timeseries with shape (n, m) where `n` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `x2` (Vec<Vec<f64>>): The second set of timeseries with shape (k, m) where `k` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `band` (f64): The Sakoe-Chiba band width, used to constrain the warping window.
/// - `cached` (bool): Whether to use cached computations if available. This can be used to speed up
///   repeated distance calculations with the same input data.
/// - `n_jobs` (i32): The number of threads to use for parallel computation. If set to `-1`,
///   all available CPU cores will be used.
///
/// # Returns
/// Vec<Vec<f64>>: A matrix of shape (n, k) where each element [i][j] represents the DTW distance
/// between the i-th timeseries of `x1` and the j-th timeseries of `x2`.
///
/// # Examples
/// ```python
/// import tsdistances
///
/// x1 = [
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ]
///
/// x2 = [
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ]
///
/// # Calculate the distances using a band of 1.0 and 4 threads
/// result = tsdistances.dtw(x1, x2, band=0.5, cached=False, n_jobs=4)
/// print(result)  # Output: [[108.0, 243.0], [26.0, 108.0]]
/// ```
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
/// Computes the pairwise Derivative Dynamic Time Warping (DDTW) distance between two sets of timeseries.
///
/// Given two sets of timeseries `x1` and `x2`, this function computes the DDTW distance
/// between each pair of timeseries (one from each set). The computation is parallelized across
/// multiple threads to improve performance. The number of threads used can be controlled
/// via the `n_jobs` parameter.
///
/// # Parameters
/// - `x1` (Vec<Vec<f64>>): The first set of timeseries with shape (n, m) where `n` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `x2` (Vec<Vec<f64>>): The second set of timeseries with shape (k, m) where `k` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `band` (f64): The Sakoe-Chiba band width, used to constrain the warping window.
/// - `cached` (bool): Whether to use cached computations if available. This can be used to speed up
///   repeated distance calculations with the same input data.
/// - `n_jobs` (i32): The number of threads to use for parallel computation. If set to `-1`,
///   all available CPU cores will be used.
///
/// # Returns
/// Vec<Vec<f64>>: A matrix of shape (n, k) where each element [i][j] represents the DDTW distance
/// between the i-th timeseries of `x1` and the j-th timeseries of `x2`.
///
/// # Examples
/// ```python
/// import tsdistances
///
/// x1 = [
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ]
///
/// x2 = [
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ]
///
/// # Calculate the distances using a band of 0.5 and 4 threads
/// result = tsdistances.ddtw(x1, x2, band=0.5, cached=False, n_jobs=4)
/// print(result)  # Output: [[0.0, 0.0], [0.0, 0.0]]
/// ```
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
/// Computes the pairwise Weighted Dynamic Time Warping (WDTW) distance between two sets of timeseries.
///
/// Given two sets of timeseries `x1` and `x2`, this function computes the WDTW distance
/// between each pair of timeseries (one from each set). The computation is parallelized across
/// multiple threads to improve performance. The number of threads used can be controlled
/// via the `n_jobs` parameter.
///
/// # Parameters
/// - `x1` (Vec<Vec<f64>>): The first set of timeseries with shape (n, m) where `n` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `x2` (Vec<Vec<f64>>): The second set of timeseries with shape (k, m) where `k` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `band` (f64): The Sakoe-Chiba band width, used to constrain the warping window.
/// - `cached` (bool): Whether to use cached computations if available. This can be used to speed up
///   repeated distance calculations with the same input data.
/// - `n_jobs` (i32): The number of threads to use for parallel computation. If set to `-1`,
///   all available CPU cores will be used.
///
/// # Returns
/// Vec<Vec<f64>>: A matrix of shape (n, k) where each element [i][j] represents the WDTW distance
/// between the i-th timeseries of `x1` and the j-th timeseries of `x2`.
///
/// # Examples
/// ```python
/// import tsdistances
///
/// x1 = [
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ]
///
/// x2 = [
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ]
///
/// # Calculate the distances using a band of 1.0 and 4 threads
/// result = tsdistances.wdtw(x1, x2, band=1.0, cached=False, n_jobs=4)
/// print(result)  # Output: [[51.9759, 116.9458], [12.6126, 51.9759]]
/// ```
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
/// Computes the pairwise Weighted Derivative Dynamic Time Warping (WDDTW) distance between two sets of timeseries.
///
/// Given two sets of timeseries `x1` and `x2`, this function computes the WDDTW distance
/// between each pair of timeseries (one from each set). The computation is parallelized across
/// multiple threads to improve performance. The number of threads used can be controlled
/// via the `n_jobs` parameter.
///
/// # Parameters
/// - `x1` (Vec<Vec<f64>>): The first set of timeseries with shape (n, m) where `n` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `x2` (Vec<Vec<f64>>): The second set of timeseries with shape (k, m) where `k` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `band` (f64): The Sakoe-Chiba band width, used to constrain the warping window.
/// - `cached` (bool): Whether to use cached computations if available. This can be used to speed up
///   repeated distance calculations with the same input data.
/// - `n_jobs` (i32): The number of threads to use for parallel computation. If set to `-1`,
///   all available CPU cores will be used.
///
/// # Returns
/// Vec<Vec<f64>>: A matrix of shape (n, k) where each element [i][j] represents the WDDTW distance
/// between the i-th timeseries of `x1` and the j-th timeseries of `x2`.
///
/// # Examples
/// ```python
/// import tsdistances
///
/// x1 = [
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ]
///
/// x2 = [
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ]
///
/// # Calculate the distances using a band of 0.5 and 4 threads
/// result = tsdistances.wddtw(x1, x2, band=1.0, cached=False, n_jobs=4)
/// print(result)  # Output: [[0.0, 0.0], [0.0, 0.0]]
/// ```
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
/// Computes the pairwise Move-Split-Merge (MSM) distance between two sets of timeseries.
///
/// Given two sets of timeseries `x1` and `x2`, this function computes the MSM distance
/// between each pair of timeseries (one from each set). The computation is parallelized across
/// multiple threads to improve performance. The number of threads used can be controlled
/// via the `n_jobs` parameter.
///
/// # Parameters
/// - `x1` (Vec<Vec<f64>>): The first set of timeseries with shape (n, m) where `n` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `x2` (Vec<Vec<f64>>): The second set of timeseries with shape (k, m) where `k` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `band` (f64): The Sakoe-Chiba band width, used to constrain the warping window.
/// - `cached` (bool): Whether to use cached computations if available. This can be used to speed up
///   repeated distance calculations with the same input data.
/// - `n_jobs` (i32): The number of threads to use for parallel computation. If set to `-1`,
///   all available CPU cores will be used.
///
/// # Returns
/// Vec<Vec<f64>>: A matrix of shape (n, k) where each element [i][j] represents the MSM distance
/// between the i-th timeseries of `x1` and the j-th timeseries of `x2`.
///
/// # Examples
/// ```python
/// import tsdistances
///
/// x1 = [
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ]
///
/// x2 = [
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ]
///
/// # Calculate the distances using a band of 1.0 and 4 threads
/// result = tsdistances.msm(x1, x2, band=1.0, cached=False, n_jobs=4)
/// print(result)  # Output: [[12.0, 15.0], [8.0, 12.0]]
/// ```
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
/// Computes the pairwise Time Warp Edit (TWE) distance between two sets of timeseries.
///
/// Given two sets of timeseries `x1` and `x2`, this function computes the TWE distance
/// between each pair of timeseries (one from each set). The computation is parallelized across
/// multiple threads to improve performance. The number of threads used can be controlled
/// via the `n_jobs` parameter.
///
/// # Parameters
/// - `x1` (Vec<Vec<f64>>): The first set of timeseries with shape (n, m) where `n` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `x2` (Vec<Vec<f64>>): The second set of timeseries with shape (k, m) where `k` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `stiffness` (f64): The stiffness parameter for TWE.
/// - `penalty` (f64): The penalty for time gaps.
/// - `band` (f64): The Sakoe-Chiba band width, used to constrain the warping window.
/// - `cached` (bool): Whether to use cached computations if available. This can be used to speed up
///   repeated distance calculations with the same input data.
/// - `n_jobs` (i32): The number of threads to use for parallel computation. If set to `-1`,
///   all available CPU cores will be used.
///
/// # Returns
/// Vec<Vec<f64>>: A matrix of shape (n, k) where each element [i][j] represents the TWE distance
/// between the i-th timeseries of `x1` and the j-th timeseries of `x2`.
///
/// # Examples
/// ```python
/// import tsdistances
///
/// x1 = [
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ]
///
/// x2 = [
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ]
///
/// # Calculate the distances using stiffness of 1.0, penalty of 0.5, and 4 threads
/// result = tsdistances.twe(x1, x2, stiffness=1.0, penalty=0.5, band=1.0, cached=False, n_jobs=4)
/// print(result)  # Output: [[30.0, 45.0], [15.0, 30.0]]
/// ```
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
/// Computes the pairwise Amercing Dynamic Time Warping (ADTW) distance between two sets of timeseries.
///
/// Given two sets of timeseries `x1` and `x2`, this function computes the ADTW distance
/// between each pair of timeseries (one from each set). The computation is parallelized across
/// multiple threads to improve performance. The number of threads used can be controlled
/// via the `n_jobs` parameter.
///
/// # Parameters
/// - `x1` (Vec<Vec<f64>>): The first set of timeseries with shape (n, m) where `n` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `x2` (Vec<Vec<f64>>): The second set of timeseries with shape (k, m) where `k` is the number
///   of timeseries and `m` is the length of each timeseries.
/// - `w` (f64): The weight used in the ADTW computation.
/// - `band` (f64): The Sakoe-Chiba band width, used to constrain the warping window.
/// - `cached` (bool): Whether to use cached computations if available. This can be used to speed up
///   repeated distance calculations with the same input data.
/// - `n_jobs` (i32): The number of threads to use for parallel computation. If set to `-1`,
///   all available CPU cores will be used.
///
/// # Returns
/// Vec<Vec<f64>>: A matrix of shape (n, k) where each element [i][j] represents the ADTW distance
/// between the i-th timeseries of `x1` and the j-th timeseries of `x2`.
///
/// # Examples
/// ```python
/// import tsdistances
///
/// x1 = [
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ]
///
/// x2 = [
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ]
///
/// # Calculate the distances using a weight of 0.5, a band of 1.0, and 4 threads
/// result = tsdistances.adtw(x1, x2, w=0.5, band=1.0, cached=False, n_jobs=4)
/// print(result)  # Output: [[108.0, 243.0], [27.0, 108.0]]
/// ```
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

// k-Shape: Efficient and Accurate Clustering of Time Series - Paparrizos J. et al., 2015
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

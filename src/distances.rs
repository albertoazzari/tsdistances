#![allow(dead_code)]
use std::cmp::max;

use rayon::prelude::*;
use pyo3::prelude::*;
use crate::elastic_distances;

const MIN_CHUNK_SIZE: usize = 16;
const CHUNKS_PER_THREAD: usize = 8;

#[pyfunction]
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
pub fn euclidean(x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>, cached: bool, n_jobs: i32) -> Vec<Vec<f64>> {
    let n_jobs = if n_jobs == -1 {
        rayon::current_num_threads() as usize
    } else {
        n_jobs.max(1) as usize
    };

    let semaphore = parking_lot::Mutex::new(n_jobs);
    let cond_var = parking_lot::Condvar::new();

    x1.par_chunks(max(MIN_CHUNK_SIZE, x1.len() / n_jobs / CHUNKS_PER_THREAD))
        .map(|a| {
            let mut guard = semaphore.lock();
            while *guard == 0 {
                cond_var.wait(&mut guard);
            }
            *guard -= 1;
            drop(guard);
            
            let result = a.iter().map(|a| {
                x2.iter()
                .map(|b| elastic_distances::euclidean(a, b, cached))
                .collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        }).flatten().collect::<Vec<_>>()
}

#[pyfunction]
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
pub fn erp(x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>, gap_penalty: f64, band: f64, cached: bool, n_jobs: i32) -> Vec<Vec<f64>> {
    let n_jobs = if n_jobs == -1 {
        rayon::current_num_threads() as usize
    } else {
        n_jobs.max(1) as usize
    };

    let semaphore = parking_lot::Mutex::new(n_jobs);
    let cond_var = parking_lot::Condvar::new();

    x1.par_chunks(max(MIN_CHUNK_SIZE, x1.len() / n_jobs / CHUNKS_PER_THREAD))
        .map(|a| {
            let mut guard = semaphore.lock();
            while *guard == 0 {
                cond_var.wait(&mut guard);
            }
            *guard -= 1;
            drop(guard);
            
            let result = a.iter().map(|a| {
                x2.iter()
                .map(|b| elastic_distances::erp(a, b, gap_penalty, band, cached))
                .collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        }).flatten().collect::<Vec<_>>()
}

#[pyfunction]
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
pub fn lcss(x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>, epsilon: f64, band: f64, cached: bool, n_jobs: i32) -> Vec<Vec<f64>> {
    let n_jobs = if n_jobs == -1 {
        rayon::current_num_threads() as usize
    } else {
        n_jobs.max(1) as usize
    };

    let semaphore = parking_lot::Mutex::new(n_jobs);
    let cond_var = parking_lot::Condvar::new();

    x1.par_chunks(max(MIN_CHUNK_SIZE, x1.len() / n_jobs / CHUNKS_PER_THREAD))
        .map(|a| {
            let mut guard = semaphore.lock();
            while *guard == 0 {
                cond_var.wait(&mut guard);
            }
            *guard -= 1;
            drop(guard);
            
            let result = a.iter().map(|a| {
                x2.iter()
                .map(|b| elastic_distances::lcss(a, b, epsilon, band, cached))
                .collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        }).flatten().collect::<Vec<_>>()
}

#[pyfunction]
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
pub fn twe(x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>, stiffness: f64, penalty: f64, band: f64, cached: bool, n_jobs: i32) -> Vec<Vec<f64>> {
    let n_jobs = if n_jobs == -1 {
        rayon::current_num_threads() as usize
    } else {
        n_jobs.max(1) as usize
    };

    let semaphore = parking_lot::Mutex::new(n_jobs);
    let cond_var = parking_lot::Condvar::new();

    x1.par_chunks(max(MIN_CHUNK_SIZE, x1.len() / n_jobs / CHUNKS_PER_THREAD))
        .map(|a| {
            let mut guard = semaphore.lock();
            while *guard == 0 {
                cond_var.wait(&mut guard);
            }
            *guard -= 1;
            drop(guard);
            
            let result = a.iter().map(|a| {
                x2.iter()
                .map(|b| elastic_distances::twe(a, b, stiffness, penalty, band, cached))
                .collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        }).flatten().collect::<Vec<_>>()
}

#[pyfunction]
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
pub fn dtw(x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>, band: f64, cached: bool, n_jobs: i32) -> Vec<Vec<f64>> {
    let n_jobs = if n_jobs == -1 {
        rayon::current_num_threads() as usize
    } else {
        n_jobs.max(1) as usize
    };

    let semaphore = parking_lot::Mutex::new(n_jobs);
    let cond_var = parking_lot::Condvar::new();

    x1.par_chunks(max(MIN_CHUNK_SIZE, x1.len() / n_jobs / CHUNKS_PER_THREAD))
        .map(|a| {
            let mut guard = semaphore.lock();
            while *guard == 0 {
                cond_var.wait(&mut guard);
            }
            *guard -= 1;
            drop(guard);
            
            let result = a.iter().map(|a| {
                x2.iter()
                .map(|b| elastic_distances::dtw(a, b, band, cached))
                .collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        }).flatten().collect::<Vec<_>>()
    
}

#[pyfunction]
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
pub fn ddtw(x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>, band: f64, cached: bool, n_jobs: i32) -> Vec<Vec<f64>> {
    let n_jobs = if n_jobs == -1 {
        rayon::current_num_threads() as usize
    } else {
        n_jobs.max(1) as usize
    };

    let semaphore = parking_lot::Mutex::new(n_jobs);
    let cond_var = parking_lot::Condvar::new();

    x1.par_chunks(max(MIN_CHUNK_SIZE, x1.len() / n_jobs / CHUNKS_PER_THREAD))
        .map(|a| {
            let mut guard = semaphore.lock();
            while *guard == 0 {
                cond_var.wait(&mut guard);
            }
            *guard -= 1;
            drop(guard);
            
            let result = a.iter().map(|a| {
                x2.iter()
                .map(|b| elastic_distances::ddtw(a, b, band, cached))
                .collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        }).flatten().collect::<Vec<_>>()
    
}

#[pyfunction]
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
pub fn wdtw(x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>, band: f64, cached: bool, n_jobs: i32) -> Vec<Vec<f64>> {
    let n_jobs = if n_jobs == -1 {
        rayon::current_num_threads() as usize
    } else {
        n_jobs.max(1) as usize
    };

    let semaphore = parking_lot::Mutex::new(n_jobs);
    let cond_var = parking_lot::Condvar::new();

    x1.par_chunks(max(MIN_CHUNK_SIZE, x1.len() / n_jobs / CHUNKS_PER_THREAD))
        .map(|a| {
            let mut guard = semaphore.lock();
            while *guard == 0 {
                cond_var.wait(&mut guard);
            }
            *guard -= 1;
            drop(guard);
            
            let result = a.iter().map(|a| {
                x2.iter()
                .map(|b| elastic_distances::wdtw(a, b, band, cached))
                .collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        }).flatten().collect::<Vec<_>>()
    
}

#[pyfunction]
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
pub fn wddtw(x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>, band: f64, cached: bool, n_jobs: i32) -> Vec<Vec<f64>> {
    let n_jobs = if n_jobs == -1 {
        rayon::current_num_threads() as usize
    } else {
        n_jobs.max(1) as usize
    };

    let semaphore = parking_lot::Mutex::new(n_jobs);
    let cond_var = parking_lot::Condvar::new();

    x1.par_chunks(max(MIN_CHUNK_SIZE, x1.len() / n_jobs / CHUNKS_PER_THREAD))
        .map(|a| {
            let mut guard = semaphore.lock();
            while *guard == 0 {
                cond_var.wait(&mut guard);
            }
            *guard -= 1;
            drop(guard);
            
            let result = a.iter().map(|a| {
                x2.iter()
                .map(|b| elastic_distances::wddtw(a, b, band, cached))
                .collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        }).flatten().collect::<Vec<_>>()
    
}

#[pyfunction]
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
pub fn msm(x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>, band: f64, cached: bool, n_jobs: i32) -> Vec<Vec<f64>> {
    let n_jobs = if n_jobs == -1 {
        rayon::current_num_threads() as usize
    } else {
        n_jobs.max(1) as usize
    };

    let semaphore = parking_lot::Mutex::new(n_jobs);
    let cond_var = parking_lot::Condvar::new();

    x1.par_chunks(max(MIN_CHUNK_SIZE, x1.len() / n_jobs / CHUNKS_PER_THREAD))
        .map(|a| {
            let mut guard = semaphore.lock();
            while *guard == 0 {
                cond_var.wait(&mut guard);
            }
            *guard -= 1;
            drop(guard);
            
            let result = a.iter().map(|a| {
                x2.iter()
                .map(|b| elastic_distances::msm(a, b, band, cached))
                .collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        }).flatten().collect::<Vec<_>>()
}

#[pyfunction]
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
pub fn adtw(x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>, w:f64, band: f64, cached: bool, n_jobs: i32) -> Vec<Vec<f64>> {
    let n_jobs = if n_jobs == -1 {
        rayon::current_num_threads() as usize
    } else {
        n_jobs.max(1) as usize
    };

    let semaphore = parking_lot::Mutex::new(n_jobs);
    let cond_var = parking_lot::Condvar::new();

    x1.par_chunks(max(MIN_CHUNK_SIZE, x1.len() / n_jobs / CHUNKS_PER_THREAD))
        .map(|a| {
            let mut guard = semaphore.lock();
            while *guard == 0 {
                cond_var.wait(&mut guard);
            }
            *guard -= 1;
            drop(guard);
            
            let result = a.iter().map(|a| {
                x2.iter()
                .map(|b| elastic_distances::adtw(a, b, w, band, cached))
                .collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        }).flatten().collect::<Vec<_>>()
}

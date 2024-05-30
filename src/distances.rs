#![allow(dead_code)]
use std::cmp::max;

use rayon::prelude::*;
use pyo3::prelude::*;
use crate::elastic_distances;

const MIN_CHUNK_SIZE: usize = 16;
const CHUNKS_PER_THREAD: usize = 8;

#[pyfunction]
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
pub fn twe(x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>, nu: f64, lambda: f64, band: f64, cached: bool, n_jobs: i32) -> Vec<Vec<f64>> {
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
                .map(|b| elastic_distances::twe(a, b, nu, lambda, band, cached))
                .collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let mut guard = semaphore.lock();
            *guard += 1;
            cond_var.notify_one();
            result
        }).flatten().collect::<Vec<_>>()
}

#[pyfunction]
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

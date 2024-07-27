import pytest
import time
import numpy as np
from tsdistances import (
    erp_distance,
    lcss_distance,
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
    wddtw_distance,
    adtw_distance,
    msm_distance,
    twe_distance,
)
from aeon import distances as aeon


def load_random_dataset():
    n_timeseries = 10
    n_timesteps = 10000

    X_train = np.random.rand(n_timeseries, n_timesteps)
    y_train = np.random.randint(0, 10, n_timeseries)

    X_test = np.random.rand(n_timeseries, n_timesteps)
    y_test = np.random.randint(0, 10, n_timeseries)

    return np.vstack((X_train, X_test)), np.hstack((y_train, y_test))


def assert_running_times(gpu_time, cpu_time):
    print(gpu_time, cpu_time)
    assert gpu_time <= cpu_time


X, y = load_random_dataset()
band = 1.0

def test_erp_distance():
    gap_penalty = 0.0

    gpu_time = time.time()
    erp_distance(X, None, band=band, gap_penalty=gap_penalty, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time
    
    cpu_time = time.time()
    erp_distance(X, None, band=band, gap_penalty=gap_penalty, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time
    
    assert_running_times(gpu_time, cpu_time)


def test_lcss_distance():
    epsilon = 0.1
    gpu_time = time.time()
    lcss_distance(X, None, band=band, epsilon=epsilon, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time

    cpu_time = time.time()
    lcss_distance(X, None, band=band, epsilon=epsilon, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time

    assert_running_times(gpu_time, cpu_time)


def test_dtw_distance():
    gpu_time = time.time()
    dtw_distance(X, None, band=band, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time

    cpu_time = time.time()
    dtw_distance(X, None, band=band, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time

    assert_running_times(gpu_time, cpu_time)


def test_ddtw_distance():
    gpu_time = time.time()
    ddtw_distance(X, None, band=band, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time

    cpu_time = time.time()
    ddtw_distance(X, None, band=band, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time

    assert_running_times(gpu_time, cpu_time)


def test_wdtw_distance():
    g = 0.05
    gpu_time = time.time()
    wdtw_distance(X, None, g=g, band=band, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time

    cpu_time = time.time()
    wdtw_distance(X, None, g=g, band=band, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time

    assert_running_times(gpu_time, cpu_time)


def test_wddtw_distance():
    g = 0.05
    gpu_time = time.time()
    wddtw_distance(X, None, g=g, band=band, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time

    cpu_time = time.time()
    wddtw_distance(X, None, g=g, band=band, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time
    assert_running_times(gpu_time, cpu_time)


def test_adtw_distance():
    warp_penalty = 1.0
    gpu_time = time.time()
    adtw_distance(X, None, band=band, warp_penalty=warp_penalty, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time

    cpu_time = time.time()
    adtw_distance(X, None, band=band, warp_penalty=warp_penalty, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time

    assert_running_times(gpu_time, cpu_time)


def test_msm_distance():
    gpu_time = time.time()
    msm_distance(X, None, band=band, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time

    cpu_time = time.time()
    msm_distance(X, None, band=band, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time

    assert_running_times(gpu_time, cpu_time)


def test_twe_distance():
    stifness = 0.1
    penalty = 0.1
    gpu_time = time.time()
    twe_distance(X, None, band=band, stifness=stifness, penalty=penalty, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time

    cpu_time = time.time()
    twe_distance(X, None, band=band, stifness=stifness, penalty=penalty, n_jobs=1, device="cpu")
    cpu_time = time.time() - cpu_time

    assert_running_times(gpu_time, cpu_time)

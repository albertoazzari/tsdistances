import pytest
import time
import numpy as np
from tsdistances import (
    euclidean_distance, erp_distance, lcss_distance, dtw_distance, ddtw_distance, 
    wdtw_distance, wddtw_distance, adtw_distance, msm_distance, twe_distance, sb_distance
)
from aeon import distances as aeon

def load_random_dataset():
    n_timeseries = 2
    n_timesteps = 10

    X_train = np.random.rand(n_timeseries, n_timesteps)
    y_train = np.random.randint(0, 10, n_timeseries)

    X_test = np.random.rand(n_timeseries, n_timesteps)
    y_test = np.random.randint(0, 10, n_timeseries)

    return np.vstack((X_train, X_test)), np.hstack((y_train, y_test))

def assert_running_times(gpu_time, cpu_time):
    print(gpu_time, cpu_time)
    assert gpu_time <= cpu_time

X, y = load_random_dataset()

def test_erp_distance():
    gpu_time = time.time()
    gap_penalty = 0.0
    sakoe_chiba_band = 0.5
    D = erp_distance(X, None, sakoe_chiba_band=sakoe_chiba_band, gap_penalty=gap_penalty, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time
    cpu_time = time.time()
    D = erp_distance(X, None, sakoe_chiba_band=sakoe_chiba_band, gap_penalty=gap_penalty, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time
    assert_running_times(gpu_time, cpu_time)

def test_lcss_distance():
    gpu_time = time.time()
    D = lcss_distance(X, None, sakoe_chiba_band=0.1, epsilon=0.1, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time
    cpu_time = time.time()
    D = lcss_distance(X, None, sakoe_chiba_band=0.1, epsilon=0.1, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time
    assert_running_times(gpu_time, cpu_time)

def test_dtw_distance():
    gpu_time = time.time()
    D = dtw_distance(X, None, sakoe_chiba_band=0.1, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time
    cpu_time = time.time()
    D = dtw_distance(X, None, sakoe_chiba_band=0.1, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time
    assert_running_times(gpu_time, cpu_time)

def test_ddtw_distance():
    gpu_time = time.time()
    D = ddtw_distance(X, None, sakoe_chiba_band=0.1, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time
    cpu_time = time.time()
    D = ddtw_distance(X, None, sakoe_chiba_band=0.1, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time
    assert_running_times(gpu_time, cpu_time)

def test_wdtw_distance():
    gpu_time = time.time()
    D = wdtw_distance(X, None, sakoe_chiba_band=0.1, g=0.05, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time
    cpu_time = time.time()
    D = wdtw_distance(X, None, sakoe_chiba_band=0.1, g=0.05, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time
    assert_running_times(gpu_time, cpu_time)

def test_wddtw_distance():
    gpu_time = time.time()
    D = wddtw_distance(X, None, sakoe_chiba_band=0.1, g=0.05, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time
    cpu_time = time.time()
    D = wddtw_distance(X, None, sakoe_chiba_band=0.1, g=0.05, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time
    assert_running_times(gpu_time, cpu_time)

def test_adtw_distance():
    gpu_time = time.time()
    D = adtw_distance(X, None, sakoe_chiba_band=0.1, warp_penalty=1.0, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time
    cpu_time = time.time()
    D = adtw_distance(X, None, sakoe_chiba_band=0.1, warp_penalty=1.0, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time
    assert_running_times(gpu_time, cpu_time)

def test_msm_distance():
    gpu_time = time.time()
    D = msm_distance(X, None, sakoe_chiba_band=0.1, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time
    cpu_time = time.time()
    D = msm_distance(X, None, sakoe_chiba_band=0.1, n_jobs=-1, device="cpu")
    cpu_time = time.time() - cpu_time
    assert_running_times(gpu_time, cpu_time)

def test_twe_distance():
    gpu_time = time.time()
    D = twe_distance(X, None, sakoe_chiba_band=0.1, stifness=0.1, penalty=0.1, n_jobs=1, device="gpu")
    gpu_time = time.time() - gpu_time
    cpu_time = time.time()
    D = twe_distance(X, None, sakoe_chiba_band=0.1, stifness=0.1, penalty=0.1, n_jobs=1, device="cpu")
    cpu_time = time.time() - cpu_time
    assert_running_times(gpu_time, cpu_time)

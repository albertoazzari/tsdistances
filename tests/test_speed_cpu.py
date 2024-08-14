import pytest
import time
import numpy as np
from tsdistances import (
    euclidean_distance,
    erp_distance,
    lcss_distance,
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
    wddtw_distance,
    adtw_distance,
    msm_distance,
    twe_distance,
    sb_distance,
)
from aeon import distances as aeon


def load_random_dataset():
    n_timeseries = 100
    n_timesteps = 100

    X_train = np.random.rand(n_timeseries, n_timesteps)
    y_train = np.random.randint(0, 10, n_timeseries)

    X_test = np.random.rand(n_timeseries, n_timesteps)
    y_test = np.random.randint(0, 10, n_timeseries)

    return np.vstack((X_train, X_test)), np.hstack((y_train, y_test))


def assert_running_times(tsd_time, aeon_time):
    print(tsd_time, aeon_time)
    assert tsd_time <= aeon_time


X, y = load_random_dataset()
band = 0.5


def test_euclidean_distance():
    tsd_times = []
    aeon_times = []
    repeat = 1000
    for i in range(repeat):

        tsd_time = time.time()
        euclidean_distance(X, None, n_jobs=1)
        tsd_time = time.time() - tsd_time
        tsd_times.append(tsd_time)

        aeon_time = time.time()
        aeon.euclidean_pairwise_distance(X)
        aeon_time = time.time() - aeon_time
        aeon_times.append(aeon_time)

    assert_running_times(sum(tsd_times) / repeat, sum(aeon_times) / repeat)


def test_erp_distance():
    tsd_time = time.time()
    gap_penalty = 0.0
    D = erp_distance(
        X, None, band=band, gap_penalty=gap_penalty, n_jobs=1
    )
    tsd_time = time.time() - tsd_time
    aeon_time = time.time()
    aeon_D = aeon.erp_pairwise_distance(X, g=gap_penalty, window=band)
    aeon_time = time.time() - aeon_time
    assert_running_times(tsd_time, aeon_time)


def test_lcss_distance():
    tsd_time = time.time()
    D = lcss_distance(X, None, band=band, epsilon=0.1, n_jobs=1)
    tsd_time = time.time() - tsd_time
    aeon_time = time.time()
    aeon_D = aeon.lcss_pairwise_distance(X, window=band, epsilon=0.1)
    aeon_time = time.time() - aeon_time
    assert_running_times(tsd_time, aeon_time)


def test_dtw_distance():
    tsd_time = time.time()
    D = dtw_distance(X, None, band=band, n_jobs=1)
    tsd_time = time.time() - tsd_time
    aeon_time = time.time()
    aeon_D = aeon.dtw_pairwise_distance(X, window=band)
    aeon_time = time.time() - aeon_time
    assert_running_times(tsd_time, aeon_time)


def test_ddtw_distance():
    tsd_time = time.time()
    D = ddtw_distance(X, None, band=band, n_jobs=1)
    tsd_time = time.time() - tsd_time
    aeon_time = time.time()
    aeon_D = aeon.ddtw_pairwise_distance(X, window=band)
    aeon_time = time.time() - aeon_time
    assert_running_times(tsd_time, aeon_time)


def test_wdtw_distance():
    tsd_time = time.time()
    D = wdtw_distance(X, None, band=band, g=0.05, n_jobs=1)
    tsd_time = time.time() - tsd_time
    aeon_time = time.time()
    aeon_D = aeon.wdtw_pairwise_distance(X, window=band, g=0.05)
    aeon_time = time.time() - aeon_time
    assert_running_times(tsd_time, aeon_time)


def test_wddtw_distance():
    tsd_time = time.time()
    D = wddtw_distance(X, None, band=band, g=0.05, n_jobs=1)
    tsd_time = time.time() - tsd_time
    aeon_time = time.time()
    aeon_D = aeon.wddtw_pairwise_distance(X, window=band, g=0.05)
    aeon_time = time.time() - aeon_time
    assert_running_times(tsd_time, aeon_time)


def test_adtw_distance():
    tsd_time = time.time()
    D = adtw_distance(X, None, band=band, warp_penalty=1.0, n_jobs=1)
    tsd_time = time.time() - tsd_time
    aeon_time = time.time()
    aeon_D = aeon.adtw_pairwise_distance(X, window=band, warp_penalty=1.0)
    aeon_time = time.time() - aeon_time
    assert_running_times(tsd_time, aeon_time)


def test_msm_distance():
    tsd_times = []
    aeon_times = []
    repeat = 100
    for i in range(repeat):
        tsd_time = time.time()
        D = msm_distance(X, None, band=band, n_jobs=1)
        tsd_time = time.time() - tsd_time
        aeon_time = time.time()
        aeon_D = aeon.msm_pairwise_distance(X, window=band)
        aeon_time = time.time() - aeon_time
        tsd_times.append(tsd_time)
        aeon_times.append(aeon_time)
    assert_running_times(sum(tsd_times)/len(tsd_times), sum(aeon_times)/len(aeon_times))


def test_twe_distance():
    tsd_time = time.time()
    D = twe_distance(X, None, band=band, stifness=0.1, penalty=0.1, n_jobs=1)
    tsd_time = time.time() - tsd_time
    aeon_time = time.time()
    aeon_D = aeon.twe_pairwise_distance(X, nu=0.1, lmbda=0.1, window=band)
    aeon_time = time.time() - aeon_time
    assert_running_times(tsd_time, aeon_time)


def test_sbd_distance():
    tsd_time = time.time()
    D = sb_distance(X, None, n_jobs=1)
    tsd_time = time.time() - tsd_time
    aeon_time = time.time()
    aeon_D = aeon.sbd_pairwise_distance(X)
    aeon_time = time.time() - aeon_time
    assert_running_times(tsd_time, aeon_time)

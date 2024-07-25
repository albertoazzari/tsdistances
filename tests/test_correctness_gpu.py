import pytest
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
    n_timeseries = 2
    n_timesteps = 10

    X_train = np.random.rand(n_timeseries, n_timesteps)
    y_train = np.random.randint(0, 10, n_timeseries)

    X_test = np.random.rand(n_timeseries, n_timesteps)
    y_test = np.random.randint(0, 10, n_timeseries)

    return np.vstack((X_train, X_test)), np.hstack((y_train, y_test))


X, y = load_random_dataset()


def check_distance_matrix(D, X):
    assert np.allclose(D, D.T, atol=1e-8)
    assert np.allclose(np.diag(D), np.zeros(X.shape[0]), atol=1e-8)
    assert np.all(D >= 0)
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i == j:
                assert np.isclose(D[i, j], 0, atol=1e-8)
            else:
                assert D[i, j] > 0


def test_erp_distance():
    gap_penalty = 0.0
    D = erp_distance(X, None, gap_penalty=gap_penalty, n_jobs=1, device="gpu")
    D = np.array(D)
    check_distance_matrix(D, X)
    aeon_D = aeon.erp_pairwise_distance(X, g=gap_penalty)
    assert np.allclose(D, aeon_D, atol=1e-2)


def test_lcss_distance():
    D = lcss_distance(X, None, epsilon=0.1, n_jobs=1, device="gpu")
    D = np.array(D)
    check_distance_matrix(D, X)
    aeon_D = aeon.lcss_pairwise_distance(X, epsilon=0.1)
    assert np.allclose(D, aeon_D, atol=1e-2)


def test_dtw_distance():
    D = dtw_distance(X, None, n_jobs=1, device="gpu")
    D = np.array(D)
    check_distance_matrix(D, X)
    aeon_D = aeon.dtw_pairwise_distance(X)
    assert np.allclose(D, aeon_D, atol=1e-2)


def test_ddtw_distance():
    D = ddtw_distance(X, None, n_jobs=1, device="gpu")
    D = np.array(D)
    check_distance_matrix(D, X)
    aeon_D = aeon.ddtw_pairwise_distance(X)
    assert np.allclose(D, aeon_D, atol=1e-2)


def test_wdtw_distance():
    D = wdtw_distance(X, None, g=0.05, n_jobs=1, device="gpu")
    D = np.array(D)
    check_distance_matrix(D, X)
    aeon_D = aeon.wdtw_pairwise_distance(X, g=0.05)
    assert np.allclose(D, aeon_D, atol=1e-2)


def test_wddtw_distance():
    D = wddtw_distance(X, None, g=0.05, n_jobs=1, device="gpu")
    D = np.array(D)
    check_distance_matrix(D, X)
    aeon_D = aeon.wddtw_pairwise_distance(X, g=0.05)
    assert np.allclose(D, aeon_D, atol=1e-2)


def test_adtw_distance():
    D = adtw_distance(X, None, warp_penalty=1.0, n_jobs=1, device="gpu")
    D = np.array(D)
    check_distance_matrix(D, X)
    aeon_D = aeon.adtw_pairwise_distance(X, warp_penalty=1.0)
    assert np.allclose(D, aeon_D, atol=1e-2)


def test_msm_distance():
    D = msm_distance(X, None, n_jobs=1, device="gpu")
    D = np.array(D)
    check_distance_matrix(D, X)
    aeon_D = aeon.msm_pairwise_distance(X)
    assert np.allclose(D, aeon_D, atol=1e-2)


def test_twe_distance():
    D = twe_distance(X, None, stifness=0.1, penalty=0.1, n_jobs=1, device="gpu")
    D = np.array(D)
    check_distance_matrix(D, X)
    aeon_D = aeon.twe_pairwise_distance(X, nu=0.1, lmbda=0.1)
    assert np.allclose(D, aeon_D, atol=1e-2)

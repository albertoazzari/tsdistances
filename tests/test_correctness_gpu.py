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
    n_timesteps = 100

    X_train = np.random.rand(n_timeseries, n_timesteps)
    y_train = np.random.randint(0, 10, n_timeseries)

    X_test = np.random.rand(n_timeseries, n_timesteps)
    y_test = np.random.randint(0, 10, n_timeseries)

    return np.vstack((X_train, X_test)), np.hstack((y_train, y_test))


X, y = load_random_dataset()
band = 1.0


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
    D_cpu = erp_distance(X, None, gap_penalty=gap_penalty, band=band, n_jobs=1, device='cpu')
    check_distance_matrix(D_cpu, X)
    D_gpu = erp_distance(X, None, gap_penalty=gap_penalty, band=band, n_jobs=1, device='gpu')
    assert np.allclose(D_cpu, D_gpu, atol=1e-8)


def test_lcss_distance():
    epsilon = 0.1
    D_cpu = lcss_distance(X, None, epsilon=epsilon, band=band, n_jobs=1, device='cpu')
    check_distance_matrix(D_cpu, X)
    D_gpu = lcss_distance(X, None, epsilon=epsilon, band=band, n_jobs=1, device='gpu')
    assert np.allclose(D_cpu, D_gpu, atol=1e-8)


def test_dtw_distance():
    D_cpu = dtw_distance(X, None, band=band, n_jobs=1, device='cpu')
    check_distance_matrix(D_cpu, X)
    D_gpu = dtw_distance(X, None, band=band, n_jobs=1, device='gpu')
    assert np.allclose(D_cpu, D_gpu, atol=1e-8)


def test_ddtw_distance():
    D_cpu = ddtw_distance(X, None, band=band, n_jobs=1, device='cpu')
    check_distance_matrix(D_cpu, X)
    D_gpu = ddtw_distance(X, None, band=band, n_jobs=1, device='gpu')
    assert np.allclose(D_cpu, D_gpu, atol=1e-8)


def test_wdtw_distance():
    g = 0.05
    D_cpu = wdtw_distance(X, None, g=g, band=band, n_jobs=1, device='cpu')
    check_distance_matrix(D_cpu, X)
    D_gpu = wdtw_distance(X, None, g=g, band=band, n_jobs=1, device='gpu')
    assert np.allclose(D_cpu, D_gpu, atol=1e-8)


def test_wddtw_distance():
    g = 0.05
    D_cpu = wddtw_distance(X, None, g=g, band=band, n_jobs=1, device='cpu')
    check_distance_matrix(D_cpu, X)
    D_gpu = wddtw_distance(X, None, g=g, band=band, n_jobs=1, device='gpu')
    assert np.allclose(D_cpu, D_gpu, atol=1e-8)


def test_adtw_distance():
    warp_penalty = 1.0
    D_cpu = adtw_distance(X, None, band=band, warp_penalty=warp_penalty, n_jobs=1, device='cpu')
    check_distance_matrix(D_cpu, X)
    D_gpu = adtw_distance(X, None, band=band, warp_penalty=warp_penalty, n_jobs=1, device='gpu')
    assert np.allclose(D_cpu, D_gpu, atol=1e-8)


def test_msm_distance():
    D_cpu = msm_distance(X, None, band=band, n_jobs=1, device='cpu')
    check_distance_matrix(D_cpu, X)
    D_gpu = msm_distance(X, None, band=band, n_jobs=1, device='gpu')
    assert np.allclose(D_cpu, D_gpu, atol=1e-8)


def test_twe_distance():
    stiffness = 0.1
    penalty = 0.1
    D_cpu = twe_distance(X, None, band=band, stifness=stiffness, penalty=penalty, n_jobs=1, device='cpu')
    check_distance_matrix(D_cpu, X)
    D_gpu = twe_distance(X, None, band=band, stifness=stiffness, penalty=penalty, n_jobs=1, device='gpu')
    assert np.allclose(D_cpu, D_gpu, atol=1e-8)
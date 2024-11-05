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
    mp_distance
)
from aeon import distances as aeon
import stumpy


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


def test_euclidean_distance():
    D = euclidean_distance(X, None, n_jobs=1)
    check_distance_matrix(D, X)
    aeon_D = aeon.euclidean_pairwise_distance(X)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_erp_distance():
    gap_penalty = 0.0
    D = erp_distance(X, None, gap_penalty=gap_penalty, band=band, n_jobs=1)
    check_distance_matrix(D, X)
    aeon_D = aeon.erp_pairwise_distance(X, g=gap_penalty, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_lcss_distance():
    epsilon = 0.1
    D = lcss_distance(X, None, epsilon=epsilon, band=band, n_jobs=1)
    check_distance_matrix(D, X)
    aeon_D = aeon.lcss_pairwise_distance(X, epsilon=epsilon, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_dtw_distance():
    D = dtw_distance(X, None, band=band, n_jobs=1)
    check_distance_matrix(D, X)
    aeon_D = aeon.dtw_pairwise_distance(X, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_ddtw_distance():
    D = ddtw_distance(X, None, band=band, n_jobs=1)
    check_distance_matrix(D, X)
    aeon_D = aeon.ddtw_pairwise_distance(X, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_wdtw_distance():
    g = 0.05
    D = wdtw_distance(X, None, g=g, band=band, n_jobs=1)
    check_distance_matrix(D, X)
    aeon_D = aeon.wdtw_pairwise_distance(X, g=g, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_wddtw_distance():
    g = 0.05
    D = wddtw_distance(X, None, g=g, band=band, n_jobs=1)
    check_distance_matrix(D, X)
    aeon_D = aeon.wddtw_pairwise_distance(X, g=g, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_adtw_distance():
    warp_penalty = 1.0
    D = adtw_distance(X, None, band=band, warp_penalty=warp_penalty, n_jobs=1)
    check_distance_matrix(D, X)
    aeon_D = aeon.adtw_pairwise_distance(X, window=band, warp_penalty=warp_penalty)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_msm_distance():
    D = msm_distance(X, None, band=band, n_jobs=1)
    check_distance_matrix(D, X)
    aeon_D = aeon.msm_pairwise_distance(X, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_twe_distance():
    stiffness = 0.1
    penalty = 0.1
    D = twe_distance(X, None, band=band, stifness=stiffness, penalty=penalty, n_jobs=1)
    check_distance_matrix(D, X)
    aeon_D = aeon.twe_pairwise_distance(X, nu=stiffness, lmbda=penalty, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)

def test_sb_distance():
    D = sb_distance(X, None, n_jobs=1)
    check_distance_matrix(D, X)
    aeon_D = aeon.sbd_pairwise_distance(X)
    assert np.allclose(D, aeon_D, atol=1e-8)

def test_mp_distance():
    D = mp_distance(X, 10, None, n_jobs=1)
    check_distance_matrix(D, X)
    D_stumpy = np.array([[stumpy.mpdist(X[i], X[j], m=10) for j in range(X.shape[0])] for i in range(X.shape[0])])
    assert np.allclose(D, D_stumpy, atol=1e-8)

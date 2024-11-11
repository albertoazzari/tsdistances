import pytest
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

def load_ArrowHead():
    print("\nLoading ArrowHead dataset")
    train = np.loadtxt("../DATA/ucr/ArrowHead/ArrowHead_TRAIN.tsv", delimiter="\t")
    test = np.loadtxt("../DATA/ucr/ArrowHead/ArrowHead_TEST.tsv", delimiter="\t")
    X_train, _ = train[:, 1:], train[:, 0].astype(int)
    X_test, _ = test[:, 1:], test[:, 0].astype(int)
    X = np.vstack((X_train, X_test))
    print(f"Shape: {X.shape}")
    return X


X = load_ArrowHead()
band = 1.0


def check_metric(D, X):
    # Check that the distance matrix is symmetric, has zeros on the diagonal, and all elements are positive
    assert np.allclose(D, D.T, atol=1e-8), "Distance matrix is not symmetric"
    assert np.allclose(np.diag(D), np.zeros(X.shape[0]), atol=1e-8), "Diagonal elements are not zeros"
    assert np.all(D >= 0), "Distance matrix has negative elements"
    

def test_erp_distance():
    gap_penalty = 0.0
    D = erp_distance(X, None, gap_penalty=gap_penalty, band=band, n_jobs=-1)
    D_gpu = erp_distance(X, None, gap_penalty=gap_penalty, band=band, device='gpu')
    check_metric(D_gpu, X)
    assert np.allclose(D, D_gpu, atol=1e-8)


def test_lcss_distance():
    epsilon = 0.1
    D = lcss_distance(X, None, epsilon=epsilon, band=band, n_jobs=-1)
    D_gpu = lcss_distance(X, None, epsilon=epsilon, band=band, device='gpu')
    check_metric(D_gpu, X)
    assert np.allclose(D, D_gpu, atol=1e-8)


def test_dtw_distance():
    D = dtw_distance(X, None, band=band, n_jobs=-1)
    D_gpu = dtw_distance(X, None, band=band, device='gpu')
    check_metric(D_gpu, X)
    assert np.allclose(D, D_gpu, atol=1e-8)


def test_ddtw_distance():
    D = ddtw_distance(X, None, band=band, n_jobs=-1)
    D_gpu =  ddtw_distance(X, None, band=band, device='gpu')
    check_metric(D_gpu, X)
    assert np.allclose(D, D_gpu, atol=1e-8)


def test_wdtw_distance():
    g = 0.05
    D = wdtw_distance(X, None, g=g, band=band, n_jobs=-1)
    D_gpu = wdtw_distance(X, None, g=g, band=band, device='gpu')
    check_metric(D_gpu, X)
    assert np.allclose(D, D_gpu, atol=1e-8)


def test_wddtw_distance():
    g = 0.05
    D = wddtw_distance(X, None, g=g, band=band, n_jobs=-1)
    D_gpu = wddtw_distance(X, None, g=g, band=band, device='gpu')
    check_metric(D_gpu, X)
    assert np.allclose(D, D_gpu, atol=1e-8)


def test_adtw_distance():
    warp_penalty = 1.0
    D = adtw_distance(X, None, band=band, warp_penalty=warp_penalty, n_jobs=-1)
    D_gpu = adtw_distance(X, None, band=band, warp_penalty=warp_penalty, device='gpu')
    check_metric(D_gpu, X)
    assert np.allclose(D, D_gpu, atol=1e-8)


def test_msm_distance():
    D = msm_distance(X, None, band=band, n_jobs=-1)
    D_gpu = msm_distance(X, None, band=band, n_jobs=-1, device='gpu')
    check_metric(D_gpu, X)
    assert np.allclose(D, D_gpu, atol=1e-8)


def test_twe_distance():
    stiffness = 0.1
    penalty = 0.1
    D = twe_distance(X, None, band=band, stifness=stiffness, penalty=penalty, n_jobs=-1)
    D_gpu = twe_distance(X, None, band=band, stifness=stiffness, penalty=penalty, device='gpu')
    check_metric(D_gpu, X)
    assert np.allclose(D, D_gpu, atol=1e-8)
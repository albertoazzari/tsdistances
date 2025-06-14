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
import time

N_SAMPLES = 10
A = np.loadtxt('tests/ACSF1/ACSF1_TRAIN.tsv', delimiter='\t')[:N_SAMPLES]
B = np.loadtxt('tests/ACSF1/ACSF1_TEST.tsv', delimiter='\t')[:2 * N_SAMPLES]
band = 1.0

def test_erp_distance():
    gap_penalty = 0.0
    start_time = time.time()
    D = erp_distance(A, B, gap_penalty=gap_penalty, band=band, par=True)
    end_time = time.time()
    start_time_gpu = time.time()
    D_gpu = erp_distance(A, B, gap_penalty=gap_penalty, band=band, device='gpu')
    end_time_gpu = time.time()
    print(f"Speedup: {end_time - start_time:.4f} / {end_time_gpu - start_time_gpu:.4f} = {(end_time - start_time) / (end_time_gpu - start_time_gpu):.2f}x")
    # Check that the GPU and CPU results are close (compare double precision with the single precision of GPU)
    assert np.allclose(D, D_gpu, rtol=1e-4, atol=1e-6)


def test_lcss_distance():
    epsilon = 0.1
    D = lcss_distance(A, B, epsilon=epsilon, band=band, par=True)
    D_gpu = lcss_distance(A, B, epsilon=epsilon, band=band, device='gpu')
    # Check that the GPU and CPU results are close (compare double precision with the single precision of GPU)
    assert np.allclose(D, D_gpu, rtol=1e-4, atol=1e-6)


def test_dtw_distance():
    D = dtw_distance(A, B, band=band, par=True)
    D_gpu = dtw_distance(A, B, band=band, device='gpu')
    # Check that the GPU and CPU results are close (compare double precision with the single precision of GPU)
    assert np.allclose(D, D_gpu, rtol=1e-4, atol=1e-6)


def test_ddtw_distance():
    D = ddtw_distance(A, B, band=band, par=True)
    D_gpu =  ddtw_distance(A, B, band=band, device='gpu')
    # Check that the GPU and CPU results are close (compare double precision with the single precision of GPU)
    assert np.allclose(D, D_gpu, rtol=1e-4, atol=1e-6)


def test_wdtw_distance():
    g = 0.05
    D = wdtw_distance(A, B, g=g, band=band, par=True)
    D_gpu = wdtw_distance(A, B, g=g, band=band, device='gpu')
    # Check that the GPU and CPU results are close (compare double precision with the single precision of GPU)
    assert np.allclose(D, D_gpu, rtol=1e-4, atol=1e-6)


def test_wddtw_distance():
    g = 0.05
    D = wddtw_distance(A, B, g=g, band=band, par=True)
    D_gpu = wddtw_distance(A, B, g=g, band=band, device='gpu')
    # Check that the GPU and CPU results are close (compare double precision with the single precision of GPU)
    assert np.allclose(D, D_gpu, rtol=1e-4, atol=1e-6)


def test_adtw_distance():
    warp_penalty = 1.0
    D = adtw_distance(A, B, band=band, warp_penalty=warp_penalty, par=True)
    D_gpu = adtw_distance(A, B, band=band, warp_penalty=warp_penalty, device='gpu')
    # Check that the GPU and CPU results are close (compare double precision with the single precision of GPU)
    assert np.allclose(D, D_gpu, rtol=1e-4, atol=1e-6)


def test_msm_distance():
    D = msm_distance(A, B, band=band, par=True)
    D_gpu = msm_distance(A, B, band=band, par=True, device='gpu')
    # Check that the GPU and CPU results are close (compare double precision with the single precision of GPU)
    assert np.allclose(D, D_gpu, rtol=1e-4, atol=1e-6)


def test_twe_distance():
    stiffness = 0.1
    penalty = 0.1
    D = twe_distance(A, B, band=band, stifness=stiffness, penalty=penalty, par=True)
    D_gpu = twe_distance(A, B, band=band, stifness=stiffness, penalty=penalty, device='gpu')
    # Check that the GPU and CPU results are close (compare double precision with the single precision of GPU)
    assert np.allclose(D, D_gpu, rtol=1e-4, atol=1e-6)
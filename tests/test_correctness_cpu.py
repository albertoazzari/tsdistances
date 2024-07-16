import unittest

import numpy as np
import os
from tsdistances import euclidean_distance, erp_distance, lcss_distance, dtw_distance, ddtw_distance, wdtw_distance, wddtw_distance, adtw_distance, msm_distance, twe_distance, sb_distance
from aeon import distances as aeon

def load_random_dataset():
    n_timeseries = 2
    n_timesteps = 10

    X_train = np.random.rand(n_timeseries, n_timesteps)
    y_train = np.random.randint(0, 10, n_timeseries)

    X_test = np.random.rand(n_timeseries, n_timesteps)
    y_test = np.random.randint(0, 10, n_timeseries)

    return np.vstack((X_train, X_test)), np.hstack((y_train, y_test))

class TestCorrectnessCPUAllDistances(unittest.TestCase):

    X, y = load_random_dataset()

    def test_euclidean_distance(self):
        # Compute the pairwise distances
        D = euclidean_distance(self.X, None, n_jobs=1)

        # Convert to numpy array
        D = np.array(D)

        # Check that the distance matrix is symmetric
        self.assertTrue(np.allclose(D, D.T, atol=1e-8))

        # Check that the diagonal is zero
        self.assertTrue(np.allclose(np.diag(D), np.zeros(self.X.shape[0]), atol=1e-8))

        # Check that the distance is positive
        self.assertTrue(np.all(D >= 0))

        # Check that the distance is zero iff the two time series are equal
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if i == j:
                    self.assertTrue(np.isclose(D[i, j], 0, atol=1e-8))
                else:
                    self.assertTrue(D[i, j] > 0)

        # Check that aeon returns the same result
        aeon_D = aeon.euclidean_pairwise_distance(self.X)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_erp_distance(self):
        # Params
        gap_penalty = 0.0
        sakoe_chiba_band = 0.5
        # Compute the pairwise distances
        D = erp_distance(self.X, None, sakoe_chiba_band=sakoe_chiba_band, gap_penalty=gap_penalty, n_jobs=1)

        # Convert to numpy array
        D = np.array(D)

        # Check that the distance matrix is symmetric
        self.assertTrue(np.allclose(D, D.T, atol=1e-8))

        # Check that the diagonal is zero
        self.assertTrue(np.allclose(np.diag(D), np.zeros(self.X.shape[0]), atol=1e-8))

        # Check that the distance is positive
        self.assertTrue(np.all(D >= 0))

        # Check that the distance is zero iff the two time series are equal
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if i == j:
                    self.assertTrue(np.isclose(D[i, j], 0, atol=1e-8))
                else:
                    self.assertTrue(D[i, j] > 0)

        # Check that aeon returns the same result
        aeon_D = aeon.erp_pairwise_distance(self.X, g=gap_penalty, window=sakoe_chiba_band)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))
    
    def test_lcss_distance(self):
        # Compute the pairwise distances
        D = lcss_distance(self.X, None, sakoe_chiba_band=0.1, epsilon=0.1, n_jobs=1)

        # Convert to numpy array
        D = np.array(D)

        # Check that the distance matrix is symmetric
        self.assertTrue(np.allclose(D, D.T, atol=1e-8))

        # Check that the diagonal is zero
        self.assertTrue(np.allclose(np.diag(D), np.zeros(self.X.shape[0]), atol=1e-8))

        # Check that the distance is positive
        self.assertTrue(np.all(D >= 0))

        # Check that the distance is zero iff the two time series are equal
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if i == j:
                    self.assertTrue(np.isclose(D[i, j], 0, atol=1e-8))
                else:
                    self.assertTrue(D[i, j] > 0)

        # Check that aeon returns the same result
        aeon_D = aeon.lcss_pairwise_distance(self.X, window=0.1, epsilon=0.1)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_dtw_distance(self):
        # Compute the pairwise distances
        D = dtw_distance(self.X, None, sakoe_chiba_band=0.1, n_jobs=1)

        # Convert to numpy array
        D = np.array(D)

        # Check that the distance matrix is symmetric
        self.assertTrue(np.allclose(D, D.T, atol=1e-8))

        # Check that the diagonal is zero
        self.assertTrue(np.allclose(np.diag(D), np.zeros(self.X.shape[0]), atol=1e-8))

        # Check that the distance is positive
        self.assertTrue(np.all(D >= 0))

        # Check that the distance is zero iff the two time series are equal
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if i == j:
                    self.assertTrue(np.isclose(D[i, j], 0, atol=1e-8))
                else:
                    self.assertTrue(D[i, j] > 0)

        # Check that aeon returns the same result
        aeon_D = aeon.dtw_pairwise_distance(self.X, window=0.1)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_ddtw_distance(self):
        # Compute the pairwise distances
        D = ddtw_distance(self.X, None, sakoe_chiba_band=0.1, n_jobs=1)

        # Convert to numpy array
        D = np.array(D)

        # Check that the distance matrix is symmetric
        self.assertTrue(np.allclose(D, D.T, atol=1e-8))

        # Check that the diagonal is zero
        self.assertTrue(np.allclose(np.diag(D), np.zeros(self.X.shape[0]), atol=1e-8))

        # Check that the distance is positive
        self.assertTrue(np.all(D >= 0))

        # Check that the distance is zero iff the two time series are equal
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if i == j:
                    self.assertTrue(np.isclose(D[i, j], 0, atol=1e-8))
                else:
                    self.assertTrue(D[i, j] > 0)

        # Check that aeon returns the same result
        aeon_D = aeon.ddtw_pairwise_distance(self.X, window=0.1)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_wdtw_distance(self):
        # Compute the pairwise distances
        D = wdtw_distance(self.X, None, sakoe_chiba_band=0.1, g=0.05, n_jobs=1)

        # Convert to numpy array
        D = np.array(D)

        # Check that the distance matrix is symmetric
        self.assertTrue(np.allclose(D, D.T, atol=1e-8))

        # Check that the diagonal is zero
        self.assertTrue(np.allclose(np.diag(D), np.zeros(self.X.shape[0]), atol=1e-8))

        # Check that the distance is positive
        self.assertTrue(np.all(D >= 0))

        # Check that the distance is zero iff the two time series are equal
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if i == j:
                    self.assertTrue(np.isclose(D[i, j], 0, atol=1e-8))
                else:
                    self.assertTrue(D[i, j] > 0)

        # Check that aeon returns the same result
        aeon_D = aeon.wdtw_pairwise_distance(self.X, window=0.1, g=0.05)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_wddtw_distance(self):
        # Compute the pairwise distances
        D = wddtw_distance(self.X, None, sakoe_chiba_band=0.1, g=0.05, n_jobs=1)

        # Convert to numpy array
        D = np.array(D)

        # Check that the distance matrix is symmetric
        self.assertTrue(np.allclose(D, D.T, atol=1e-8))

        # Check that the diagonal is zero
        self.assertTrue(np.allclose(np.diag(D), np.zeros(self.X.shape[0]), atol=1e-8))

        # Check that the distance is positive
        self.assertTrue(np.all(D >= 0))

        # Check that the distance is zero iff the two time series are equal
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if i == j:
                    self.assertTrue(np.isclose(D[i, j], 0, atol=1e-8))
                else:
                    self.assertTrue(D[i, j] > 0)

        # Check that aeon returns the same result
        aeon_D = aeon.wddtw_pairwise_distance(self.X, window=0.1, g=0.05)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_adtw_distance(self):
        # Compute the pairwise distances
        D = adtw_distance(self.X, None, sakoe_chiba_band=0.1, warp_penalty=1.0, n_jobs=1)

        # Convert to numpy array
        D = np.array(D)

        # Check that the distance matrix is symmetric
        self.assertTrue(np.allclose(D, D.T, atol=1e-8))

        # Check that the diagonal is zero
        self.assertTrue(np.allclose(np.diag(D), np.zeros(self.X.shape[0]), atol=1e-8))

        # Check that the distance is positive
        self.assertTrue(np.all(D >= 0))

        # Check that the distance is zero iff the two time series are equal
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if i == j:
                    self.assertTrue(np.isclose(D[i, j], 0, atol=1e-8))
                else:
                    self.assertTrue(D[i, j] > 0)

        # Check that aeon returns the same result
        aeon_D = aeon.adtw_pairwise_distance(self.X, window=0.1, warp_penalty=1.0)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_msm_distance(self):
        # Compute the pairwise distances
        D = msm_distance(self.X, None, sakoe_chiba_band=0.1, n_jobs=1)

        # Convert to numpy array
        D = np.array(D)

        # Check that the distance matrix is symmetric
        self.assertTrue(np.allclose(D, D.T, atol=1e-8))

        # Check that the diagonal is zero
        self.assertTrue(np.allclose(np.diag(D), np.zeros(self.X.shape[0]), atol=1e-8))

        # Check that the distance is positive
        self.assertTrue(np.all(D >= 0))

        # Check that the distance is zero iff the two time series are equal
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if i == j:
                    self.assertTrue(np.isclose(D[i, j], 0, atol=1e-8))
                else:
                    self.assertTrue(D[i, j] > 0)

        # Check that aeon returns the same result
        aeon_D = aeon.msm_pairwise_distance(self.X, window=0.1)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))


    def test_twe_distance(self):
        # Compute the pairwise distances
        D = twe_distance(self.X, None, sakoe_chiba_band=0.1, stifness=0.1, penalty=0.1, n_jobs=1)

        # Convert to numpy array
        D = np.array(D)

        # Check that the distance matrix is symmetric
        self.assertTrue(np.allclose(D, D.T, atol=1e-8))

        # Check that the diagonal is zero
        self.assertTrue(np.allclose(np.diag(D), np.zeros(self.X.shape[0]), atol=1e-8))

        # Check that the distance is positive
        self.assertTrue(np.all(D >= 0))

        # Check that the distance is zero iff the two time series are equal
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if i == j:
                    self.assertTrue(np.isclose(D[i, j], 0, atol=1e-8))
                else:
                    self.assertTrue(D[i, j] > 0)

        # Check that aeon returns the same result
        aeon_D = aeon.twe_pairwise_distance(self.X, nu=0.1, lmbda=0.1, window=0.1)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_sbd_distance(self):
        # Compute the pairwise distances
        D = sb_distance(self.X, None, n_jobs=1)

        # Convert to numpy array
        D = np.array(D)

        # Check that the distance matrix is symmetric
        self.assertTrue(np.allclose(D, D.T, atol=1e-8))

        # Check that the diagonal is zero
        self.assertTrue(np.allclose(np.diag(D), np.zeros(self.X.shape[0]), atol=1e-8))

        # Check that the distance is positive
        self.assertTrue(np.all(D >= 0))

        # Check that the distance is zero iff the two time series are equal
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if i == j:
                    self.assertTrue(np.isclose(D[i, j], 0, atol=1e-8))
                else:
                    self.assertTrue(D[i, j] > 0)

        # Check that aeon returns the same result
        aeon_D = aeon.sbd_pairwise_distance(self.X)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))
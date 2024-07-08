import unittest

import numpy as np
import os
from tsdistances import euclidean_distance, erp_distance, lcss_distance, dtw_distance, ddtw_distance, wdtw_distance, wddtw_distance, adtw_distance, msm_distance, twe_distance, sb_distance
from aeon import distances as aeon

UCR_ARCHIVE_PATH = "/media/aazzari/UCRArchive_2018/"

def load_random_ucr_dataset():
    # Load the UCR dataset
    dataset_name = "Coffee"# np.random.choice(os.listdir(UCR_ARCHIVE_PATH))
    dataset_path = os.path.join(UCR_ARCHIVE_PATH, dataset_name, dataset_name)

    # Load the dataset
    train = np.loadtxt(dataset_path + "_TRAIN.tsv", delimiter='\t')
    test = np.loadtxt(dataset_path + "_TEST.tsv", delimiter='\t')
    X_train, y_train = train[:, 1:], train[:, 0]
    X_test, y_test = test[:, 1:], test[:, 0]

    return np.vstack((X_train, X_test)), np.hstack((y_train, y_test))

class TestCorrectnessAllDistances(unittest.TestCase):

    X, y = load_random_ucr_dataset()

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
        # Compute the pairwise distances
        D = erp_distance(self.X, None, gap_penalty=0.0, n_jobs=1, device='gpu')

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
        aeon_D = aeon.erp_pairwise_distance(self.X, g=0.0)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))
    
    def test_lcss_distance(self):
        # Compute the pairwise distances
        D = lcss_distance(self.X, None, epsilon=0.1, n_jobs=1, device='gpu')

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
        aeon_D = aeon.lcss_pairwise_distance(self.X, epsilon=0.1)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_dtw_distance(self):
        # Compute the pairwise distances
        D = dtw_distance(self.X, None, n_jobs=1)

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
        aeon_D = aeon.dtw_pairwise_distance(self.X)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_ddtw_distance(self):
        # Compute the pairwise distances
        D = ddtw_distance(self.X, None, n_jobs=1)

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
        aeon_D = aeon.ddtw_pairwise_distance(self.X)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_wdtw_distance(self):
        # Compute the pairwise distances
        D = wdtw_distance(self.X, None, g=0.05, n_jobs=1)

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
        aeon_D = aeon.wdtw_pairwise_distance(self.X, g=0.05)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_wddtw_distance(self):
        # Compute the pairwise distances
        D = wddtw_distance(self.X, None, g=0.05, n_jobs=1)

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
        aeon_D = aeon.wddtw_pairwise_distance(self.X, g=0.05)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_adtw_distance(self):
        # Compute the pairwise distances
        D = adtw_distance(self.X, None, warp_penalty=1.0, n_jobs=1)

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
        aeon_D = aeon.adtw_pairwise_distance(self.X, warp_penalty=1.0)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))

    def test_msm_distance(self):
        # Compute the pairwise distances
        D = msm_distance(self.X, None, n_jobs=1)

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
        aeon_D = aeon.msm_pairwise_distance(self.X)
        self.assertTrue(np.allclose(D, aeon_D, atol=1e-8))


    def test_twe_distance(self):
        # Compute the pairwise distances
        D = twe_distance(self.X, None, stifness=0.1, penalty=0.1, n_jobs=1)

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
        aeon_D = aeon.twe_pairwise_distance(self.X, nu=0.1, lmbda=0.1)
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
import unittest

import numpy as np
import os
from tsdistances import euclidean_distance, erp_distance, lcss_distance, dtw_distance, ddtw_distance, wdtw_distance, wddtw_distance, adtw_distance, msm_distance, twe_distance, sb_distance
from aeon import distances as aeon
import time

def load_random_dataset():
    n_timeseries = 200
    n_timesteps = 1000

    X_train = np.random.rand(n_timeseries, n_timesteps)
    y_train = np.random.randint(0, 10, n_timeseries)

    X_test = np.random.rand(n_timeseries, n_timesteps)
    y_test = np.random.randint(0, 10, n_timeseries)

    return np.vstack((X_train, X_test)), np.hstack((y_train, y_test))

class TestSpeedCPUAllDistances(unittest.TestCase):

    X, y = load_random_dataset()

    def test_euclidean_distance(self):
        start_D = time.time()
        euclidean_distance(self.X, None, n_jobs=1)
        end_D = time.time()

        start_aeon = time.time()
        aeon.euclidean_pairwise_distance(self.X)
        end_aeon = time.time()
        print(end_aeon - start_aeon, end_D - start_D)
        self.assertLessEqual(end_D - start_D, end_aeon - start_aeon)

    def test_erp_distance(self):
        start_D = time.time()
        erp_distance(self.X, None, gap_penalty=0.05, n_jobs=-1)
        end_D = time.time()

        start_aeon = time.time()
        aeon.erp_pairwise_distance(self.X, g=0.05)
        end_aeon = time.time()
        print(end_aeon - start_aeon, end_D - start_D)
        self.assertLessEqual(end_D - start_D, end_aeon - start_aeon)
    
    def test_lcss_distance(self):
        start_D = time.time()
        lcss_distance(self.X, None, epsilon=0.1, n_jobs=1)
        end_D = time.time()

        start_aeon = time.time()
        aeon.lcss_pairwise_distance(self.X, epsilon=0.1)
        end_aeon = time.time()
        print(end_aeon - start_aeon, end_D - start_D)
        self.assertLessEqual(end_D - start_D, end_aeon - start_aeon)

    def test_dtw_distance(self):
        start_D = time.time()
        dtw_distance(self.X, None, n_jobs=1)
        end_D = time.time()

        start_aeon = time.time()
        aeon.dtw_pairwise_distance(self.X)
        end_aeon = time.time()
        print(end_aeon - start_aeon, end_D - start_D)
        self.assertLessEqual(end_D - start_D, end_aeon - start_aeon)

    def test_ddtw_distance(self):
        start_D = time.time()
        ddtw_distance(self.X, None, n_jobs=1)
        end_D = time.time()

        start_aeon = time.time()
        aeon.ddtw_pairwise_distance(self.X)
        end_aeon = time.time()
        print(end_aeon - start_aeon, end_D - start_D)
        self.assertLessEqual(end_D - start_D, end_aeon - start_aeon)

    def test_wdtw_distance(self):
        start_D = time.time()
        wdtw_distance(self.X, None, g=0.05, n_jobs=1)
        end_D = time.time()

        start_aeon = time.time()
        aeon.wdtw_pairwise_distance(self.X, g=0.05)
        end_aeon = time.time()
        print(end_aeon - start_aeon, end_D - start_D)
        self.assertLessEqual(end_D - start_D, end_aeon - start_aeon)

    def test_wddtw_distance(self):
        start_D = time.time()
        wddtw_distance(self.X, None, g=0.05, n_jobs=1)
        end_D = time.time()

        start_aeon = time.time()
        aeon.wddtw_pairwise_distance(self.X, g=0.05)
        end_aeon = time.time()
        print(end_aeon - start_aeon, end_D - start_D)
        self.assertLessEqual(end_D - start_D, end_aeon - start_aeon)

    def test_adtw_distance(self):
        start_D = time.time()
        adtw_distance(self.X, None, warp_penalty=1.0, n_jobs=1)
        end_D = time.time()

        start_aeon = time.time()
        aeon.adtw_pairwise_distance(self.X, warp_penalty=1.0)
        end_aeon = time.time()
        print(end_aeon - start_aeon, end_D - start_D)
        self.assertLessEqual(end_D - start_D, end_aeon - start_aeon)

    def test_msm_distance(self):
        start_D = time.time()
        msm_distance(self.X, None, n_jobs=1)
        end_D = time.time()

        start_aeon = time.time()
        aeon.msm_pairwise_distance(self.X)
        end_aeon = time.time()
        print(end_aeon - start_aeon, end_D - start_D)
        self.assertLessEqual(end_D - start_D, end_aeon - start_aeon)


    def test_twe_distance(self):
        start_D = time.time()
        twe_distance(self.X, None, stifness=0.1, penalty=0.1, n_jobs=1)
        end_D = time.time()

        start_aeon = time.time()
        aeon.twe_pairwise_distance(self.X, nu=0.1, lmbda=0.1)
        end_aeon = time.time()
        print(end_aeon - start_aeon, end_D - start_D)
        self.assertLessEqual(end_D - start_D, end_aeon - start_aeon)

    def test_sbd_distance(self):
        start_D = time.time()
        sb_distance(self.X, None, n_jobs=1)
        end_D = time.time()

        start_aeon = time.time()
        aeon.sbd_pairwise_distance(self.X)
        end_aeon = time.time()
        print(end_aeon - start_aeon, end_D - start_D)
        self.assertLessEqual(end_D - start_D, end_aeon - start_aeon)
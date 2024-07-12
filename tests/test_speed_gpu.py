import unittest

import numpy as np
from tsdistances import erp_distance, lcss_distance, dtw_distance, ddtw_distance, wdtw_distance, wddtw_distance, adtw_distance, msm_distance, twe_distance
from aeon import distances as aeon
import time


def load_random_ucr_dataset():
    n_timeseries = 10
    n_timesteps = 10000

    X_train = np.random.rand(n_timeseries, n_timesteps)
    y_train = np.random.randint(0, 10, n_timeseries)

    X_test = np.random.rand(n_timeseries, n_timesteps)
    y_test = np.random.randint(0, 10, n_timeseries)

    return np.vstack((X_train, X_test)), np.hstack((y_train, y_test))

class TestSpeedGPUAllDistances(unittest.TestCase):

    X, y = load_random_ucr_dataset()

    def test_erp_distance(self):
        print(self.X.shape)
        start_gpu = time.time()
        erp_distance(self.X, None, gap_penalty=0.05, n_jobs=1, device="gpu")
        end_gpu = time.time()

        start_cpu = time.time()
        erp_distance(self.X, None, gap_penalty=0.05, n_jobs=-1, device="cpu")
        end_cpu = time.time()
        print(f"CPU: {end_cpu - start_cpu}, GPU: {end_gpu - start_gpu}s")
        self.assertLessEqual(end_gpu - start_gpu, end_cpu - start_cpu)
    
    def test_lcss_distance(self):
        start_gpu = time.time()
        lcss_distance(self.X, None, epsilon=0.1, n_jobs=1, device="gpu")
        end_gpu = time.time()

        start_cpu = time.time()
        lcss_distance(self.X, None, epsilon=0.1, n_jobs=-1, device="cpu")
        end_cpu = time.time()
        print(f"CPU: {end_cpu - start_cpu}, GPU: {end_gpu - start_gpu}s")
        self.assertLessEqual(end_gpu - start_gpu, end_cpu - start_cpu)

    def test_dtw_distance(self):
        start_gpu = time.time()
        dtw_distance(self.X, None, n_jobs=1, device="gpu")
        end_gpu = time.time()

        start_cpu = time.time()
        dtw_distance(self.X, None, n_jobs=-1, device="cpu")
        end_cpu = time.time()
        print(f"CPU: {end_cpu - start_cpu}, GPU: {end_gpu - start_gpu}s")
        self.assertLessEqual(end_gpu - start_gpu, end_cpu - start_cpu)

    def test_ddtw_distance(self):
        start_gpu = time.time()
        ddtw_distance(self.X, None, n_jobs=1, device="gpu")
        end_gpu = time.time()

        start_cpu = time.time()
        ddtw_distance(self.X, None, n_jobs=-1, device="cpu")
        end_cpu = time.time()
        print(f"CPU: {end_cpu - start_cpu}, GPU: {end_gpu - start_gpu}s")
        self.assertLessEqual(end_gpu - start_gpu, end_cpu - start_cpu)

    def test_wdtw_distance(self):
        start_gpu = time.time()
        wdtw_distance(self.X, None, g=0.05, n_jobs=1, device="gpu")
        end_gpu = time.time()

        start_cpu = time.time()
        wdtw_distance(self.X, None, g=0.05, n_jobs=-1, device="cpu")
        end_cpu = time.time()
        print(f"CPU: {end_cpu - start_cpu}, GPU: {end_gpu - start_gpu}s")
        self.assertLessEqual(end_gpu - start_gpu, end_cpu - start_cpu)

    def test_wddtw_distance(self):
        start_gpu = time.time()
        wddtw_distance(self.X, None, g=0.05, n_jobs=1, device="gpu")
        end_gpu = time.time()

        start_cpu = time.time()
        wddtw_distance(self.X, None, g=0.05, n_jobs=-1, device="cpu")
        end_cpu = time.time()
        print(f"CPU: {end_cpu - start_cpu}, GPU: {end_gpu - start_gpu}s")
        self.assertLessEqual(end_gpu - start_gpu, end_cpu - start_cpu)

    def test_adtw_distance(self):
        start_gpu = time.time()
        adtw_distance(self.X, None, warp_penalty=1.0, n_jobs=1, device="gpu")
        end_gpu = time.time()

        start_cpu = time.time()
        adtw_distance(self.X, None, warp_penalty=1.0, n_jobs=-1, device="cpu")
        end_cpu = time.time()
        print(f"CPU: {end_cpu - start_cpu}, GPU: {end_gpu - start_gpu}s")
        self.assertLessEqual(end_gpu - start_gpu, end_cpu - start_cpu)

    def test_msm_distance(self):
        start_gpu = time.time()
        msm_distance(self.X, None, n_jobs=1, device="gpu")
        end_gpu = time.time()

        start_cpu = time.time()
        msm_distance(self.X, None, n_jobs=-1, device="cpu")
        end_cpu = time.time()
        print(f"CPU: {end_cpu - start_cpu}, GPU: {end_gpu - start_gpu}s")
        self.assertLessEqual(end_gpu - start_gpu, end_cpu - start_cpu)


    def test_twe_distance(self):
        start_gpu = time.time()
        twe_distance(self.X, None, stifness=0.1, penalty=0.1, n_jobs=1, device="gpu")
        end_gpu = time.time()

        start_cpu = time.time()
        twe_distance(self.X, None, stifness=0.1, penalty=0.1, n_jobs=-1, device="cpu")
        end_cpu = time.time()
        print(f"CPU: {end_cpu - start_cpu}, GPU: {end_gpu - start_gpu}s")
        self.assertLessEqual(end_gpu - start_gpu, end_cpu - start_cpu)
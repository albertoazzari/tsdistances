import pytest
import numpy as np
from tsdistances import (
    euclidean_distance,
    catcheucl_distance,
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
    mp_distance,
)
import time
import pathlib
import pandas as pd

UCR_ARCHIVE_PATH = pathlib.Path('/media/DATA/albertoazzari/UCRArchive_2018')
BENCHMARKS_DS = ["ArrowHead", "Beef", "HouseTwenty", "CBF", "DiatomSizeReduction", "ItalyPowerDemand", "FreezerSmallTrain", "CinCECGTorso", "ECG200", "Ham", "ACSF1", "Adiac", "CricketX", "Haptics", "ChlorineConcentration", "FreezerRegularTrain", "MixedShapesSmallTrain", "DistalPhalanxOutlineCorrect", "Strawberry", "ShapesAll", "EthanolLevel", "Wafer", "UWaveGestureLibraryX", "NonInvasiveFetalECGThorax1"]
DISTANCES = [euclidean_distance, catcheucl_distance, erp_distance, lcss_distance, dtw_distance, ddtw_distance, wdtw_distance, wddtw_distance, adtw_distance, msm_distance, twe_distance, sb_distance, mp_distance]
MODALITIES = ["", "par", "gpu"]
def load_benchmark():
    benchmark_ds = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.name in BENCHMARKS_DS])
    return benchmark_ds

DATASETS_PATH = load_benchmark()

def test_tsdistances():
    times = np.full((len(DATASETS_PATH), len(DISTANCES), len(MODALITIES)), np.nan)

    for i, dataset in enumerate(DATASETS_PATH):
        print(f"\nDataset: {dataset.name}")
        train = np.loadtxt(dataset / f"{dataset.name}_TRAIN.tsv", delimiter="\t")
        test = np.loadtxt(dataset / f"{dataset.name}_TEST.tsv", delimiter="\t")
        X_train, _ = train[:, 1:], train[:, 0]
        X_test, _= test[:, 1:], test[:, 0]

        X = np.vstack((X_train, X_test))

        for j, distance in enumerate(DISTANCES):
            print(f"\tDistance: {distance.__name__}")
            print("\t\tSingle thread")
            start = time.time()
            D = distance(X, None, n_jobs=1)
            end = time.time()
            times[i, j, 0] = end - start

            print("\t\tParallel")
            start = time.time()
            D = distance(X, None, n_jobs=-1)
            end = time.time()
            times[i, j, 1] = end - start

            if distance.__name__ in ["erp_distance", "lcss_distance", "dtw_distance", "ddtw_distance", "wdtw_distance", "wddtw_distance", "adtw_distance", "msm_distance", "twe_distance"]:
                print("\t\tGPU")
                start = time.time()
                D = distance(X, None, device='gpu')
                end = time.time()
                times[i, j, 2] = end - start

    np.save("times_tsdistances.npy", times)
    

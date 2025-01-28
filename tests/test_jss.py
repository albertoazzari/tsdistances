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
from aeon import (
    euclidean_pairwise_distance,
    erp_pairwise_distance,
    lcss_pairwise_distance,
    dtw_pairwise_distance,
    ddtw_pairwise_distance,
    wdtw_pairwise_distance,
    wddtw_pairwise_distance,
    adtw_pairwise_distance,
    msm_pairwise_distance,
    twe_pairwise_distance,
    sbd_pairwise_distance,
)
import time
import pathlib

UCR_ARCHIVE_PATH = pathlib.Path('../../DATA/ucr')
BENCHMARKS_DS = ["ACSF1", "Adiac", "Beef", "CBF", "ChlorineConcentration", "CinCECGTorso", "CricketX", "DiatomSizeReduction", "DistalPhalanxOutlineCorrect", "ECG200", "EthanolLevel", "FreezerRegularTrain", "FreezerSmallTrain", "Ham", "Haptics", "HouseTwenty", "ItalyPowerDemand", "MixedShapesSmallTrain", "NonInvasiveFetalECGThorax1", "ShapesAll", "Strawberry", "UWaveGestureLibraryX", "Wafer"]
TSDISTANCES = [euclidean_distance, catcheucl_distance, erp_distance, lcss_distance, dtw_distance, ddtw_distance, wdtw_distance, wddtw_distance, adtw_distance, msm_distance, twe_distance, sb_distance, mp_distance]
AEONDISTANCES = [euclidean_pairwise_distance, erp_pairwise_distance, lcss_pairwise_distance, dtw_pairwise_distance, ddtw_pairwise_distance, wdtw_pairwise_distance, wddtw_pairwise_distance, adtw_pairwise_distance, msm_pairwise_distance, twe_pairwise_distance, sbd_pairwise_distance]
MODALITIES = ["", "par", "gpu"]

def load_benchmark():
    benchmark_ds = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.name in BENCHMARKS_DS])
    return benchmark_ds

DATASETS_PATH = load_benchmark()

def test_draw_scatter_ucr():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    ucr_datasets = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.is_dir()])
    ucr_info = np.zeros((len(ucr_datasets), 2))
    is_benchmark = np.empty(len(ucr_datasets), dtype=str)

    for i, dataset in enumerate(ucr_datasets):
        train = np.loadtxt(dataset / f"{dataset.name}_TRAIN.tsv", delimiter="\t")
        test = np.loadtxt(dataset / f"{dataset.name}_TEST.tsv", delimiter="\t")
        X_train, _ = train[:, 1:], train[:, 0]
        X_test, _ = test[:, 1:], test[:, 0]

        X = np.vstack((X_train, X_test))
        ucr_info[i] = X.shape
        is_benchmark[i] = "Benchmarked" if dataset.name in BENCHMARKS_DS else "Non-benchmarked"

    # Create the scatter plot
    data = pd.DataFrame({"Dataset size": ucr_info[:, 0], "Time-series Length": ucr_info[:, 1], "Benchmark Status": is_benchmark})
    
    sns.scatterplot(data=data[data["Benchmark Status"]=="N"], x='Dataset size', y='Time-series Length', label='Non-Benchmarked', marker='o')
    sns.scatterplot(data=data[data["Benchmark Status"]=="B"], x='Dataset size', y='Time-series Length', label='Benchmarked', marker='x', linewidth=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dataset size (log scale)")
    plt.ylabel("Time-series Length (log scale)")
    plt.legend()
    plt.title("UCR Datasets")

    for i in range(len(ucr_datasets)):
        if is_benchmark[i] == "B":
            if i != 43:
                plt.text(
                    ucr_info[i, 0], 
                    ucr_info[i, 1],
                    str(i+1),
                    ha = 'center',
                    va = 'top',
                    color = 'black',
                    fontsize=6,
                )
            else:
                plt.text(
                    ucr_info[i, 0], 
                    ucr_info[i, 1],
                    f"[{44}-{45}]",
                    ha = 'center',
                    va = 'top',
                    color = 'black',
                    fontsize=6,
                )

    plt.savefig("ucr_scatter_log_scaled.pdf", dpi=300)


def test_tsdistances():
    tsdistances_times = np.full((len(DATASETS_PATH), len(TSDISTANCES), len(MODALITIES)), np.nan)
    aeon_times = np.full((len(DATASETS_PATH), len(TSDISTANCES)), np.nan)

    for i, dataset in enumerate(DATASETS_PATH):
        print(f"\nDataset: {dataset.name}")
        train = np.loadtxt(dataset / f"{dataset.name}_TRAIN.tsv", delimiter="\t")
        test = np.loadtxt(dataset / f"{dataset.name}_TEST.tsv", delimiter="\t")
        X_train = train[:, 1:]
        X_test = test[:, 1:]

        X = np.vstack((X_train, X_test))

        for j, distance in enumerate(TSDISTANCES):
            print(f"\tDistance: {distance.__name__}")
            print("\t\tSingle thread")
            start = time.time()
            D = distance(X, None, n_jobs=1)
            end = time.time()
            tsdistances_times[i, j, 0] = end - start

            print("\t\tParallel")
            start = time.time()
            D = distance(X, None, n_jobs=-1)
            end = time.time()
            tsdistances_times[i, j, 1] = end - start

            if distance.__name__ in ["erp_distance", "lcss_distance", "dtw_distance", "ddtw_distance", "wdtw_distance", "wddtw_distance", "adtw_distance", "msm_distance", "twe_distance"]:
                print("\t\tGPU")
                start = time.time()
                D = distance(X, None, device='gpu')
                end = time.time()
                tsdistances_times[i, j, 2] = end - start
        
        for j, distance in enumerate(AEONDISTANCES):
            print(f"\tDistance: {distance.__name__}")
            print("\t\tSingle thread")
            start = time.time()
            D = distance(X)
            end = time.time()
            aeon_times[i, j] = end - start

    np.save("times_tsdistances.npy", tsdistances_times)
    np.save("times_aeon.npy", aeon_times)
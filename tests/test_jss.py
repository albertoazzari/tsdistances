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
import time
import pathlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

UCR_ARCHIVE_PATH = pathlib.Path('/media/DATA/albertoazzari/UCRArchive_2018')

def load_ucr_dataset():
    benchmark_ds = np.loadtxt("tests/benchmark_ds.csv", delimiter=',', dtype=str)
    benchmark_ds = [x for x in benchmark_ds[1:, 0] if x != 'X']
    benchmark_ds = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.name in benchmark_ds])
    return benchmark_ds

def test_alldist_single():
    archives = load_ucr_dataset()
    # #dataset, #libraries, #distances, #steps (0: distances, 1: classification)
    times_tsdistances = np.zeros((len(archives), 2, 11, 2))
    accuracy_scores = np.zeros((len(archives), 2, 11))
    for i, archive in enumerate(archives):
        dataset = archive.stem
        print("\n", dataset)
        train = np.loadtxt(archive / f'{dataset}_TRAIN.tsv', delimiter='\t')
        test = np.loadtxt(archive / f'{dataset}_TEST.tsv', delimiter='\t')

        X_train, y_train = train[:, 1:], train[:, 0]
        X_test, y_test = test[:, 1:], test[:, 0]

        if np.isnan(X_train).any() or np.isnan(X_test).any():
            times_tsdistances[i] = np.nan
            accuracy_scores[i] = np.nan
            continue

        print("\ttsdistance")
        for j, distance in enumerate([euclidean_distance, erp_distance, lcss_distance, dtw_distance, ddtw_distance, wdtw_distance, wddtw_distance, adtw_distance, msm_distance, twe_distance, sb_distance]):
            start = time.time()
            d_train = distance(X_train, X_train, n_jobs=1)
            d_test = distance(X_test, X_train, n_jobs=1)
            end = time.time()
            times_tsdistances[i, 0, j, 0] = end - start
            print(f"\t\t{distance.__name__}: {end - start}")
            # d_train = np.where(d_train < 0, 0, d_train)
            # d_test = np.where(d_test < 0, 0, d_test)
            # start = time.time()
            # model = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
            # model.fit(d_train, y_train)
            # y_pred = model.predict(d_test)
            # end = time.time()
            # times_tsdistances[i, 0, j, 1] = end - start
            # accuracy_scores[i, 0, j] = accuracy_score(y_test, y_pred)
            
        # print("\taeon")
        # for j, distance in enumerate([aeon.euclidean_pairwise_distance, aeon.erp_pairwise_distance, aeon.lcss_pairwise_distance, aeon.dtw_pairwise_distance, aeon.ddtw_pairwise_distance, aeon.wdtw_pairwise_distance, aeon.wddtw_pairwise_distance, aeon.adtw_pairwise_distance, aeon.msm_pairwise_distance, aeon.twe_pairwise_distance, aeon.sbd_pairwise_distance]):
        #     start = time.time()
        #     d_train = distance(X_train, X_train)
        #     d_test = distance(X_test, X_train)
        #     end = time.time()
        #     times_tsdistances[i, 1, j, 0] = end - start
        #     print(f"\t\t{distance.__name__}: {end - start}")
            # start = time.time()
            # model = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
            # model.fit(d_train, y_train)
            # y_pred = model.predict(d_test)
            # end = time.time()
            # times_tsdistances[i, 1, j, 1] = end - start
            # accuracy_scores[i, 1, j] = accuracy_score(y_test, y_pred)

        # np.save('/media/DATA/JSS_results/times_benchmark_ds_single.npy', times_tsdistances)
        # np.save('/media/DATA/JSS_results/accuracy_benchmark_ds_single.npy', accuracy_scores)

def test_alldist_par():
    archives = load_ucr_dataset()
    # #archives, #libraries, #distances, #steps (0: distances, 1: classification)
    times_tsdistances = np.zeros((len(archives), 2, 11, 2))
    accuracy_scores = np.zeros((len(archives), 2, 11))
    aeon_times = np.load('/media/DATA/JSS_results/times_tsdistances.npy')
    aeon_accuracy = np.load('/media/DATA/JSS_results/accuracy_scores.npy')

    for i, archive in enumerate(archives):
        print(archive)
        dataset = archive.stem
        train = np.loadtxt(archive / f'{dataset}_TRAIN.tsv', delimiter='\t')
        test = np.loadtxt(archive / f'{dataset}_TEST.tsv', delimiter='\t')

        X_train, y_train = train[:, 1:], train[:, 0]
        X_test, y_test = test[:, 1:], test[:, 0]

        for j, distance in enumerate([euclidean_distance, erp_distance, lcss_distance, dtw_distance, ddtw_distance, wdtw_distance, wddtw_distance, adtw_distance, msm_distance, twe_distance, sb_distance]):
            start = time.time()
            d_train = distance(X_train, X_train, n_jobs=-1)
            d_test = distance(X_test, X_train, n_jobs=-1)
            end = time.time()
            times_tsdistances[i, 0, j, 0] = end - start
            start = time.time()
            model = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
            model.fit(d_train, y_train)
            y_pred = model.predict(d_test)
            end = time.time()
            times_tsdistances[i, 0, j, 1] = end - start
            accuracy_scores[i, 0, j] = accuracy_score(y_test, y_pred)
        
        for j, distance in enumerate([aeon.euclidean_pairwise_distance, aeon.erp_pairwise_distance, aeon.lcss_pairwise_distance, aeon.dtw_pairwise_distance, aeon.ddtw_pairwise_distance, aeon.wdtw_pairwise_distance, aeon.wddtw_pairwise_distance, aeon.adtw_pairwise_distance, aeon.msm_pairwise_distance, aeon.twe_pairwise_distance, aeon.sbd_pairwise_distance]):
            times_tsdistances[i, 1, j, 0] = aeon_times[i, 1, j, 0]
            times_tsdistances[i, 1, j, 1] = aeon_times[i, 1, j, 1]
            accuracy_scores[i, 1, j] = aeon_accuracy[i, 1, j]
        
        np.save('/media/DATA/JSS_results/times_tsdistances_par.npy', times_tsdistances)
        np.save('/media/DATA/JSS_results/accuracy_scores_par.npy', accuracy_scores)

def test_alldist_gpu():
    archives = load_ucr_dataset()
    # #dataset, #libraries, #distances, #steps (0: distances, 1: classification)
    times_tsdistances = np.zeros((len(archives), 11, 2))
    accuracy_scores = np.zeros((len(archives), 11))
    for i, archive in enumerate(archives):
        dataset = archive.stem
        print("\n", dataset)
        train = np.loadtxt(archive / f'{dataset}_TRAIN.tsv', delimiter='\t')
        test = np.loadtxt(archive / f'{dataset}_TEST.tsv', delimiter='\t')

        X_train, y_train = train[:, 1:], train[:, 0]
        X_test, y_test = test[:, 1:], test[:, 0]

        if np.isnan(X_train).any() or np.isnan(X_test).any():
            times_tsdistances[i] = np.nan
            accuracy_scores[i] = np.nan
            continue

        print("\ttsdistance")
        for j, distance in enumerate([erp_distance, lcss_distance, dtw_distance, ddtw_distance, wdtw_distance, wddtw_distance, adtw_distance, msm_distance, twe_distance]):
            start = time.time()
            d_train = distance(X_train, X_train, n_jobs=1, device="gpu")
            d_test = distance(X_test, X_train, n_jobs=1, device="gpu")
            end = time.time()
            times_tsdistances[i, j, 0] = end - start
            d_train = np.where(d_train < 0, 0, d_train)
            d_test = np.where(d_test < 0, 0, d_test)
            start = time.time()
            model = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
            model.fit(d_train, y_train)
            y_pred = model.predict(d_test)
            end = time.time()
            times_tsdistances[i, j, 1] = end - start
            accuracy_scores[i, j] = accuracy_score(y_test, y_pred)

        np.save('/media/DATA/JSS_results/times_tsdistances_gpu.npy', times_tsdistances)
        np.save('/media/DATA/JSS_results/accuracy_scores_gpu.npy', accuracy_scores)


def test_plots():
    aeon_times = np.load('/media/DATA/JSS_results/times_benchmark_ds.npy')
    datasets = np.loadtxt("tests/benchmark_ds.csv", delimiter=',', dtype=str)
    datasets = [x for x in datasets[1:, 0] if x != 'X']
    datasets = [x for x in UCR_ARCHIVE_PATH.iterdir() if x.name in datasets]
    # Produce a bar plot showing the differences in time of 5 random datasets
    # for each distance
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    i = 0
    j = 0
    while i < 3:
        if np.isnan(aeon_times[j, 1, 1:, 0]).any() or aeon_times[j, 1, -2, 0] < 10000:
            j+=1
            continue
        aeon_time = aeon_times[j, 1, 1:, 0]
        tsdistances_time = aeon_times[j, 0, 1:, 0]
        # log scale
        ax[i].bar(np.arange(10), aeon_time, alpha=0.5, label='aeon')
        ax[i].bar(np.arange(10), tsdistances_time, alpha=0.5, label='tsdistances')
        ax[i].set_yscale('log')
        ax[i].set_ylabel('Time (s)')
        ax[i].set_title(f'{datasets[j]}')
        ax[i].set_xticks(np.arange(10))
        ax[i].set_xticklabels(['ERP', 'LCSS', 'DTW', 'DDTW', 'WDTW', 'WDDTW', 'ADTW', 'MSM', 'TWE', 'SBD'], rotation=45)
        ax[i].legend(loc='upper right')

        i += 1
        j += 1
    plt.tight_layout()
    plt.savefig('times_comparison.png', dpi=300)
        

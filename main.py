import pathlib
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
import tsdistances
import time

UCRARCHIVE_PATH = pathlib.Path('/media/aazzari/UCRArchive_2018/')
DISTANCES = ["euclidean", "erp", "lcss", "dtw", "ddtw", "wdtw", "wddtw", "msm", "twe"]

if __name__ == '__main__':
    # Load the data
    ucr_distances = UCRARCHIVE_PATH.iterdir()
    ucr_distances = sorted([dir for dir in ucr_distances if dir.is_dir()])
    clustering = np.zeros((len(ucr_distances), len(DISTANCES)))
    times = np.zeros((len(ucr_distances), len(DISTANCES)))
    for i, dir in enumerate(ucr_distances):
        print(f'Processing {dir.name}...')
        
        train = pd.read_csv(UCRARCHIVE_PATH.joinpath(dir.name)/ f'{dir.name}_TRAIN.tsv', header=None, sep='\t').values
        test = pd.read_csv(UCRARCHIVE_PATH.joinpath(dir.name)/ f'{dir.name}_TEST.tsv', header=None, sep='\t').values

        X_train, y_train = train[:, 1:], train[:, 0]
        X_test, y_test = test[:, 1:], test[:, 0]

        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        # convert all nans in zeros
        X = np.nan_to_num(X)

        n_clusters = len(np.unique(y))
        for j, dist in enumerate(DISTANCES):
            start_time = time.time()
            print(f'\tProcessing {dist}...')
            if dist == 'euclidean':
                X = tsdistances.euclidean(X, cached=False, n_jobs=-1)
            elif dist == 'erp':
                X = tsdistances.erp(X, gap_penalty=0.0, band=1.0, cached=False, n_jobs=-1)
            elif dist == 'lcss':
                X = tsdistances.lcss(X, epsilon=1.0, band=1.0, cached=False, n_jobs=-1)
            elif dist == 'dtw':
                X = tsdistances.dtw(X, band=1.0, cached=False, n_jobs=-1)
            elif dist == 'ddtw':
                X = tsdistances.ddtw(X, band=1.0, cached=False, n_jobs=-1)
            elif dist == 'wdtw':
                X = tsdistances.wdtw(X, band=1.0, cached=False, n_jobs=-1)
            elif dist == 'wddtw':
                X = tsdistances.wddtw(X, band=1.0, cached=False, n_jobs=-1)
            elif dist == 'msm':
                X = tsdistances.msm(X, band=1.0, cached=False, n_jobs=-1)
            elif dist == 'twe':
                X = tsdistances.twe(X, stiffness=0.001, penalty=1.0, band=1.0, cached=False, n_jobs=-1)
            end_time = time.time()
            print(f'\tTime: {end_time - start_time}')
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
            y_pred = model.fit_predict(X)
            ari = adjusted_rand_score(y, y_pred)
            clustering[i, j] = ari
            times[i, j] = end_time - start_time
        np.savetxt('clustering.csv', clustering)
        np.savetxt('times.csv', times)
        





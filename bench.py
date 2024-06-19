import tsdistances
from sktime.distances import euclidean_distance
from dtaidistance import dtw
import pathlib
import pandas as pd
import time
import numpy as np

UCRARCHIVE_PATH = pathlib.Path('/media/aazzari/UCRArchive_2018/')

if __name__ == "__main__":
    ucr_distances = UCRARCHIVE_PATH.iterdir()
    ucr_distances = sorted([dir for dir in ucr_distances if dir.is_dir()])
    for i, dir in enumerate(ucr_distances):
        print(f'Processing {dir.name}...')
        train = pd.read_csv(UCRARCHIVE_PATH.joinpath(dir.name)/ f'{dir.name}_TRAIN.tsv', header=None, sep='\t').values
        test = pd.read_csv(UCRARCHIVE_PATH.joinpath(dir.name)/ f'{dir.name}_TEST.tsv', header=None, sep='\t').values

        tsdistances_time = []
        sktime_time = []

        tsdistances_v = []
        sktime_v = []

        for i in range(100):
            print(f'Iteration {i + 1}')
            idx_train = np.random.randint(0, train.shape[0])
            idx_test = np.random.randint(0, test.shape[0])
            ts1 = train[idx_train, 1:]
            ts2 = test[idx_test, 1:]

            start_tsdistances = time.time()
            tsdistances_v.append(tsdistances.dtw([ts1], [ts2], n_jobs=1)[0][0])
            end_tsdistances = time.time()
            tsdistances_time.append(end_tsdistances - start_tsdistances)

            start_sktime = time.time()
            sktime_v.append(dtw.distance_fast(ts1, ts2))
            end_sktime = time.time()
            sktime_time.append(end_sktime - start_sktime)
            # assert tsdistances_v[-1] == sktime_v[-1], f'Results are not the same\n{tsdistances_v[-1]}\n{sktime_v[-1]}'

        print(f'Average time for tsdistances: {np.mean(tsdistances_time)}')
        print(f'Average time for sktime: {np.mean(sktime_time)}')
        assert np.allclose(tsdistances_v, [x**2 for x in sktime_v]), 'Results are not the same'
        
        exit(1)
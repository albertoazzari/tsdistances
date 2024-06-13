import tsdistances
from sktime.distances import dtw_distance
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


        ts1 = np.hstack(train[:10, 1:])
        ts2 = np.hstack(test[:10, 1:])

        tsdistances_time = []
        tsdistances_time2 = []
        sktime_time = []
        dtai_time = []

        for i in range(10):
            print(f'Iteration {i}')
            start_tsdistances = time.time()
            a = tsdistances.dtw([ts1], [ts2], cached=False, n_jobs=1)
            end_tsdistances = time.time()
            tsdistances_time.append(end_tsdistances - start_tsdistances)

            start_tsdistances2 = time.time()
            b = tsdistances.dtw_diag([ts1], [ts2], cached=False, n_jobs=1)
            end_tsdistances2 = time.time()
            tsdistances_time2.append(end_tsdistances2 - start_tsdistances2)

            # start_sktime = time.time()
            # b = dtw_distance(ts1, ts2)
            # end_sktime = time.time()
            # sktime_time.append(end_sktime - start_sktime)

            start_dtai = time.time()
            c = dtw.distance_fast(ts1, ts2)
            end_dtai = time.time()
            dtai_time.append(end_dtai - start_dtai)

        time1 = sum(tsdistances_time) / len(tsdistances_time)
        #time2 = sum(sktime_time) / len(sktime_time)
        time3 = sum(dtai_time) / len(dtai_time)
        time4 = sum(tsdistances_time2) / len(tsdistances_time2)
        print(f'Tsdistances: {time1}')
        #print(f'Sktime: {time2}')
        print(f'Dtai: {time3}')
        print(f'Tsdistances2: {time4}')
        
        #print(f'\tSktime: {time1 / time2}')
        print(f'\tDtai: {time1 / time3}')
        print(f'\tTsdistances2: {time1 / time4}')

        assert a == b == c, f"{a}, {b}, {c}"
        


        exit(1)
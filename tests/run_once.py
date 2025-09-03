import time, numpy as np
from tsdistances import (
    euclidean_distance,
    erp_distance,
    lcss_distance,
    dtw_distance,
    twe_distance,
)
train = np.loadtxt("tests/ACSF1/ACSF1_TRAIN.tsv", delimiter="\t")
test  = np.loadtxt("tests/ACSF1/ACSF1_TEST.tsv", delimiter="\t")
X_train, X_test = train[:,1:], test[:,1:]

start = time.time()
res = dtw_distance(X_train, X_test, par=True)
print(f"Time: {time.time() - start:.4f}")
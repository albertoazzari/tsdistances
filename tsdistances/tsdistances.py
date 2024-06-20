import numpy as np

def dwt(X: list[float] | list[list[float]], Y: list[float] | list[list[float]], n_jobs: int = -1) -> float | list[float]:
    return tsdistances.dtw(X, Y, n_jobs=n_jobs)

    
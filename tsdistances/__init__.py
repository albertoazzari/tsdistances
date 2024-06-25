from .tsdistances import *
import numpy as np

def euclidean_distance(u, v=None, n_jobs=1):
    """
    Computes the Euclidean distance between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise Euclidean distances within `u`.

    Parameters
    ----------
    u : (N,) array_like or (M, N) array_like
        Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N) array_like, optional
        Input array. If provided, `v` should have the same shape as `u`.
        If `v` is None, pairwise distances within `u` are computed.
    n_jobs : int, optional
        Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
        The Euclidean distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> euclidean_distance([1, 0, 0], [0, 1, 0])
    1.4142135623730951
    >>> euclidean_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[1.41421356, 2.44948974],
           [1.        , 1.73205081]])
    >>> euclidean_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.        , 1.        ],
           [1.        , 0.        ]])

    """

    # Convert inputs to NumPy arrays for consistency and efficiency
    u = np.asarray(u)

    # If v is not provided, compute pairwise distances within u
    if v is None:
        if u.ndim != 2:
            raise ValueError("u must be a 2-D array when v is None.")
        return euclidean(u, None, n_jobs)

    v = np.asarray(v)

    # Check if shapes are compatible
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape.")

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return euclidean([u], [v], n_jobs)[0][0]

    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return euclidean(u, v, n_jobs)

    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")
    

def erp_distance(u, v=None, gap_penalty=0.0, n_jobs=1):
    """
    Computes the Edit Distance with Real Penalty (ERP) between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise ERP distances within `u`.
    The length of the input arrays are not required to be the same.
    
    Parameters
    ----------
    u : (N,) array_like or (M, N)
        Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
        Input array. If provided, `v` should have the same shape as `u`.
        If `v` is None, pairwise distances within `u` are computed.
    gap_penalty : double, optional
        Penalty for gap insertion/deletion (default is 0.0).
    n_jobs : int, optional
        Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
        The ERP distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> erp_distance([1, 0, 0], [0, 1, 0])
    2.0
    >>> erp_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[2.0, 4.0], [1.0, 3.0]])
    >>> erp_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 1.0], [1.0, 0.0]])
    """

    # Convert inputs to NumPy arrays for consistency and efficiency
    u = np.asarray(u)

    # If v is not provided, compute pairwise distances within u
    if v is None:
        if u.ndim != 2:
            raise ValueError("u must be a 2-D array when v is None.")
        return erp(u, None, gap_penalty, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return erp([u], [v], gap_penalty, n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return erp(u, v, gap_penalty, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")

def lcss_distance(u, v=None, epsilon=1.0, n_jobs=1):
    """
    Computes the Longest Common Subsequence (LCSS) between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise LCSS distances within `u`.
    The length of the input arrays are not required to be the same.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
        Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional

    epsilon : double, optional
        Threshold value for the distance between two elements (default is 1.0).
    n_jobs : int, optional

    Returns
    -------
    distance : double or ndarray
        The LCSS distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> lcss_distance([1, 0, 0], [0, 1, 0], epsilon=0.5)
    0.3333333333333333
    >>> lcss_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]], epsilon=0.5)
    array([[0.3333333333333333, 0.6666666666666666], [0.0, 0.3333333333333333]])
    >>> lcss_distance([[1, 1, 1], [0, 1, 1]], epsilon=0.5)
    array([[0.0, 0.3333333333333333], [0.3333333333333333, 0.0]])
    """
    # Convert inputs to NumPy arrays for consistency and efficiency
    u = np.asarray(u)

    # If v is not provided, compute pairwise distances within u
    if v is None:
        if u.ndim != 2:
            raise ValueError("u must be a 2-D array when v is None.")
        return lcss(u, None, epsilon, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return lcss([u], [v], epsilon, n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return lcss(u, v, epsilon, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")

def dtw_distance(u, v=None, n_jobs=1):
    """
    Computes the Dynamic Time Warping (DTW) between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise DTW distances within `u`.
    The length of the input arrays are not required to be the same.
    
    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array. If provided, `v` should have the same shape as `u`.
    If `v` is None, pairwise distances within `u` are computed.
    n_jobs : int, optional
    Number of jobs to use for computation (default is 1).
    
    Returns
    -------
    distance : double or ndarray
    The DTW distance(s) between vectors/sets `u` and `v`.
    
    Examples
    --------
    >>> dtw_distance([1, 0, 0], [0, 1, 0])
    1.0
    >>> dtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[2.0, 6.0], [1.0, 3.0]])
    >>> dtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.        , 1.        ], [1.        , 0.        ]])
    """
    # Convert inputs to NumPy arrays for consistency and efficiency
    u = np.asarray(u)

    # If v is not provided, compute pairwise distances within u
    if v is None:
        if u.ndim != 2:
            raise ValueError("u must be a 2-D array when v is None.")
        return dtw(u, None, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return dtw([u], [v], n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return dtw(u, v, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")


__all__ = ['euclidean', 'erp', 'lcss', 'dtw']
from tsdistances import tsdistances as tsd
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
        return tsd.euclidean(u, None, n_jobs)

    v = np.asarray(v)

    # Check if shapes are compatible
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape.")

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return tsd.euclidean([u], [v], n_jobs)[0][0]

    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return tsd.euclidean(u, v, n_jobs)

    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")
    

def erp_distance(u, v=None, gap_penalty=0.0, n_jobs=1):
    """
    Computes the Edit Distance with Real Penalty (ERP) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise ERP distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Chen, L. et al., On The Marriage of Lp-norms and Edit Distance, 2004.
    
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
        return tsd.erp(u, None, gap_penalty, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return tsd.erp([u], [v], gap_penalty, n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return tsd.erp(u, v, gap_penalty, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")

def lcss_distance(u, v=None, epsilon=1.0, n_jobs=1):
    """
    Computes the Longest Common Subsequence (LCSS) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise LCSS distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Vlachos, M. et al., Discovering Similar Multidimensional Trajectories, 2002.

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
        return tsd.lcss(u, None, epsilon, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return tsd.lcss([u], [v], epsilon, n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return tsd.lcss(u, v, epsilon, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")

def dtw_distance(u, v=None, n_jobs=1):
    """
    Computes the Dynamic Time Warping (DTW) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise DTW distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Berndt, D.J. and Clifford, J., Using Dynamic Time Warping to Find Patterns in Time Series, 1994.
    
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
        return tsd.dtw(u, None, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return tsd.dtw([u], [v], n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return tsd.dtw(u, v, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")


def ddtw_distance(u, v=None, n_jobs=1):
    """
    Computes the Derivative Dynamic Time Warping (DDTW) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise DDTW distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Keogh, E. et al., Derivative Dynamic Time Warping, 2001.
    
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
    The DDTW distance(s) between vectors/sets `u` and `v`.
    
    Examples
    --------
    >>> ddtw_distance([1, 0, 0], [0, 1, 0])
    4.6875
    >>> ddtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[0.75, 1.6875], [0.1875, 0.0]])
    >>> ddtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 1.6875], [1.6875, 0.0]])
    """
    # Convert inputs to NumPy arrays for consistency and efficiency
    u = np.asarray(u)

    # If v is not provided, compute pairwise distances within u
    if v is None:
        if u.ndim != 2:
            raise ValueError("u must be a 2-D array when v is None.")
        return tsd.ddtw(u, None, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return tsd.ddtw([u], [v], n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return tsd.ddtw(u, v, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")

def wdtw_distance(u, v=None, g=0.05, n_jobs=1):
    """
    Computes the Weighted Dynamic Time Warping (WDTW) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise WDTW distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Jeong Y.-S. et al., Weighted dynamic time warping for time series classification, 2011.
    
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
    The WDTW distance(s) between vectors/sets `u` and `v`.
    
    Examples
    --------
    >>> wdtw_distance([1, 0, 0], [0, 1, 0])
    0.18242552380635635
    >>> wdtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[0.3648510476127127, 1.094553142838138], [0.18242552380635635, 0.547276571419069]])
    >>> wdtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 0.18242552380635635], [0.18242552380635635, 0.0]])
    """

    # Convert inputs to NumPy arrays for consistency and efficiency
    u = np.asarray(u)

    # If v is not provided, compute pairwise distances within u
    if v is None:
        if u.ndim != 2:
            raise ValueError("u must be a 2-D array when v is None.")
        return tsd.wdtw(u, None, g, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return tsd.wdtw([u], [v], g, n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return tsd.wdtw(u, v, g, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")

def wddtw_distance(u, v=None, g=0.05, n_jobs=1):
    """
    Computes the Weighted Derivative Dynamic Time Warping (WDDTW) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise WDDTW distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Jeong, Y.-S. et al., Weighted dynamic time warping for time series classification, 2011.
    
    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array. If provided, `v` should have the same shape as `u`.
    If `v` is None, pairwise distances within `u` are computed.
    n_jobs : int, optional
    
    Returns
    -------
    distance : double or ndarray
    The WDDTW distance(s) between vectors/sets `u` and `v`.
    
    Examples
    --------
    >>> wddtw_distance([1, 0, 0], [0, 1, 0])
    0.8551196428422955
    >>> wddtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[0.13681914285476726, 0.3078430714232263], [0.034204785713691815, 0.0]])
    >>> wddtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 0.3078430714232263], [0.3078430714232263, 0.0]])
    """
    # Convert inputs to NumPy arrays for consistency and efficiency
    u = np.asarray(u)

    # If v is not provided, compute pairwise distances within u
    if v is None:
        if u.ndim != 2:
            raise ValueError("u must be a 2-D array when v is None.")
        return tsd.wddtw(u, None, g, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return tsd.wddtw([u], [v], g, n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return tsd.wddtw(u, v, g, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")

def adtw_distance(u, v=None, warp_penalty=0.1, n_jobs=1):
    """
    Computes the Amercing Dynamic Time Warping (ADTW) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise ADTW distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Hermann, M. et al., Amercing: An intuitive and effective constraint for dynamic time warping, 2023
    
    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array. If provided, `v` should have the same shape as `u`.
    If `v` is None, pairwise distances within `u` are computed.
    w : double, optional
    Weight amercing penalty (default is 0.1).
    n_jobs : int, optional
    Number of jobs to use for computation (default is 1).
    
    Returns
    -------
    distance : double or ndarray
    The ADTW distance(s) between vectors/sets `u` and `v`.
    
    Examples
    --------
    >>> adtw_distance([1, 0, 0], [0, 1, 0])
    0.0
    >>> adtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[0.0, 0.0], [0.0, 0.0]])
    >>> adtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 0.0], [0.0, 0.0]])
    """

    # Convert inputs to NumPy arrays for consistency and efficiency
    u = np.asarray(u)

    # If v is not provided, compute pairwise distances within u
    if v is None:
        if u.ndim != 2:
            raise ValueError("u must be a 2-D array when v is None.")
        return tsd.adtw(u, None, warp_penalty, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return tsd.adtw([u], [v], warp_penalty, n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return tsd.adtw(u, v, warp_penalty, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")

def msm_distance(u, v=None, n_jobs=1):
    """
    Computes the Move-Split-Merge (MSM) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise MSM distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Stefan, A. et al., The Move-Split-Merge Metric for Time Series, 2012.
    
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
    The MSM distance(s) between vectors/sets `u` and `v`.
    
    Examples
    --------
    >>> msm_distance([1, 0, 0], [0, 1, 0])
    2.0
    >>> msm_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[2.0, 4.0], [1.0, 3.0]])
    >>> msm_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 1.0], [1.0, 0.0]])
    """
    # Convert inputs to NumPy arrays for consistency and efficiency
    u = np.asarray(u)

    # If v is not provided, compute pairwise distances within u
    if v is None:
        if u.ndim != 2:
            raise ValueError("u must be a 2-D array when v is None.")
        return tsd.msm(u, None, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return tsd.msm([u], [v], n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return tsd.msm(u, v, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")

def twe_distance(u, v=None, stifness=0.001, penalty=1.0, n_jobs=1):
    """
    Computes the Time Warp Edit (TWE) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise TWE distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Marteau, P., Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching, 2008.
    
    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array. If provided, `v` should have the same shape as `u`.
    If `v` is None, pairwise distances within `u` are computed.
    stifness : double, optional
    Elasticity parameter (default is 0.001).
    penalty : double, optional
    Penalty for gap insertion/deletion (default is 1.0).
    n_jobs : int, optional
    Number of jobs to use for computation (default is 1).
    
    Returns
    -------
    distance : double or ndarray
    The TWE distance(s) between vectors/sets `u` and `v`.
    
    Examples
    --------
    >>> twe_distance([1, 0, 0], [0, 1, 0])
    4.0
    >>> twe_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[3.0, 7.0], [1.0, 5.0]])
    >>> twe_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 2.0], [2.0, 0.0]])
    """
    # Convert inputs to NumPy arrays for consistency and efficiency
    u = np.asarray(u)

    # If v is not provided, compute pairwise distances within u
    if v is None:
        if u.ndim != 2:
            raise ValueError("u must be a 2-D array when v is None.")
        return tsd.twe(u, None, stifness, penalty, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return tsd.twe([u], [v], stifness, penalty, n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return tsd.twe(u, v, stifness, penalty, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")

def sb_distance(u, v=None, n_jobs=1):
    """
    Computes the Shape-Based Distance (SBD) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise SBD distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Paparrizos, J. et al., k-Shape: Efficient and Accurate Clustering of Time Series, 2015.
    
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
    The SBD distance(s) between vectors/sets `u` and `v`.
    
    Examples
    --------
    >>> sb_distance([1, 0, 0], [0, 1, 0])
    1.4142135623730951
    >>> sb_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[1.41421356, 2.44948974], [1.        , 1.73205081]])
    >>> sb_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.        , 1.        ], [1.        , 0.        ]])
    """
    # Convert inputs to NumPy arrays for consistency and efficiency
    u = np.asarray(u)

    # If v is not provided, compute pairwise distances within u
    if v is None:
        if u.ndim != 2:
            raise ValueError("u must be a 2-D array when v is None.")
        return tsd.sbd(u, None, n_jobs)
    
    v = np.asarray(v)

    # If inputs are 1-D arrays, compute the distance directly
    if u.ndim == 1:
        return tsd.sbd([u], [v], n_jobs)[0][0]
    
    # If inputs are 2-D arrays, compute pairwise distances
    if u.ndim == 2:
        return tsd.sbd(u, v, n_jobs)
    
    # Raise an error if inputs are neither 1-D nor 2-D
    raise ValueError("Inputs must be either 1-D or 2-D arrays.")

__all__ = ['euclidean', 'erp', 'lcss', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'adtw', 'msm', 'twe', 'sbd']
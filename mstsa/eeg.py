"""mstsa.eeg — functions operating on raw EEG data (not microstate sequences)."""

import numpy as np
from numba import jit
from typing import Tuple

from .mstsa import ScalarFloatArray, _ffi_se, _lib_se


def complexity_ospl(data: ScalarFloatArray, fs: float, verbose: bool = False) -> Tuple[float, float, float, float]:
    """EEG complexity measures Omega, Sigma, Phi, and LC.
    NOTE: These metrics all use the natural logarithm.

    Parameters
    ----------
    data : ndarray of float, shape (N, C)
        EEG data with *N* samples and *C* channels.
    fs : float
        Sampling frequency in Hz.
    verbose : bool, optional
        Print intermediate values (default False).

    Returns
    -------
    Omega : float
        Effective dimensionality (exponential of the spectral entropy of the
        covariance eigenvalue distribution).
    Sigma : float
        Effective voltage per channel (root-mean-square amplitude).
    Phi : float
        Generalised frequency in Hz.
    LC : float
        Palus complexity coefficient.

    References
    ----------
    .. [1] Wackermann, J. (1996). Beyond mapping: estimating complexity of
           multichannel EEG recordings. *Acta Neurobiologiae Experimentalis*,
           56, 197-208.
    .. [2] Palus, M. et al. (1992). Spatio-temporal dynamics of human EEG.
           *Physica A*, 185, 433-438.
    """
    n_ch = data.shape[1] # number of channels

    # (1) PCA by diagonalization of the covariance matrix
    u = data - data.mean(axis=0)
    # m0, m1 from Wackermann
    # m0: squared voltages, sum over time, mean across channels
    m0 = np.mean(np.sum(u**2, axis=1))
    # temporal derivative of voltage at each channel
    du = np.diff(u, n=1, axis=0)
    # squared derivative, sum over time, mean across channels
    m1 = np.mean(np.sum(du**2, axis=1))
    Sigma = np.sqrt(m0/n_ch) # 'effective voltage per channel'
    Phi = fs/(2*np.pi)*np.sqrt(m1/m0) # 'generalized frequency'

    # Calculate the covariance matrix
    # NumPy default: row -> variable, col -> observations
    # here: row = time (observations), col = channels (variables)
    # => set rowvar = False in np.cov
    R = np.cov(data, rowvar=False)
    L, _ = np.linalg.eigh(R) # faster than eig
    L.sort() # ascending
    L = L[::-1] # descending
    L /= L.sum() # normalize
    lsum = np.sum(L[L>0]*np.log(L[L>0]))
    Omega = np.exp(-lsum)
    LC = -1/np.sum(np.log(L[L>0]))

    if verbose:
        print(f"m0 = {m0:.3f} ")
        print(f"m1 = {m1:.3f} ")
        print(f"Lsum = {lsum:.3f}")
        print(f"LC (Palus) = {LC:.3f}")
        print(f"Sigma = {Sigma:.3f} ")
        print(f"Phi = {Phi:.3f} ")
        print(f"Omega: {Omega:.3f}")

    return Omega, Sigma, Phi, LC


def sample_entropy_cont_py(x: ScalarFloatArray, m: int, tau: int, r: float,
                           base: str = '2') -> ScalarFloatArray:
    """Slow sample entropy for continuous signals, range of history lengths (legacy).

    Computes sample entropy for history lengths ``order = 0, 1, ..., m`` by
    running the standard double loop once per history length.  Kept for
    readability: the inner loop reads directly as the mathematical definition.

    At ``order=0`` no history is used at all; the returned value is the
    order-2 (collision) Renyi entropy of the marginal signal distribution,
    not the Shannon entropy.
    """
    if m < 0:
        raise ValueError("sample_entropy_cont_py: m must be >= 0.")
    _log = np.log2 if base == '2' else np.log
    n = len(x)
    e = np.zeros(m + 1)
    for order in range(m + 1):
        B = 0.0
        A = 0.0
        tmax = n - order*tau
        for i in range(tmax):
            for j in range(i+1, tmax):
                # distance norm: max_k |x_k - y_k|, start at k=0
                k = 0
                d = np.abs(x[i+k*tau] - x[j+k*tau])
                while (d < r) and (k < order):
                    k += 1
                    d = np.abs(x[i+k*tau] - x[j+k*tau])
                if k == order:
                    B += 1  # |x_k - y_k| < r for all k=0..order-1
                    if np.abs(x[i+order*tau] - x[j+order*tau]) < r:
                        A += 1  # |x_k - y_k| < r for k=order too
        e[order] = -_log(A/B)
    return e


@jit(nopython=True, cache=True)
def _fast_cont_core(y: ScalarFloatArray, m: int, r: float) -> Tuple[ScalarFloatArray, ScalarFloatArray]:
    """JIT core for sample_entropy_fast_cont_py; returns raw match counts A, B for orders 0..m."""
    n = len(y)
    # run[jj]: length of the current uninterrupted run of matching symbols
    # at lag jj+1 from the current reference index i; lastrun holds the
    # values from the previous iteration of i
    run = np.zeros(n)
    lastrun = np.zeros(n)
    A = np.zeros(m + 1)
    B = np.zeros(m + 1)
    for i in range(n - 1):
        nj = n - i - 1
        y1 = y[i]
        for jj in range(nj):
            j = (i + 1) + jj
            if np.abs(y[j] - y1) < r:
                run[jj] = lastrun[jj] + 1
                order_max = int(min(m + 1, run[jj]))
                for order in range(order_max):
                    A[order] += 1
                    if j < n - 1:
                        B[order] += 1
            else:
                run[jj] = 0
        for j in range(nj):
            lastrun[j] = run[j]
    return A, B


def sample_entropy_fast_cont_py(y: ScalarFloatArray, m: int, r: float,
                                base: str = '2') -> ScalarFloatArray:
    """Fast sample entropy for continuous signals over multiple history lengths.

    Uses a runs-tracking algorithm that computes estimates for all history
    lengths ``order = 0, ..., m`` in a single O(N^2) pass, rather than one pass
    per history length.

    Parameters
    ----------
    y : array_like of float, shape (N,)
        Input signal.
    m : int
        Maximum history length (>= 0).
    r : float
        Matching tolerance.
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    e : ndarray of float, shape (m+1,)
        Sample entropy estimates for history lengths ``order = 0, 1, ..., m``.
        ``e[order]`` is the classical Richman-Moorman ``SampEn(order)``;
        ``e[0]`` (no conditioning) is the order-2 Renyi collision entropy of
        the marginal signal distribution, not the Shannon entropy.

    References
    ----------
    .. [1] Richman, J. S., & Moorman, J. R. (2000). Physiological time-series
       analysis using approximate entropy and sample entropy.
       *Am J Physiol Heart Circ Physiol*, 278(6), H2039-H2049.
    .. [2] Lake, D. E., Richman, J. S., Griffin, M. P., & Moorman, J. R.
       (2002). Sample entropy analysis of neonatal heart rate variability.
       *Am J Physiol Regul Integr Comp Physiol*, 283(3), R789-R797.
    .. [3] PhysioNet sampen package, ``sampenc.m``.
       https://physionet.org/content/sampen/1.0.0/
    """
    if m < 0:
        raise ValueError("sample_entropy_fast_cont_py: m must be >= 0.")
    _log = np.log2 if base == '2' else np.log
    n = len(y)
    A, B = _fast_cont_core(np.asarray(y, dtype=np.float64), m, r)
    N = n * (n - 1) / 2
    B = np.hstack((N, B[:-1]))  # [N;B(1:m)]
    return -_log(A / B)


def sample_entropy_cont(x: ScalarFloatArray, m: int, tau: int, r: float,
                   base: str = '2') -> ScalarFloatArray:
    """Sample entropy of a continuous-valued signal, range of history lengths.

    Parameters
    ----------
    x : array_like of float, shape (N,)
        Input signal.
    m : int
        Maximum history length (>= 0); estimates are returned for
        ``order = 0, ..., m``.
    tau : int
        Time delay (subsample factor).
    r : float
        Matching tolerance (as a multiple of the signal standard deviation).
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    e : ndarray of float, shape (m+1,)
        Sample entropy estimates for history lengths ``order = 0, 1, ..., m``.
        ``e[order]`` is the classical Richman-Moorman ``SampEn(order)``;
        ``e[0]`` (no conditioning) is the order-2 Renyi collision entropy of
        the marginal signal distribution, not the Shannon entropy.

    Notes
    -----
    Uses the fast C extension ``_lib_se`` when available (computes in nats,
    converted here when ``base='2'``); falls back to
    ``sample_entropy_fast_cont_py`` otherwise.
    """
    if m < 0:
        raise ValueError("sample_entropy_cont: m must be >= 0.")
    n = len(x)
    x_norm = np.ascontiguousarray((x - x.mean()) / x.std(), dtype=np.float64)
    if _lib_se is not None:
        buf_se = np.zeros(m + 1, dtype=np.float64)
        se_c = _ffi_se.cast("double *", _ffi_se.from_buffer(buf_se))
        x_c  = _ffi_se.cast("double *", _ffi_se.from_buffer(x_norm))
        _lib_se.sample_entropy_cont_fast(se_c, x_c, n, m + 1, r)
        e = buf_se
        if base == '2':
            e /= np.log(2)
        return e
    else:
        return sample_entropy_fast_cont_py(x_norm, m, r, base=base)

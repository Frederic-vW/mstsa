#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mstsa: Microstate Time Series Analysis.

A Python package for information-theoretic and statistical analysis
of discrete symbolic time series, with a focus on EEG microstate sequences.
"""

import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np

from numba import jit
from scipy.signal import welch
from scipy.stats import chi2

from typing import Generic, Tuple, TypeVar

ScalarFloatType = TypeVar("ScalarFloatType", np.float32, np.float64)
ScalarIntType = TypeVar("ScalarIntType", np.int8, np.int16, np.int32, np.int64)
ScalarType = TypeVar(
    "ScalarType", np.int8, np.int16, np.int32, np.int64, np.float32, np.float64
)


class ScalarFloatArray(np.ndarray, Generic[ScalarFloatType]):
    pass


class ScalarIntArray(np.ndarray, Generic[ScalarIntType]):
    pass


class ScalarArray(np.ndarray, Generic[ScalarType]):
    pass

try:
    from mstsa._c_entropy import ffi as _ffi_c_entropy, lib as _lib_c_entropy
except ImportError:
    _ffi_c_entropy = None
    _lib_c_entropy = None
    warnings.warn(
        "mstsa: compiled extension '_c_entropy' not found; falling back to "
        "the pure-Python/Numba implementation, which is considerably slower. "
        "Reinstall mstsa with a working C compiler available to build the "
        "faster extension."
    )

try:
    from mstsa._se import ffi as _ffi_se, lib as _lib_se
except ImportError:
    _ffi_se = None
    _lib_se = None
    warnings.warn(
        "mstsa: compiled extension '_se' not found; sample entropy will use "
        "the pure-Python/Numba fallback, which is considerably slower. "
        "Reinstall mstsa with a working C compiler available to build the "
        "faster extension."
    )

try:
    from mstsa._lz76 import ffi as _ffi_lz76, lib as _lib_lz76
except ImportError:
    _ffi_lz76 = None
    _lib_lz76 = None
    warnings.warn(
        "mstsa: compiled extension '_lz76' not found; falling back to "
        "the pure-Python/Numba implementation, which is considerably slower. "
        "Reinstall mstsa with a working C compiler available to build the "
        "faster extension."
    )


def aif(x: ScalarIntArray, K: int, kmax: int,
        base: str = '2') -> ScalarFloatArray:
    """Auto-information function (AIF) — time-lagged mutual information.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    kmax : int
        Maximum time lag.
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    y : ndarray of float, shape (kmax,)
        AIF coefficients ``I(X_t; X_{t+k})`` for lags ``k = 0, ..., kmax-1``.

    Notes
    -----
    The AIF at lag *k* equals the mutual information between the process
    and its time-shifted copy:

    .. math:: I(X_t; X_{t+k}) = H(X_t) + H(X_{t+k}) - H(X_t, X_{t+k})

    Examples
    --------
    >>> import numpy as np
    >>> from mstsa import aif
    >>> rng = np.random.default_rng(0)
    >>> x = rng.integers(0, 4, size=1000)
    >>> y = aif(x, K=4, kmax=10)

    References
    ----------
    .. [1] von Wegner, F. et al. (2017). Mutual information identifies
           spurious Hurst phenomena in resting state EEG.
           *NeuroImage*, 152, 98-108.
    .. [2] von Wegner, F. et al. (2017). Information-theoretical analysis of
           resting state EEG microstate sequences — non-Markovianity,
           non-stationarity and periodicities.
           *NeuroImage*, 158, 99-111.
    .. [3] von Wegner, F. et al. (2018). EEG microstate sequences from
           different clustering algorithms are information-theoretically
           invariant. *Frontiers in Computational Neuroscience*, 12, 70.
    """
    n = len(x)
    y = np.zeros(kmax)
    for k in range(kmax):
        #if (k%10 == 0):
        #    print(f"\t(aif)  {k:d}/{kmax:d}", end="\r")
        nmax = n-k
        e1 = h1(x[:nmax], K, base=base)
        e2 = h1(x[k:k+nmax], K, base=base)
        e12 = h2(x[:nmax], x[k:k+nmax], K, base=base)
        y[k] = e1 + e2 - e12
    #print()
    return y


def ais(x: ScalarIntArray, K: int, k: int = 1, base: str = '2') -> float:
    """Active information storage (AIS).

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    k : int, optional
        History length (default 1).
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    a : float
        Active information storage :math:`I(X_{n+1}; X_n^{(k)})`.

    Notes
    -----
    AIS decomposes as:

    .. math::

        I(X_{n+1}; X_n^{(k)})
        = H(X_{n+1}) + H(X_n^{(k)}) - H(X_{n+1}^{(k+1)})

    Examples
    --------
    >>> import numpy as np
    >>> from mstsa import ais
    >>> rng = np.random.default_rng(0)
    >>> x = rng.integers(0, 4, size=1000)
    >>> ais(x, K=4, k=2)

    References
    ----------
    .. [1] Lizier, J. T. (2014). JIDT: An information-theoretic toolkit for
           studying the dynamics of complex systems. *Frontiers in Robotics and AI*, 1, 11.
    """
    n = len(x)
    e1 = hk(x, K, 1, base=base) # H(X[n+1])
    e2 = hk(x, K, k, base=base) # H(X[n:n-k])
    e3 = hk(x, K, k+1, base=base) # H(X[n+1], X[n:n-k])
    a = e1 + e2 - e3 # I(X[n+1]|X[n:n-k])
    return a


@jit(nopython=True, cache=True)
def _dfa_impl(y: ScalarFloatArray, nx: int, ls: ScalarIntArray, n: int) -> ScalarFloatArray:
    """Numba core of :func:`dfa`: RMS fluctuation at each scale."""
    fs = np.zeros(n) # fluctuations
    for i in range(n):
        l = ls[i]
        nb = nx // l
        # analytical sums for x = 0..l-1, identical for every block
        sx    = l * (l - 1) / 2.0
        sxx   = l * (l - 1) * (2*l - 1) / 6.0
        denom = l * sxx - sx * sx
        fluct = 0.0
        for b in range(nb):
            sy = 0.0; sxy = 0.0
            for j in range(l):
                v = y[b*l + j]
                sy  += v
                sxy += j * v
            slope     = (l * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / l
            for j in range(l):
                r = y[b*l + j] - (slope * j + intercept)
                fluct += r * r
        fs[i] = np.sqrt(fluct / (nb * l))
    return fs


def dfa(x: ScalarFloatArray, lmin: int, lmax: int, fitmin: float, fitmax: float,
        nsteps: int, doplot: bool = False) -> float:
    """Detrended fluctuation analysis (DFA).

    Parameters
    ----------
    x : array_like of float, shape (N,)
        Input time series.
    lmin : int
        Shortest time scale (samples).
    lmax : int
        Longest time scale (samples).
    fitmin : float
        Minimum scale for the log-log linear fit.
    fitmax : float
        Maximum scale for the log-log linear fit.
    nsteps : int
        Number of logarithmically spaced scales between ``lmin`` and ``lmax``.
    doplot : bool, optional
        Display a log-log plot of fluctuations and the fitted line
        (default False).

    Returns
    -------
    h_dfa : float
        Hurst exponent estimate (slope of the log-log fluctuation curve).

    References
    ----------
    .. [1] Peng, C.-K. et al. (1994). Mosaic organization of DNA nucleotides.
           *Physical Review E*, 49, 1685.
    """
    nx = len(x)
    y  = np.cumsum(x - np.mean(x))
    ls = np.logspace(start=np.log2(lmin), stop=np.log2(lmax), num=nsteps,
                     endpoint=True, base=2, dtype=np.int64)
    ls = np.unique(ls[ls > 1])
    fs = _dfa_impl(y, nx, ls, len(ls))
    i_fitmin = np.argmin((ls - fitmin)**2)
    i_fitmax = np.argmin((ls - fitmax)**2)
    p_fit    = np.polyfit(np.log2(ls[i_fitmin:i_fitmax]),
                          np.log2(fs[i_fitmin:i_fitmax]), 1)
    h_dfa = p_fit[0]
    if doplot:
        fsize = 16
        p_txt = {'fontsize':fsize, 'fontweight':'normal'}
        fig = plt.figure(1, figsize=(4,4))
        ax = plt.gca()
        ax.loglog(ls, fs, 'ok', ms=8, alpha=0.5)
        #ax.loglog(ls_fit, 10**(p_fit[1]) * ls_fit**h_dfa, '-b', linewidth=2)
        #ax.loglog(ls_fit, 2**(p_fit[1]) * ls_fit**h_dfa, '-b', linewidth=2)
        ax.loglog(ls, 2**(p_fit[1]) * ls**h_dfa, '-b', linewidth=2)
        #ax.loglog(ls_fit, 2**(p_fit[1]) * ls_fit**h_dfa, '-b', linewidth=4)
        ax.axvline(fitmin, linewidth=2)
        ax.axvline(fitmax, linewidth=2)
        #ax.grid()
        ax.set_xlabel("scale l", **p_txt)
        ax.set_ylabel("fluct. F(l)", **p_txt)
        ax.tick_params(axis='both', which='major', labelsize=fsize)
        plt.title(r"$H_{DFA} = $" + f"{h_dfa:.3f}", **p_txt)
        plt.show()
    return h_dfa


def rescaled_range(x: ScalarFloatArray) -> float:
    """Rescaled range (R/S) analysis.

    Parameters
    ----------
    x : array_like of float, shape (N,)
        Input time series.

    Returns
    -------
    h_rs : float
        Hurst exponent estimate.

    References
    ----------
    .. [1] Hurst, H. E. (1951). Long-term storage capacity of reservoirs.
           *Transactions of the American Society of Civil Engineers*, 116, 770–779.
    """
    w = np.cumsum(x - np.mean(x))
    RS = (np.max(w) - np.min(w)) / np.std(x)
    h_rs = np.log(RS) / np.log(len(x))
    return h_rs


def hurst_exponents_dfa(x: ScalarIntArray, lmin: int, lmax: int,
                    fitmin: float, fitmax: float,
                    nsteps: int) -> ScalarFloatArray:
    """DFA Hurst exponents of the characteristic microstate indicator functions.

    For each symbol *k*, constructs the binary indicator sequence
    ``y_k[t] = 1 if x[t] == k else 0`` and estimates its Hurst exponent
    via detrended fluctuation analysis.  Analogous to :func:`spectra`, which
    applies Welch's method to the same indicator functions.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    lmin : int
        Shortest DFA time scale (samples).
    lmax : int
        Longest DFA time scale (samples).
    fitmin : float
        Minimum scale for the log-log linear fit.
    fitmax : float
        Maximum scale for the log-log linear fit.
    nsteps : int
        Number of logarithmically spaced scales between ``lmin`` and ``lmax``.

    Returns
    -------
    h : ndarray of float, shape (K,)
        Hurst exponent for each symbol's indicator function.
    """
    x = np.asarray(x)
    K = len(np.unique(x))
    h = np.zeros(K)
    for k in range(K):
        h[k] = dfa((x == k).astype(np.float64), lmin, lmax, fitmin, fitmax, nsteps)
    return h


def hurst_exponents_rs(x: ScalarIntArray) -> ScalarFloatArray:
    """Hurst coefficient using the rescaled range (R/S) method.'''

    Args:
        x: data
    Returns:
        H: estimated Hurst exponent
    """
    x = np.asarray(x)
    K = len(np.unique(x))
    h = np.zeros(K)
    for k in range(K):
        h[k] = rescaled_range((x == k).astype(np.float64))
    return h


def _diffusion_entropy_scale(y: ScalarFloatArray, lmin: int, lmax: int,
                             fitmin: float, fitmax: float, nsteps: int) -> float:
    """Numpy core of :func:`diffusion_entropy`: DEA scaling exponent of one indicator."""
    ls = np.logspace(start=np.log2(lmin), stop=np.log2(lmax), num=nsteps,
                     endpoint=True, base=2, dtype=np.int64)
    ls = np.unique(ls[ls > 1])
    ss = np.zeros(len(ls))
    for i, l in enumerate(ls):
        disp = np.convolve(y, np.ones(l), mode='valid')
        counts = np.bincount(np.round(disp).astype(np.int64))
        p = counts[counts > 0] / counts.sum()
        ss[i] = -np.sum(p * np.log2(p))
    i_fitmin = np.argmin((ls - fitmin)**2)
    i_fitmax = np.argmin((ls - fitmax)**2)
    p_fit = np.polyfit(np.log2(ls[i_fitmin:i_fitmax]),
                       ss[i_fitmin:i_fitmax], 1)
    return p_fit[0]


def diffusion_entropy(x: ScalarIntArray, lmin: int, lmax: int,
                      fitmin: float, fitmax: float,
                      nsteps: int) -> ScalarFloatArray:
    """Diffusion entropy analysis (DEA) of the characteristic microstate indicator functions.

    For each symbol *k*, constructs the binary indicator sequence
    ``y_k[t] = 1 if x[t] == k else 0``. For each scale *l*, the sequence is
    swept with an overlapping window of length *l* and the displacement
    (window sum) is recorded at every position, giving an empirical
    distribution :math:`p(\cdot, l)` whose Shannon entropy :math:`S(l)` is
    computed. The DEA scaling exponent is the slope of :math:`S(l)` against
    :math:`\log_2 l`, analogous to :func:`hurst_exponents_dfa` and
    :func:`spectra`, which apply DFA and Welch's method (respectively) to
    the same indicator functions.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    lmin : int
        Shortest window length (samples).
    lmax : int
        Longest window length (samples).
    fitmin : float
        Minimum scale for the log-log linear fit.
    fitmax : float
        Maximum scale for the log-log linear fit.
    nsteps : int
        Number of logarithmically spaced scales between ``lmin`` and ``lmax``.

    Returns
    -------
    delta : ndarray of float, shape (K,)
        Diffusion entropy scaling exponent for each symbol's indicator
        function (0.5 for uncorrelated, normal-diffusion behaviour).

    References
    ----------
    .. [1] Scafetta, N., & Grigolini, P. (2002). Scaling detection in time
           series: Diffusion entropy analysis. *Physical Review E*, 66, 036130.
    """
    x = np.asarray(x)
    K = len(np.unique(x))
    delta = np.zeros(K)
    for k in range(K):
        delta[k] = _diffusion_entropy_scale((x == k).astype(np.float64),
                                            lmin, lmax, fitmin, fitmax, nsteps)
    return delta


def embedded_process(x: ScalarIntArray, exclude_first: bool = True,
                     exclude_last: bool = True) -> ScalarIntArray:
    """Extract the embedded jump process from a symbolic sequence.

    Removes repeated consecutive symbols so that, e.g.,
    ``AAABBCCCCDDAA`` becomes ``ABCDA``, and returns both the jump
    process and the (possibly trimmed) input sequence.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence, possibly with repeated consecutive symbols.
    exclude_first : bool, optional
        Trim the initial run so the sequence starts at a transition
        (default True).
    exclude_last : bool, optional
        Trim the final run so the sequence ends at a transition
        (default True).

    Returns
    -------
    y : ndarray of int
        Jump process (one symbol per sojourn).
    x : ndarray of int
        Trimmed input sequence.
    """
    # apply boundary conditions
    if (exclude_first or exclude_last):
        # indices right after the switch, i.e. first index of new state
        switch_indices = 1 + np.where(np.diff(x))[0]
        first_switch = switch_indices[0]
        last_switch = switch_indices[-1]
    
    exclude_both = exclude_first and exclude_last
    if exclude_both:
        x = x[first_switch:last_switch]
    elif exclude_first:
        x = x[first_switch:]
    elif exclude_last:
        x = x[:last_switch]
    else:
        pass

    # re-calculate switch indices for (possibly) truncated sequence
    # insert 0 to capture first state
    switch_indices = np.hstack(([0], 1+np.where(np.diff(x))[0]))
    
    # jump process
    y = x[switch_indices]
    return y, x


def entropy_rate(x: ScalarIntArray, K: int, kmax: int,
                 doplot: bool = False,
                 base: str = '2') -> "tuple[float, float]":
    """Estimate the entropy rate and excess entropy by linear regression.

    Fits :math:`H(X^{(k)}) = h \cdot k + E` over block lengths
    ``k = 1, ..., kmax+1``, where :math:`X^{(k)}` is a block of *k*
    consecutive symbols (so :math:`H(X^{(k)})` is the joint entropy computed
    by :func:`hk`), *h* is the entropy rate, and *E* is the excess entropy.

    ``kmax`` is the conditioning history length: the classical order-``kmax``
    conditional entropy :math:`H(X_{t+1} \mid X_t, \ldots, X_{t-kmax+1}) =
    H(X^{(kmax+1)}) - H(X^{(kmax)})` is the slope between the last two points
    of this fit.  Rather than returning that single finite difference,
    ``entropy_rate`` fits a line across the whole range of block lengths
    ``1, ..., kmax+1``, giving a more robust estimate of the same asymptotic
    slope (following the standard approach in the literature, e.g. [1]).

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    kmax : int
        Maximum conditioning history length (>= 1); joint entropies are
        computed for block lengths ``1, ..., kmax+1``.
    doplot : bool, optional
        Display joint-entropy values and the fitted line (default False).
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    er : float
        Entropy rate estimate (bits per sample if ``base='2'``, nats if ``base='e'``).
    ee : float
        Excess entropy (intercept of the linear fit).

    Raises
    ------
    ValueError
        If ``kmax < 1``.

    References
    ----------
    .. [1] Cover, T. M. & Thomas, J. A. (2006). *Elements of Information
           Theory* (2nd ed.). Wiley.
    .. [2] von Wegner, F. et al. (2024). Complexity measures for EEG microstate
           sequences: concepts and algorithms.
           *Brain Topography*, 37(2), 296-311.
    """
    if kmax < 1:
        raise ValueError("entropy_rate: kmax must be >= 1 (history length 0 is undefined).")
    nblocks = kmax + 1
    h_ = np.zeros(nblocks)
    for k in range(nblocks):
        h_[k] = hk(x, K, k+1, base=base)
    ks = np.arange(1, nblocks+1)
    er, ee = np.polyfit(ks, h_, 1)
    if doplot:
        fsize = 16
        plt.figure(figsize=(5,5))
        ax = plt.gca()
        ax.plot(ks, h_, 'ok', ms=12, alpha=0.6)
        ax.plot(ks, er*ks+ee, '-b', label='fit')
        ax.set_xlabel("block length k", fontsize=fsize)
        ax.set_ylabel("joint entropy "+r"$H\left( \mathbf{X}_n^{(k)} \right)$",\
                      fontsize=fsize)
        ax.tick_params(axis='both', which='major', labelsize=fsize)
        ax.set_title("Entropy rate: " + r"$h_X$" + f" = {er:.3f} bit/sample", \
                     fontsize=fsize)
        ax.grid()
        ax.legend(fontsize=fsize)
        plt.tight_layout()
        plt.show()
    return er, ee


def h1(x: ScalarIntArray, K: int, base: str = '2') -> float:
    """Shannon entropy of a symbolic sequence.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    h : float
        Shannon entropy :math:`H(X) = -\sum_i p_i \log p_i`.

    Examples
    --------
    >>> import numpy as np
    >>> from mstsa import h1
    >>> rng = np.random.default_rng(0)
    >>> x = rng.integers(0, 4, size=1000)
    >>> h1(x, K=4)  # close to 2.0 bits for uniform distribution
    """
    _log = np.log2 if base == '2' else np.log
    n = len(x)
    p = np.zeros(K) # symbol distribution
    for t in range(n):
        p[x[t]] += 1.0
    p /= n
    h = -np.sum(p[p>0]*_log(p[p>0]))
    return h


def h2(x: ScalarIntArray, y: ScalarIntArray, K: int, base: str = '2') -> float:
    """Joint Shannon entropy of two symbolic sequences.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        First symbolic sequence with integer labels in ``[0, K)``.
    y : array_like of int, shape (M,)
        Second symbolic sequence with integer labels in ``[0, K)``.
        If ``len(y) != len(x)`` a warning is issued and the shorter length
        is used.
    K : int
        Number of distinct symbols (shared alphabet).
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    h : float
        Joint Shannon entropy :math:`H(X, Y)`.
    """
    _log = np.log2 if base == '2' else np.log
    if (len(x) != len(y)):
        warnings.warn("h2: x and y have unequal lengths; using the shorter.")
    n = min([len(x), len(y)])
    p = np.zeros((K, K)) # joint distribution
    for t in range(n):
        p[x[t],y[t]] += 1.0
    p /= n
    h = -np.sum(p[p>0]*_log(p[p>0]))
    return h


def hk(x: ScalarIntArray, K: int, k: int,
       bias_correction: bool = False,
       base: str = '2') -> float:
    """Joint Shannon entropy of k consecutive symbols.

    Computes :math:`H(X_t, X_{t+1}, \ldots, X_{t+k-1})` from the empirical
    *k*-gram distribution.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    k : int
        History length (number of consecutive symbols in the joint block).
    bias_correction : bool, optional
        Apply the Miller-Madow bias correction (default False).
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    h : float
        Joint Shannon entropy :math:`H_k`.

    Notes
    -----
    Uses the C extension ``_lib_c_entropy`` when available (always computes in
    nats, converted to bits when ``base='2'``); otherwise a pure-NumPy fallback
    is used.
    """
    _log = np.log2 if base == '2' else np.log
    n = len(x)
    if _lib_c_entropy is not None:
        buf = np.ascontiguousarray(x, dtype=np.int32)
        x_c = _ffi_c_entropy.cast("int *", _ffi_c_entropy.from_buffer(buf))
        h = _lib_c_entropy.c_entropy(x_c, n, K, k)
        if base == '2':
            h /= np.log(2)
    else:
        f = np.zeros(tuple(k * [K]))
        for t in range(n - k + 1):
            f[tuple(x[t:t + k])] += 1.0
        f /= (n - k)
        h = -np.sum(f[f > 0] * _log(f[f > 0]))

    if bias_correction:
        # Miller correction
        h += (K - 1) / (2 * (n - k))
    return h


def topological_entropy(x: ScalarIntArray, K: int, k: int, base: str = '2') -> float:
    """Topological entropy of a symbolic sequence.

    Estimates the growth rate of the number of distinct length-*k* words
    (blocks) appearing in ``x``, :math:`h_{top}(k) = \\frac{1}{k}\\log N(k)`,
    where ``N(k)`` is the number of distinct *k*-grams observed. Unlike
    :func:`hk`, which weights each block by its empirical probability,
    topological entropy only records whether a block occurs at all, and so
    upper-bounds the metric (Shannon) entropy rate.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    k : int
        Block length (number of consecutive symbols).
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    h : float
        Topological entropy estimate :math:`h_{top}(k)`.

    References
    ----------
    .. [1] Adler, R. L., Konheim, A. G., & McAndrew, M. H. (1965). Topological
           entropy. *Transactions of the American Mathematical Society*, 114, 309-319.
    """
    _log = np.log2 if base == '2' else np.log
    n = len(x)
    f = np.zeros(tuple(k * [K]))
    for t in range(n - k + 1):
        f[tuple(x[t:t + k])] += 1.0
    N = np.count_nonzero(f)
    h = _log(N) / k
    return h


def dispersion_entropy(x: ScalarIntArray, K: int, k: int,
                       bias_correction: bool = False,
                       base: str = '2') -> float:
    """Dispersion entropy of a symbolic sequence.

    For already-classified (symbolic) data, dispersion entropy reduces to
    the joint (block) Shannon entropy of *k* consecutive symbols, so this
    simply forwards to :func:`hk`.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols (dispersion classes).
    k : int
        Embedding dimension (number of consecutive symbols per pattern).
    bias_correction : bool, optional
        Apply the Miller-Madow bias correction (default False).
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    h : float
        Dispersion entropy, equal to :func:`hk`'s joint Shannon entropy
        :math:`H_k`.

    References
    ----------
    .. [1] Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A measure
           for time-series analysis. *IEEE Signal Processing Letters*, 23(5), 610-614.
    """
    return hk(x, K, k, bias_correction=bias_correction, base=base)


def sojourn_time_histograms(x: ScalarIntArray, K: int) -> list:
    """Compute sojourn-time distributions for each symbol.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.

    Returns
    -------
    tau_dist : list of list of float, length K
        ``tau_dist[s]`` is a histogram of sojourn lengths for symbol *s*,
        indexed from 1 (index 0 corresponds to duration 1).
    """
    x = np.asarray(x)
    n = len(x)
    sw = np.concatenate(([0], 1 + np.where(np.diff(x))[0], [n])) # switch indices
    run_lengths = np.diff(sw)
    run_symbols = x[sw[:-1]]
    return [
        np.bincount(run_lengths[run_symbols == k] - 1).astype(float).tolist()
        for k in range(K)
    ]


def sojourn_times_unordered(x: ScalarIntArray, K: int) -> list:
    """Collect all sojourn times for each symbol as unsorted lists.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.

    Returns
    -------
    tau_list : list of list of float, length K
        ``tau_list[s]`` contains all sojourn durations of symbol *s* in the
        order they appear in *x*.

    Notes
    -----
    For the sequence ``...0 0 3 3 1 1 1 0 2 2 2 2 0 0 3 3 3 0...`` the
    result is ``[[2.0, 1.0, 2.0, 1.0], [3.0], [4.0], [2.0, 3.0]]``.

    Examples
    --------
    >>> x = [0, 0, 3, 3, 1, 1, 1, 0, 2, 2, 2, 2, 0, 0, 3, 3, 3, 0]
    >>> sojourn_times_unordered(x, K=4)
    [[2.0, 1.0, 2.0, 1.0], [3.0], [4.0], [2.0, 3.0]]
    """
    x = np.asarray(x)
    n = len(x)
    sw = np.concatenate(([0], 1 + np.where(np.diff(x))[0], [n])) # switch indices
    run_lengths = np.diff(sw).astype(float)
    run_symbols = x[sw[:-1]]
    return [run_lengths[run_symbols == k].tolist() for k in range(K)]


def dur_occ_cov(x: ScalarIntArray,
             fs: float,
             exclude_first: bool = True,
             exclude_last: bool = True) -> Tuple[dict, ScalarFloatArray, ScalarFloatArray]:
    """Compute microstate duration, occurrence, and coverage from a sequence.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    fs : float
        Sampling rate in Hz.
    exclude_first : bool, optional
        Exclude the initial (possibly partial) microstate (default True).
    exclude_last : bool, optional
        Exclude the final (possibly partial) microstate (default True).

    Returns
    -------
    dur_dict : dict
        Sojourn durations (ms) for each symbol.  Keys are symbol indices
        ``0, 1, ..., K-1``; values are lists of individual durations.
    occ : ndarray of float, shape (K,)
        Occurrence rate (events per second) for each symbol.
    cov : ndarray of float, shape (K,)
        Fractional coverage (0–1) for each symbol.
    """
    #fs, dt = 1, 1 # for testing of short sequences
    dt = 1000./fs
    K = len(np.unique(x))
    
    # apply boundary conditions (first and last state have unknown duration)
    if (exclude_first or exclude_last):
        # indices right after the switch, i.e. first index of new state
        switch_indices = 1 + np.where(np.diff(x))[0]
        i_first = switch_indices[0]
        i_last = switch_indices[-1]
    exclude_both = exclude_first and exclude_last
    if exclude_both:
        x = x[i_first:i_last]
    elif exclude_first:
        x = x[i_first:]
    elif exclude_last:
        x = x[:i_last]
    else:
        pass

    nx = len(x)
    times = dt*np.arange(nx) # time axis in msec
    # re-calculate switch indices for (possibly) truncated sequence
    # insert 0 to capture first state
    switch_indices = np.hstack(([0], 1+np.where(np.diff(x))[0]))
    
    # jump process
    y = x[switch_indices]

    # (1) DURATION
    # state start/stop times
    ts_on = times[switch_indices]
    ts_off = np.hstack((times[switch_indices[1:]], [dt*len(x)]))
    # durations = sojourn times
    dur_list = ts_off - ts_on # durations in order of natural occurrence
    dur_dict = {
        k : [l for i, l in enumerate(dur_list) if y[i]==k] for k in range(K)
    } # durations as dict, ordered by state (=dict key)

    # (2) OCCURRENCE
    total_time = nx/fs # total sequence duration in sec
    occ = np.array([np.sum(y==k)/total_time for k in range(K)])

    # (3) COVERAGE
    cov = np.array([np.sum(x==k)/nx for k in range(K)])

    return dur_dict, occ, cov


def paif(x: ScalarIntArray, K: int, kmax: int,
         base: str = '2') -> ScalarFloatArray:
    """Partial auto-information function (PAIF).

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    kmax : int
        Maximum time lag.
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    z : ndarray of float, shape (kmax,)
        PAIF coefficients.

    Notes
    -----
    The PAIF at lag *k* is computed as:

    .. math:: \\text{PAIF}(k) = 2 H_k - H_{k-1} - H_{k+1}

    The first coefficient (lag 0) equals the Shannon entropy.

    References
    ----------
    .. [1] von Wegner, F. (2018). Partial autoinformation to characterize
           symbolic sequences. *Frontiers in Physiology*, 9, 1382.
    .. [2] von Wegner, F. et al. (2024). Complexity measures for EEG microstate
           sequences: concepts and algorithms.
           *Brain Topography*, 37(2), 296-311.
    """
    n = len(x)
    y = aif(x, K, 2, base=base) # AIF, for first two coefficients
    z = np.zeros(kmax) # PAIF
    z[0] = y[0]
    #z[1] = y[1]
    for k in range(1,kmax):
        #if (k%10 == 0):
        #    print(f"\t(paif) {k:d}/{kmax:d}")
        #H(x,K,p=l,n=0)-H(x,K,p=l-1,n=0)-H(x,K,p=l,n=1)+H(x,K,p=l-1,n=1)
        h1 = hk(x, K, k, base=base)
        h2 = hk(x, K, k-1, base=base)
        h3 = hk(x, K, k+1, base=base)
        #h4 = hk(x,K,k)
        z[k] = 2*h1 - h2 - h3
    return z


def partitions(K: int, verbose: bool = False) -> list:
    """
    Generate partitions to use with microstate sequence random walk analysis
    parts = [[[0,1],[2,3]], [[0,2],[1,3]], [[0,3],[1,2]]] # for K=4 microstates
    
    """
    # auxiliary function, returns complement set as list
    f_C = lambda set_, subset_: list(set_.difference(subset_))
    # make partitions
    full_set = set(range(K))
    if verbose:
        print("\nFull set: ", full_set)
    combinations = itertools.combinations(full_set, K//2)
    parts0 = []
    for i, comb in enumerate(combinations):
        comb = list(comb) # cast tuple --> list
        comb_C = f_C(full_set, comb) # complement
        if (comb in parts0) or (comb_C in parts0):
            pass
        else:
            parts0.append(comb)
    parts = [[part, f_C(full_set, part)] for part in parts0]
    if verbose:
        for i, part in enumerate(parts):
            print(f"Partition {i*1:d}: {part}")
    return parts


def randomwalk(x: ScalarIntArray, part: list) -> ScalarFloatArray:
    """Convert a symbolic sequence to a ±1 random walk via a binary partition.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence.
    part : list of two lists
        Binary partition of the symbol set.  Symbols in ``part[0]`` map to
        ``+1``; symbols in ``part[1]`` map to ``-1``.

    Returns
    -------
    r : ndarray of float, shape (N,)
        ±1 indicator sequence (cumulative sum gives the random walk).

    References
    ----------
    .. [1] Van De Ville, D. et al. (2010). EEG microstate sequences in
           healthy humans at rest reveal scale-free dynamics.
           *PNAS*, 107, 18179-18184.
    .. [2] von Wegner, F. et al. (2016). Analytical and empirical fluctuation
           functions of the EEG microstate random walk — short-range vs.
           long-range correlations. *NeuroImage*, 141, 442-451.
    """
    r = np.array( [1.0 if s in part[0] else -1.0 for s in x] )
    #rw = np.cumsum(r)
    return r


def renyi_entropy(p: ScalarFloatArray, a: float, base: str = '2') -> float:
    """Rényi entropy of order *a*.

    Parameters
    ----------
    p : array_like of float, shape (K,)
        Probability distribution (must sum to 1).
    a : float
        Rényi order.  Must satisfy ``a > 0`` and ``a != 1``.
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    H_a : float
        Rényi entropy
        :math:`H_a(X) = \\frac{1}{1-a} \log\!\left(\sum_i p_i^a\\right)`.

    Raises
    ------
    ValueError
        If ``a <= 0`` or ``a == 1``.
    """
    _log = np.log2 if base == '2' else np.log
    if a <= 0 or np.abs(a - 1) < 1e-6:
        raise ValueError("Renyi parameter a must be positive and not equal to 1.")
    H_a = 1.0 / (1.0 - a) * _log(np.sum(p ** a))
    return H_a


def sample_entropy_disc_py(x: ScalarIntArray, m: int, tau: int,
                           base: str = '2') -> ScalarFloatArray:
    """Slow sample entropy for discrete signals, range of history lengths (legacy).

    Computes sample entropy for history lengths ``order = 0, 1, ..., m`` by
    running the standard double loop once per history length.  Kept for
    readability: the inner loop reads directly as the mathematical definition.

    At ``order=0`` no history is used at all; the returned value is the
    order-2 (collision) Renyi entropy of the marginal symbol distribution,
    not the Shannon entropy.
    """
    if m < 0:
        raise ValueError("sample_entropy_disc_py: m must be >= 0.")
    _log = np.log2 if base == '2' else np.log
    n = len(x)
    e = np.zeros(m + 1)
    for order in range(m + 1):
        B = 0.0
        A = 0.0
        tmax = n - order*tau
        for i in range(tmax):
            for j in range(i+1, tmax):
                # exact-match norm: max_k (x_k - y_k) == 0, start at k=0
                k = 0
                d = x[i+k*tau] - x[j+k*tau]
                while (d == 0) and (k < order):
                    k += 1
                    d = x[i+k*tau] - x[j+k*tau]
                if k == order:
                    B += 1  # x_k == y_k for all k=0..order-1
                    if x[i+order*tau] - x[j+order*tau] == 0:
                        A += 1  # x_k == y_k for k=order too
        e[order] = -_log(A/B)
    return e


@jit(nopython=True, cache=True)
def _fast_disc_core(y: ScalarIntArray, m: int) -> Tuple[ScalarFloatArray, ScalarFloatArray]:
    """JIT core for sample_entropy_fast_disc_py; returns raw match counts A, B for orders 0..m."""
    n = len(y)
    # run[jj]: length of the current uninterrupted run of matching symbols
    # at lag jj+1 from the current reference index i
    run = np.zeros(n)
    lastrun = np.zeros(n)
    A = np.zeros(m + 1)
    B = np.zeros(m + 1)
    for i in range(n - 1):
        nj = n - i - 1
        y1 = y[i]
        for jj in range(nj):
            j = (i + 1) + jj
            if y[j] == y1:
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


def sample_entropy_fast_disc_py(y: ScalarIntArray, m: int, base: str = '2') -> ScalarFloatArray:
    """Fast sample entropy for discrete sequences over multiple history lengths.

    Uses a runs-tracking algorithm that computes estimates for all history
    lengths ``order = 0, ..., m`` in a single O(N^2) pass.  Matching is by
    exact equality (``y[j] == y[i]``), which is ~2x faster than the
    tolerance-based comparison used for continuous signals.

    Parameters
    ----------
    y : array_like of int, shape (N,)
        Input discrete sequence.
    m : int
        Maximum history length (>= 0).
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    e : ndarray of float, shape (m+1,)
        Sample entropy estimates for history lengths ``order = 0, 1, ..., m``.
        ``e[order]`` is the classical Richman-Moorman ``SampEn(order)``;
        ``e[0]`` (no conditioning) is the order-2 Renyi collision entropy of
        the marginal symbol distribution, not the Shannon entropy.

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
        raise ValueError("sample_entropy_fast_disc_py: m must be >= 0.")
    _log = np.log2 if base == '2' else np.log
    n = len(y)
    A, B = _fast_disc_core(np.asarray(y, dtype=np.int64), m)
    N = n * (n - 1) / 2
    B = np.hstack((N, B[:-1]))  # [N;B(1:m)]
    return -_log(A / B)


def sample_entropy_disc(y: ScalarIntArray, m: int,
                        base: str = '2') -> ScalarFloatArray:
    """Sample entropy of a discrete microstate sequence, range of history lengths.

    Parameters
    ----------
    y : array_like of int, shape (N,)
        Input discrete sequence.
    m : int
        Maximum history length (>= 0); estimates are returned for
        ``order = 0, ..., m``.
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    e : ndarray of float, shape (m+1,)
        Sample entropy estimates for history lengths ``order = 0, 1, ..., m``.
        ``e[order]`` is the classical Richman-Moorman ``SampEn(order)``;
        ``e[0]`` (no conditioning) is the order-2 Renyi collision entropy of
        the marginal symbol distribution, not the Shannon entropy.

    Notes
    -----
    Uses the fast C extension ``_lib_se`` when available (computes in nats,
    converted here when ``base='2'``); falls back to
    ``sample_entropy_fast_disc_py`` otherwise.
    """
    if m < 0:
        raise ValueError("sample_entropy_disc: m must be >= 0.")
    y_long = np.ascontiguousarray(y, dtype=np.int64)
    n = len(y_long)
    if _lib_se is not None:
        buf_se = np.zeros(m + 1, dtype=np.float64)
        se_c = _ffi_se.cast("double *", _ffi_se.from_buffer(buf_se))
        y_c  = _ffi_se.cast("long *",   _ffi_se.from_buffer(y_long))
        _lib_se.sample_entropy_disc_fast(se_c, y_c, n, m + 1)
        e = buf_se
        if base == '2':
            e /= np.log(2)
        return e
    else:
        return sample_entropy_fast_disc_py(y_long, m, base=base)


def spectra(x: ScalarIntArray, fs: float, nperseg: int) -> Tuple[ScalarFloatArray, ScalarFloatArray]:
    """Power spectral densities of the characteristic microstate functions.

    For each symbol *k*, the characteristic function is the binary indicator
    sequence ``y_k[t] = 1 if x[t] == k else 0``.  Welch's method is used to
    estimate the PSD of each indicator.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Length of each Welch segment.

    Returns
    -------
    freqs : ndarray of float, shape (nperseg // 2 + 1,)
        Frequency axis in Hz.
    spectra : ndarray of float, shape (K, nperseg // 2 + 1)
        PSD matrix; row *k* is the PSD of symbol *k*'s indicator function.

    References
    ----------
    .. [1] Hermann, B. et al. (2024). Propofol Reversibly Attenuates
           Short-Range Microstate Ordering and 20 Hz Microstate Oscillations.
           *Brain Topography*, 37(2), 329-342.
           https://doi.org/10.1007/s10548-023-01023-1
    """
    x = np.asarray(x)
    K = len(np.unique(x))
    n_freqs = nperseg // 2 + 1
    psds = np.zeros((K, n_freqs))
    for k in range(K):
        freqs, psds[k] = welch((x == k).astype(np.float64), fs, nperseg=nperseg)
    return freqs, psds


def tsallis_entropy(p: ScalarFloatArray, q: float, base: str = '2') -> float:
    """Tsallis entropy of order *q*.

    Parameters
    ----------
    p : array_like of float, shape (K,)
        Probability distribution (must sum to 1).
    q : float
        Tsallis order (:math:`q \in \mathbb{R}`).  At ``q=1`` the formula
        reduces to the Shannon entropy.
    base : {'2', 'e'}, optional
        Logarithm base used when ``q=1`` (Shannon limit): ``'2'`` for bits
        (default), ``'e'`` for nats.  For ``q != 1`` the result is
        dimensionless and unaffected by this parameter.

    Returns
    -------
    H_q : float
        Tsallis entropy
        :math:`H_q(X) = \\frac{1}{q-1}\\left(1 - \sum_i p_i^q\\right)`.
    """
    _log = np.log2 if base == '2' else np.log
    if q == 1:
        H_q = -np.sum(p[p > 0] * _log(p[p > 0]))
    else:
        H_q = 1/(q-1)*(1 - np.sum(p**q))
    return H_q


@jit(nopython=True, cache=True)
def _lz76_impl(x: ScalarIntArray, n: int) -> float:
    """Numba fallback for lz76: returns normalized LZ complexity."""
    c = 1
    l = 1
    i = 0
    k = 1
    k_max = 1
    stop = False
    while not stop:
        if x[i+k] != x[l+k]:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l+1 > n-1:
                    stop = True
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
        else:
            k += 1
            if l+k > n-1:
                c += 1
                stop = True
    b = n / np.log2(n)
    return c / b


def lz76(x: ScalarIntArray) -> float:
    """Lempel-Ziv complexity (LZ76) of a symbolic sequence.

    Parameters
    ----------
    x : array_like of int
        Symbolic sequence of integer labels.

    Returns
    -------
    lzc : float
        Normalized LZ complexity ``c(n) / b(n)`` where ``b(n) = n / log2(n)``.
        A value near 1 indicates high (random-like) complexity; values near 0
        indicate low complexity (high regularity).

    Notes
    -----
    Uses the C extension ``_lib_lz76`` when available; falls back to a
    numba-JIT implementation otherwise.

    References
    ----------
    Lempel A, Ziv J (1976). On the Complexity of Finite Sequences.
    *IEEE Trans Inf Theory* 22(1):75-81.

    Kaspar F, Schuster HG (1987). Easily calculable measure for the
    complexity of spatiotemporal patterns. *Phys Rev A* 36(2):842-848.
    """
    x = np.asarray(x, dtype=np.int32)
    n = len(x)
    if _lib_lz76 is not None:
        buf = np.ascontiguousarray(x, dtype=np.int32)
        x_c = _ffi_lz76.cast("int *", _ffi_lz76.from_buffer(buf))
        c = _lib_lz76.lz76(x_c, n)
        return c / (n / np.log2(n))
    return _lz76_impl(x, n)


from .markov import tpm_cond  # must stay at bottom to avoid circular import


def main() -> None:
    pass


if __name__ == "__main__":
    main()

"""mstsa.markov — Markov chain models, transition matrices, and detailed balance."""

import warnings

import numpy as np
from numba import jit

from typing import Tuple
from .mstsa import ScalarFloatArray, ScalarIntArray, h1


@jit(nopython=True, cache=True)
def pmf(x: ScalarIntArray, K: int) -> ScalarFloatArray:
    """Empirical probability mass function of a symbolic sequence.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.

    Returns
    -------
    p : ndarray of float, shape (K,)
        Empirical symbol probabilities (sums to 1).

    Examples
    --------
    >>> from mstsa import pmf
    >>> pmf([0, 3, 1, 1, 2, 1, 2, 2, 0, 0, 3, 3], K=4)
    array([0.25, 0.25, 0.25, 0.25])
    """
    n = len(x)
    p = np.zeros(K)
    for i in range(n):
        p[x[i]] += 1.0
    p /= n
    return p


@jit(nopython=True, cache=True)
def tpm_cond(x: ScalarIntArray, K: int) -> ScalarFloatArray:
    """Empirical conditional transition probability matrix.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.

    Returns
    -------
    T : ndarray of float, shape (K, K)
        Row-stochastic transition matrix with
        :math:`T_{ij} = P(X_{t+1}=j \\mid X_t=i)`.

    Examples
    --------
    >>> from mstsa import tpm_cond
    >>> tpm_cond([0, 1, 0, 2, 1, 2, 0], K=3)
    """
    n = len(x)
    T = np.zeros((K, K))
    for t in range(n - 1):
        T[x[t], x[t + 1]] += 1.0
    for i in range(K):
        s = T[i].sum()
        if s > 0.0:
            T[i] /= s
    return T


@jit(nopython=True, cache=True)
def tpm_joint(x: ScalarIntArray, K: int) -> ScalarFloatArray:
    """Empirical joint transition probability matrix.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.

    Returns
    -------
    P : ndarray of float, shape (K, K)
        Joint transition matrix with :math:`P_{ij} = P(X_t=i, X_{t+1}=j)`
        (not row-normalised; rows sum to the marginal probability of state *i*).
    """
    n = len(x)
    P = np.zeros((K,K))
    for i in range(n-1):
        P[x[i], x[i+1]] += 1.0
    P /= (n-1)
    return P


@jit(nopython=True, cache=True)
def tpm_joint_exp(p: ScalarFloatArray) -> ScalarFloatArray:
    """Expected joint transition probability matrix under an independence model.

    Computes :math:`P^*_{ij} = p_i p_j / (1 - p_i)` for :math:`i \\neq j`
    and zero on the diagonal.

    Parameters
    ----------
    p : array_like of float, shape (K,)
        Empirical marginal symbol distribution.

    Returns
    -------
    P : ndarray of float, shape (K, K)
        Expected joint transition matrix under the assumption that transitions
        are independent of the current state (given no self-transitions).

    References
    ----------
    .. [1] Lehmann, D. et al. (2005). EEG microstate duration and syntax in
           attention-deficit hyperactivity disorder.
           *Psychiatry Research: Neuroimaging*, 138, 141-152.
           https://doi.org/10.1016/j.pscychresns.2004.05.007
    """
    K = len(p)
    P = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            if (i != j):
                P[i,j] = p[i]*p[j] / (1-p[i])
    return P


def p_stationary(x: "ScalarIntArray | ScalarFloatArray") -> ScalarFloatArray:
    """Stationary distribution of a transition matrix.

    Accepts either a symbolic sequence (whose transition matrix is estimated)
    or a square 2-D transition matrix passed directly as *x*.

    Parameters
    ----------
    x : array_like of int, shape (N,) or ndarray of float, shape (K, K)
        Symbolic sequence **or** row-stochastic transition matrix.

    Returns
    -------
    p_eq : ndarray of float, shape (K,)
        Stationary distribution (left eigenvector for eigenvalue 1,
        normalised to sum to 1).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        T = tpm_cond(x, len(np.unique(x)))
    elif x.ndim == 2 and x.shape[0] == x.shape[1]:
        T = x
    else:
        raise ValueError("x must be a 1-D sequence or a square 2-D transition matrix.")
    evals, evecs = np.linalg.eig(T.transpose())
    i = np.where(np.isclose(evals, 1.0, atol=1e-6))[0][0]
    p_eq = np.abs(evecs[:, i])
    p_eq /= p_eq.sum()
    return p_eq


@jit(nopython=True, cache=True)
def _detailed_balance_impl(x: ScalarIntArray, K: int) -> Tuple[float, float]:
    n = len(x)
    Tj = np.zeros((K,K)) # joint probabilities P( x(t), x(t+1) )
    for i in range(n-1):
        Tj[x[i],x[i+1]] += 1.0
    Tc = np.copy(Tj) # conditional probabilities P( x(t+1) | x(t) )
    p_row = np.sum(Tc,axis=1)
    # normalize joint probabilities
    Tj /= (n-1)
    # row-normalize conditional probabilities
    for i in range(K):
        if ( p_row[i] != 0.0 ):
            for j in range(K):
                Tc[i,j] /= p_row[i]
    t1 = 0.0
    t2 = 0.0
    for i in range(K):
        for j in range(K):
            if (i != j):
                if Tc[i,j] > 0:
                    if Tc[j,i] > 0:
                        t1 += (Tj[i,j]*np.log(Tc[i,j]/Tc[j,i])) # Roldan
                if Tj[i,j] > 0:
                    if Tj[j,i] > 0:
                        t2 += (Tj[i,j]*np.log(Tj[i,j]/Tj[j,i])) # Lynn
    return t1, t2


def detailed_balance(x: ScalarIntArray, K: int, base: str = '2') -> Tuple[float, float]:
    """Compute detailed-balance statistics for a symbolic sequence.

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
    t1 : float
        Roldan-Parrondo statistic:
        :math:`\\sum_{x,y} p(x,y) \\log[p(x|y)/p(y|x)]`.
    t2 : float
        Lynn statistic:
        :math:`\\sum_{x,y} p(x,y) \\log[p(x,y)/p(y,x)]`.

    References
    ----------
    .. [1] Roldan, E. & Parrondo, J. M. R. (2010). Estimating dissipation
           from single stationary trajectories. *Physical Review Letters*,
           105, 150607.
    .. [2] Lynn, C. W. et al. (2021). Broken detailed balance and entropy
           production in the human brain. *PNAS*, 118, e2109889118.
    """
    t1, t2 = _detailed_balance_impl(x, K)
    if base == '2':
        t1 /= np.log(2)
        t2 /= np.log(2)
    return t1, t2


def detailed_balance_cond(T: ScalarFloatArray, base: str = '2') -> float:
    """Detailed-balance statistic from a conditional transition matrix.

    Parameters
    ----------
    T : ndarray of float, shape (K, K)
        Joint transition matrix with :math:`T_{ij} = P(X_t=i, X_{t+1}=j)`.
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    t : float
        Roldan-Parrondo statistic:
        :math:`\\sum_{x \\neq y} T_{xy} \\log[p(x|y)/p(y|x)]`.

    References
    ----------
    .. [1] Roldan, E. & Parrondo, J. M. R. (2010). Estimating dissipation
           from single stationary trajectories. *Physical Review Letters*,
           105, 150607.
    """
    _log = np.log2 if base == '2' else np.log
    K = T.shape[0] # NOTE: square matrix property not asserted
    # convert joint to conditional matrix
    Tc = np.copy(T)
    p_row = np.sum(Tc, axis=1)
    # row-normalize conditional probabilities
    for i in range(K):
        if ( p_row[i] != 0.0 ):
            for j in range(K):
                Tc[i,j] /= p_row[i] # row sums --> 1.0
    t = 0
    for i in range(K):
        for j in range(K):
            if (i != j):
                if Tc[i,j] > 0:
                    if Tc[j,i] > 0:
                        t += (T[i,j]*_log(Tc[i,j]/Tc[j,i]))
    return t


def detailed_balance_joint(T: ScalarFloatArray, base: str = '2') -> float:
    """Detailed-balance statistic from a joint transition matrix.

    Parameters
    ----------
    T : ndarray of float, shape (K, K)
        Joint transition matrix with :math:`T_{ij} = P(X_t=i, X_{t+1}=j)`.
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    t : float
        Lynn statistic:
        :math:`\\sum_{x \\neq y} T_{xy} \\log[T_{xy}/T_{yx}]`.

    References
    ----------
    .. [1] Lynn, C. W. et al. (2021). Broken detailed balance and entropy
           production in the human brain. *PNAS*, 118, e2109889118.
    """
    _log = np.log2 if base == '2' else np.log
    K = T.shape[0] # NOTE: square matrix property not asserted
    t = 0
    for i in range(K):
        for j in range(K):
            if (i != j):
                if T[i,j] > 0:
                    if T[j,i] > 0:
                        t += (T[i,j]*_log(T[i,j]/T[j,i]))
    return t


def entropy_rate_mc(x: "ScalarIntArray | ScalarFloatArray", base: str = '2') -> float:
    """Theoretical entropy rate under a first-order Markov assumption.

    Accepts either a symbolic sequence *x* (from which the transition matrix
    is estimated) or a square 2-D transition matrix passed as *x*.

    Parameters
    ----------
    x : array_like of int, shape (N,) or ndarray of float, shape (K, K)
        Symbolic sequence **or** transition matrix.  When 1-D, the transition
        matrix is estimated from the sequence.  When 2-D and square, it is
        treated directly as the row-stochastic transition matrix.
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    h : float
        Theoretical entropy rate
        :math:`h = -\\sum_i \\sum_j \\pi_i T_{ij} \\log T_{ij}`.

    Notes
    -----
    The stationary distribution :math:`\\pi` is computed from the left
    eigenvector of *T* corresponding to eigenvalue 1.

    References
    ----------
    .. [1] von Wegner, F. et al. (2026). A Quantitative Comparison of Two
           Methods for Higher-Order EEG Microstate Syntax Analysis.
           *Brain Topography*, 39, 45.
           https://doi.org/10.1007/s10548-026-01196-5
    """
    x = np.asarray(x)
    if x.ndim == 1:
        T = tpm_cond(x, len(np.unique(x)))
    elif x.ndim == 2 and x.shape[0] == x.shape[1]:
        T = x.copy()
    else:
        raise ValueError("x must be a 1-D sequence or a square 2-D transition matrix.")
    _log = np.log2 if base == '2' else np.log
    p_row = T.sum(axis=1, keepdims=True)
    p_row[p_row == 0] = 1.
    T /= p_row
    p_eq = p_stationary(T)
    h = 0.0
    for i, j in np.ndindex(T.shape):
        if T[i, j] > 0:
            h -= p_eq[i] * T[i, j] * _log(T[i, j])
    return h


def entropy_rates_mc(x: ScalarIntArray,
                         K: int,
                         base: str = '2') -> "tuple[float, float, float, float]":
    """Theoretical entropy rates under zero- and first-order Markov models.

    Computes expected entropy rates for the full sequence and the embedded
    jump process under both Markov-0 (i.i.d.) and Markov-1 assumptions.

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
    er_full_mc0 : float
        Markov-0 entropy rate of the full sequence.
    er_jump_mc0 : float
        Markov-0 entropy rate of the jump (embedded) sequence.
    er_full_mc1 : float
        Markov-1 entropy rate of the full sequence.
    er_jump_mc1 : float
        Markov-1 entropy rate of the jump (embedded) sequence.

    References
    ----------
    .. [1] von Wegner, F. et al. (2026). A Quantitative Comparison of Two
           Methods for Higher-Order EEG Microstate Syntax Analysis.
           *Brain Topography*, 39, 45.
           https://doi.org/10.1007/s10548-026-01196-5
    """
    # Markov-1, full
    T1_x = tpm_cond(x, K)
    # stationary distribution
    evals, evecs = np.linalg.eig(T1_x.transpose())
    i = np.where(np.isclose(evals, 1.0, atol=1e-6))[0][0]
    px = np.abs(evecs[:,i]) # make non-negative
    px /= px.sum()
    #print("\n", px, "\n", np.dot(px,T1_x)) # check stationarity

    # Markov-1, jump
    T1_y = T1_x - np.diag(np.diag(T1_x))
    rowsum = T1_y.sum(axis=1, keepdims=True)
    rowsum[rowsum==0] = 1
    T1_y /= rowsum
    # stationary distribution
    evals, evecs = np.linalg.eig(T1_y.transpose())
    i = np.where(np.isclose(evals, 1.0, atol=1e-6))[0][0]
    py = np.abs(evecs[:,i]) # make non-negative
    py /= py.sum()
    #print("\n", py, "\n", np.dot(py,T1_y)) # check stationarity

    # Markov-0, full
    T0_x = np.tile(px, (K,1))
    #print("\n", px, "\n", np.dot(px,T0_x)) # check stationarity

    # Markov-0, jump
    T0_y = T0_x - np.diag(np.diag(T0_x))
    rowsum = T0_y.sum(axis=1, keepdims=True)
    rowsum[rowsum==0] = 1
    T0_y /= rowsum
    #print("\n", py, "\n", np.dot(py,T0_y)) # check stationarity

    _log = np.log2 if base == '2' else np.log
    er_full_mc0 = 0
    er_jump_mc0 = 0
    er_full_mc1 = 0
    er_jump_mc1 = 0
    for i in range(K):
        for j in range(K):
            er_full_mc0 -= ( px[i]*T0_x[i,j]*_log(T0_x[i,j]) )
            if T0_y[i,j] > 0:
                er_jump_mc0 -= ( py[i]*T0_y[i,j]*_log(T0_y[i,j]) )
            er_full_mc1 -= ( px[i]*T1_x[i,j]*_log(T1_x[i,j]) )
            if T1_y[i,j] > 0:
                er_jump_mc1 -= ( py[i]*T1_y[i,j]*_log(T1_y[i,j]) )

    return er_full_mc0, er_jump_mc0, er_full_mc1, er_jump_mc1


def mc_sample_path(x: ScalarIntArray = None,
       T: ScalarFloatArray = None,
       p: ScalarFloatArray = None,
       n: int = 0) -> ScalarIntArray:
    """Generate a first-order Markov chain surrogate sequence.

    Supply either an observed sequence *x* (whose transition matrix and
    stationary distribution are estimated automatically) **or** an explicit
    transition matrix *T* with an optional starting distribution *p*.

    Parameters
    ----------
    x : array_like of int, optional
        Observed symbolic sequence used to estimate *T* and *p*.
    T : ndarray of float, shape (K, K), optional
        Row-stochastic transition matrix with
        :math:`T_{ij} = P(X_{t+1}=j \\mid X_t=i)`.
    p : array_like of float, shape (K,), optional
        Initial symbol distribution.  Defaults to the stationary
        distribution of *T*.
    n : int
        Length of the generated sequence.

    Returns
    -------
    y : ndarray of int, shape (n,)
        Simulated Markov chain.
    """
    if x is not None and T is not None:
        raise ValueError("Provide either x or T, not both.")
    if x is not None:
        K = len(np.unique(x))
        T = tpm_cond(x, K)
    if T is not None:
        assert T.ndim == 2
        assert T.shape[0] == T.shape[1]
    if p is None:
        p = p_stationary(T)
    return _mc_numba(T=T, p=p, n=n)


@jit(nopython=True, cache=True)
def _mc_numba(T: ScalarFloatArray, p: ScalarFloatArray, n: int) -> ScalarIntArray:
    """Numba-JIT core of :func:`mc_sample_path`."""
    r = np.random.rand()
    s = 0
    u = p[s]
    y = np.zeros(n, dtype='i')
    while u < r:
        s += 1
        u += p[s]
    y[0] = s
    for i in range(1, n):
        r = np.random.rand()
        s = y[i-1]
        t = 0
        u = T[s, t]
        while u < r:
            t += 1
            u += T[s, t]
        y[i] = t
    return y


def aif_mc(x: ScalarIntArray, lmax: int, base: str = '2') -> ScalarFloatArray:
    """Theoretical auto-information function (AIF) of a first-order Markov chain.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Discrete symbolic sequence.
    lmax : int
        Maximum time lag (samples).
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    mi : ndarray of float, shape (lmax,)
        AIF coefficients as predicted from a first-order Markov model.
    """
    _log = np.log2 if base == '2' else np.log
    K = len(set(x)) # number of symbols
    mi = np.zeros(lmax)
    h_1 = h1(x, K, base=base) # entropy
    mi[0] = h_1
    p = pmf(x, K)
    T = tpm_cond(x, K)
    #eps = 1e-12
    for l in range(1,lmax):
        T_l = np.linalg.matrix_power(T,l)
        h_12 = 0.0
        for i in range(K):
            s = 0.0 # conditional entropy H( X_{k+l} | X(X_{k}) )
            for j in range(K):
                if (T_l[i,j] > 0.0):
                    s += ( T_l[i,j] * _log(T_l[i,j]) )
            h_12 += (p[i]*s)
        # I( X_{k} , X_{k+l} ) = H( X_{k+l } - H( X_{k+l} | X_{k} )
        mi[l] = h_1 + h_12
        #if ( mi[l] < 0 ): mi[l] = eps

    return mi


def dur_occ_cov_mc(x: ScalarIntArray, fs: float, steady_state: bool = False) -> Tuple[ScalarFloatArray, ScalarFloatArray, ScalarFloatArray]:
    """Theoretical microstate duration, occurrence, and coverage under a Markov model.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    fs : float
        Sampling rate in Hz.
    steady_state : bool, optional
        If True, coverage is the stationary distribution of the estimated
        transition matrix; if False, the empirical symbol probabilities are
        used (default False).

    Returns
    -------
    dur : ndarray of float, shape (K,)
        Theoretical mean duration per symbol in milliseconds,
        :math:`1/(1 - T_{kk}) \\cdot \\Delta t`.
    occ : ndarray of float, shape (K,)
        Theoretical occurrence rate per symbol in Hz,
        :math:`\\pi_k (1 - T_{kk}) \\cdot f_s`.
    cov : ndarray of float, shape (K,)
        Coverage (fraction of total time) per symbol.
    """
    dt = 1000/fs # sampling interval in msec
    K = len(set(x))
    p = pmf(x,K)
    T = tpm_cond(x,K)

    # (1) Theoretical DURATION (msec)
    dur = 1/(1-np.diag(T)) * dt

    # (2) Theoretical OCCURRENCE (1/sec, Hz)
    occ = np.array([p[k]*(1-T[k,k]) for k in range(K)]) * fs
    # same as:
    # occ = np.array([np.dot(p, T[:,k]) for k in range(K)]) * fs

    # (3) Theoretical COVERAGE (%/100)
    if steady_state:
        evals, evecs = np.linalg.eig(T.transpose())
        i = np.where(np.isclose(evals, 1.0, atol=1e-6))[0][0]
        p_eq = np.abs(evecs[:,i]) # make non-negative
        p_eq /= p_eq.sum()
        cov = p_eq
    else:
        cov = p

    return dur, occ, cov


def generator_matrix(x: ScalarIntArray = None,
                 T: ScalarFloatArray = None,
                 fs: float = 1) -> ScalarFloatArray:
    """Continuous-time Markov chain (CTMC) generator matrix.

    Computes the generator matrix *Q* from either an observed discrete
    sequence *x* or an explicit transition matrix *T*.
    Use to predict evolution of p: p(t) = expm(Q*t)

    Parameters
    ----------
    x : array_like of int, optional
        Observed symbolic sequence used to estimate *T*.
    T : ndarray of float, shape (K, K), optional
        Row-stochastic transition matrix.
    fs : float, optional
        Sampling rate in Hz, used to convert rates to physical units
        (default 1 — rates in samples\ :sup:`-1`).

    Returns
    -------
    Q : ndarray of float, shape (K, K)
        CTMC generator matrix satisfying :math:`T_t = e^{Qt}`.
    """
    if x is not None and T is not None:
        raise ValueError("Provide either x or T, not both.")
    if x is not None:
        K = len(np.unique(x))
        T = tpm_cond(x, K)
    if T is not None:
        assert T.ndim == 2
        assert T.shape[0] == T.shape[1]
        K = T.shape[0]

    #p = p_stationary(T)
    T_diag = np.diag(T) # diagonal elements of Tx: P(X_{t+1}=i | X_{t}=i)
    # tau: mean lifetime (in samples) = expected value of geometric distribution
    tau = 1/(1-T_diag)
    # q_i = 1/tau_i = -Q_ii; assure diagonal yields mean lifetime
    q = 1/tau

    # jump process transition matrix (conditional probabilities)
    T_jump = T - np.diag(T_diag) # no self-transitions
    T_jump /= T_jump.sum(axis=1, keepdims=True) # re-scale rows to sum=1

    # CTMC generator matrix
    Q = np.diag(-q) + q[:,None]*T_jump
    Q = Q*fs # scales to 1/s (Hz)
    return Q


def relaxation_time(x: ScalarIntArray, K: int = None) -> float:
    """Relaxation time (inverse spectral gap) of a symbolic sequence.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int, optional
        Number of distinct symbols.  Inferred from *x* if not provided.

    Returns
    -------
    t_rlx : float
        Relaxation time :math:`1 / (\\lambda_0 - \\lambda_1)` where
        :math:`\\lambda_0 \\ge \\lambda_1` are the two largest eigenvalues
        of the transition matrix.
    """
    T = tpm_cond(x, K)
    ev = np.sort(np.real(np.linalg.eigvals(T)))  # ascending; ev[-1] == 1
    return 1.0 / (ev[-1] - ev[-2])  # 1 / spectral gap


def p_joint_mc(m: int, K: int, p: ScalarFloatArray,
               T: ScalarFloatArray) -> ScalarFloatArray:
    """Joint probability distribution of an *m*-gram under a Markov model.

    Computes :math:`P(x_0, \\ldots, x_{m-1}) = p(x_0) \\prod_{i=0}^{m-2} T(x_i, x_{i+1})`.

    Parameters
    ----------
    m : int
        Length of the joint block.
    K : int
        Number of distinct symbols.
    p : array_like of float, shape (K,)
        Marginal symbol distribution.
    T : ndarray of float, shape (K, K)
        Row-stochastic transition matrix.

    Returns
    -------
    p_joint : ndarray of float, shape (K,) * m
        *m*-dimensional joint probability array. For ``m=0`` this is the
        0-dimensional array ``array(1.0)`` — the probability of the (unique)
        empty block.

    Notes
    -----
    Each entry ``p_joint[i, j, ...]`` equals
    ``p_ngram_mc(np.array([i, j, ...]), p, T)``.
    """
    if m == 0:
        return np.array(1.0)
    shape = tuple(m*[K])
    p_joint = np.zeros(shape)
    for idx in np.ndindex(shape):
        p_joint[idx] = p[idx[0]]
        for i in range(m-1):
            p_joint[idx] *= T[idx[i], idx[i+1]]
    return p_joint


@jit(nopython=True, cache=True)
def p_ngram_mc(x: ScalarIntArray, p: ScalarFloatArray,
                    T: ScalarFloatArray) -> float:
    """Probability of a trajectory under a first-order Markov model.

    Computes :math:`P(x_0, \\ldots, x_{m-1}) = p(x_0) \\prod_{i} T(x_i, x_{i+1})`.

    Parameters
    ----------
    x : array_like of int, shape (m,)
        Observed state trajectory.
    p : array_like of float, shape (K,)
        Marginal symbol distribution.
    T : ndarray of float, shape (K, K)
        Row-stochastic transition matrix.

    Returns
    -------
    p_traj : float
        Trajectory probability.
    """
    # p[x[0]] * np.prod([T[i,j] for i,j in zip(x[:-1], x[1:])])
    p_traj = p[x[0]]
    for i in range(len(x) - 1):
        p_traj *= T[x[i], x[i+1]]
    return p_traj


def sample_entropy_mc(m: int, n: int, K: int, p: ScalarFloatArray,
                          T: ScalarFloatArray,
                          base: str = '2') -> "tuple[float, float, float]":
    """Expected sample entropy under a first-order Markov model.

    Parameters
    ----------
    m : int
        History length (>= 0), matching the empirical
        ``sample_entropy_disc``/``sample_entropy_cont`` convention. At
        ``m=0`` (no conditioning), the returned value is the theoretical
        order-2 Renyi collision entropy of the stationary distribution
        ``p``, i.e. :math:`-\\log \\sum_k p_k^2`.
    n : int
        Sequence length.
    K : int
        Number of distinct symbols.
    p : array_like of float, shape (K,)
        Marginal symbol distribution.
    T : ndarray of float, shape (K, K)
        Row-stochastic transition matrix.
    base : {'2', 'e'}, optional
        Logarithm base: ``'2'`` for bits (default), ``'e'`` for nats.

    Returns
    -------
    se : float
        Expected sample entropy :math:`-\\log(A_m / B_m)`.
    A_m : float
        Expected count of *(m+1)*-block matches.
    B_m : float
        Expected count of *m*-block matches.

    Raises
    ------
    ValueError
        If ``m < 0``.

    References
    ----------
    .. [1] von Wegner, F. et al. (2026). A Quantitative Comparison of Two
           Methods for Higher-Order EEG Microstate Syntax Analysis.
           *Brain Topography*, 39, 45.
           https://doi.org/10.1007/s10548-026-01196-5
    """
    if m < 0:
        raise ValueError("sample_entropy_mc: m must be >= 0.")
    _log = np.log2 if base == '2' else np.log
    N_m = (n-m)*(n-m-1)/2 # number of comparisons for given m, n
    B_m = N_m*np.sum( (p_joint_mc(m  , K, p, T))**2 )
    A_m = N_m*np.sum( (p_joint_mc(m+1, K, p, T))**2 )
    se = -_log(A_m/B_m)
    return se, A_m, B_m


def sample_entropies_mc(x: ScalarIntArray,
                           K: int,
                           base: str = '2') -> "tuple[float, float, float, float]":
    """Expected sample entropy under zero- and first-order Markov models.

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
    se_full_mc0 : float
        Markov-0 expected sample entropy of the full sequence.
    se_jump_mc0 : float
        Markov-0 expected sample entropy of the jump sequence.
    se_full_mc1 : float
        Markov-1 expected sample entropy of the full sequence.
    se_jump_mc1 : float
        Markov-1 expected sample entropy of the jump sequence.

    References
    ----------
    .. [1] von Wegner, F. et al. (2026). A Quantitative Comparison of Two
           Methods for Higher-Order EEG Microstate Syntax Analysis.
           *Brain Topography*, 39, 45.
           https://doi.org/10.1007/s10548-026-01196-5
    """
    _log = np.log2 if base == '2' else np.log
    # Markov-1, full
    T1_x = tpm_cond(x, K)
    # stationary distribution
    evals, evecs = np.linalg.eig(T1_x.transpose())
    i = np.where(np.isclose(evals, 1.0, atol=1e-6))[0][0]
    px = np.abs(evecs[:,i]) # make non-negative
    px /= px.sum()
    #print("\n", px, "\n", np.dot(px,T1_x)) # check stationarity --> correct

    # Markov-1, jump
    T1_y = T1_x - np.diag(np.diag(T1_x))
    rowsum = T1_y.sum(axis=1, keepdims=True)
    rowsum[rowsum==0] = 1
    T1_y /= rowsum
    # stationary distribution
    evals, evecs = np.linalg.eig(T1_y.transpose())
    i = np.where(np.isclose(evals, 1.0, atol=1e-6))[0][0]
    py = np.abs(evecs[:,i]) # make non-negative
    py /= py.sum()
    #print("\n", py, "\n", np.dot(py,T1_y)) # check stationarity --> correct

    # Markov-0, full
    T0_x = np.tile(px, (K,1))
    #print("\n", px, "\n", np.dot(px,T0_x)) # check stationarity --> correct

    # Markov-0, jump
    T0_y = T0_x - np.diag(np.diag(T0_x))
    rowsum = T0_y.sum(axis=1, keepdims=True)
    rowsum[rowsum==0] = 1
    T0_y /= rowsum
    #print("\n", py, "\n", np.dot(py,T0_y)) # check stationarity --> correct

    sum_full_mc0 = 0
    sum_jump_mc0 = 0
    sum_full_mc1 = 0
    sum_jump_mc1 = 0
    for i in range(K):
        for j in range(K):
            sum_full_mc0 += ( px[i]*T0_x[i,j]**2 )
            sum_jump_mc0 += ( py[i]*T0_y[i,j]**2 )
            sum_full_mc1 += ( px[i]*T1_x[i,j]**2 )
            sum_jump_mc1 += ( py[i]*T1_y[i,j]**2 )
    se_full_mc0 = -_log(sum_full_mc0)
    se_jump_mc0 = -_log(sum_jump_mc0)
    se_full_mc1 = -_log(sum_full_mc1)
    se_jump_mc1 = -_log(sum_jump_mc1)

    return se_full_mc0, se_jump_mc0, se_full_mc1, se_jump_mc1


def nearest_reversible_mc(A: ScalarFloatArray, pi: ScalarFloatArray, weighted: bool = False) -> ScalarFloatArray:
    """Compute the nearest reversible Markov chain transition matrix.

    Solves a convex quadratic programme to find the row-stochastic matrix
    closest to *A* that satisfies detailed balance with respect to *pi*.
    Uses ``scipy.optimize.minimize`` with the SLSQP method.

    Parameters
    ----------
    A : ndarray of float, shape (n, n)
        Input (possibly non-reversible) transition matrix.
    pi : array_like of float, shape (n,)
        Target stationary distribution.
    weighted : bool, optional
        Use the *pi*-weighted distance metric (default False, unweighted).

    Returns
    -------
    U : ndarray of float, shape (n, n)
        Nearest reversible transition matrix.

    References
    ----------
    .. [1] Nielsen, A. J. N. & Weber, M. (2015). Computing the nearest
           reversible Markov chain. *Numerical Linear Algebra with
           Applications*. https://doi.org/10.1002/nla.1967
    """
    from scipy.optimize import minimize

    # Check if input is valid
    a, n = A.shape
    if a != n:
        raise ValueError("A must be a square matrix.")
    if n != len(pi):
        raise ValueError("Stationary distribution pi has wrong size.")
    if not np.all(pi) and weighted:
        raise ValueError("pi has zero entries, which is not allowed for the weighted scheme.")

    # compute number of basis vectors (see Proposition 2.1)
    tempB = np.sum(pi==0)
    m = int( (n-1)*n/2 +1 + (tempB-1)*tempB/2 )

    # myBasis is an array of m matricies.
    # It contains the basis vectors of the supspace U (see Proposition 2.1)
    myBasis = [None for _ in range(m)]
    index = 0 # Python
    for r in range(n-1):
        for s in range(r+1,n):
            if ((pi[s]==0) and (pi[r]==0)):
                B = np.eye(n)
                B[r,r] = 0
                B[r,s] = 1
                myBasis[index] = B
                index += 1
                B[r,r] = 1
                B[r,s] = 0
                B[s,s] = 0
                B[s,r] = 1
                myBasis[index] = B
            else:
                B = np.eye(n)
                B[r,s] = pi[s]
                B[s,r] = pi[r]
                B[r,r] = 1 - pi[s]
                B[s,s] = 1 - pi[r]
                myBasis[index] = B
            index += 1
    # the last basis vector is the identity matrix
    myBasis[index] = np.eye(n)

    # Compute D and D^(-1) if weighted scheme
    if weighted:
        D = np.diag(pi)
        Di = np.linalg.inv(D)

    f = np.zeros(m) # column vector zeros(m,1) in Matlab
    Q = np.zeros((m,m))

    if not weighted:
        # Step 1 from 3: Compute f from convex optimization problem
        for i in range(m):
            B = myBasis[i]
            f[i] = -2 *np.trace(B.T @ A)

        # Step 2 from 3: Compute Q from convex optimization problem
        for ii in range(m):
            B = myBasis[ii]
            for jj in range(m):
                H = myBasis[jj]
                t = 2*np.trace(B.T @ H)
                Q[ii,jj] = t
                Q[jj,ii] = t
    else:
        # case: weighted
        # Step 1 from 3: Compute f from convex optimization problem
        for i in range(m):
            B = myBasis[i]
            f[i] = -2*np.trace(D @ B @ Di @ A.T)

        # Step 2 from 3: Compute Q from convex optimization problem
        for ii in range(m):
            B = myBasis[ii]
            Z = D @ B @ Di
            for jj in range(m):
                H = myBasis[jj]
                t = 2*np.trace(H.T @ Z)
                Q[ii,jj] = t
                Q[jj,ii] = t

    # Step 3 from 3: Compute C from convex optimization problem
    C = -np.eye(m-1+n, m)
    C[m-1,m-1] = 0

    # We only need to compute rows from m to n+m-1 (n rows).
    # Each row is equal to -g_i(j) (see page 5 from article).
    for i in range(n):
        index = 0 #index=1;
        # iterate through basis v_j , j=1,...,m
        # j is stored as alias index.
        for r in range(n-1):
            for s in range(r+1,n):
                if ((pi[s]==0) and (pi[r]==0)):
                    # iterate through 2 basis vectors delta^[r,s] and delta^[s,r]
                    if (r != i):
                        C[m-1+i,index] = -1  # Case: else
                    else:
                        C[m-1+i,index] = 0  # Case: v_j = delta^[i,s]
                    index = index + 1
                    if (s != i):
                        C[m-1+i,index] = -1  # Case: else
                    else:
                        C[m-1+i,index] = 0  # Case: v_j = delta^[i,s]
                elif (s==i):
                    C[m-1+i,index] = -1 + pi[r]  # Case: v_j = A^[r,i]
                elif (r==i):
                    C[m-1+i,index] = -1 + pi[s]  # Case: v_j = A^[i,s]
                else:
                    # Case: else, this is (r != i) and (s != i)
                    C[m-1+i,index] = -1
                index = index + 1
        # v_m = Id, thus always g_i(m)=1
        C[m-1+i,m-1] = -1

    n_var = Q.shape[1]
    x0 = np.ones(n_var) / n_var
    constraints = [
        {'type': 'eq',   'fun': lambda x: x.sum() - 1,
                         'jac': lambda _: np.ones(n_var)},
        {'type': 'ineq', 'fun': lambda x: -(C @ x),
                         'jac': lambda _: -C},
    ]
    sol = minimize(lambda x: 0.5 * x @ Q @ x + f @ x, x0,
                   jac=lambda x: Q @ x + f,
                   method='SLSQP', constraints=constraints)
    sol_x = sol.x

    # Compute U
    U = np.zeros((n,n))
    for i in range(m):
        U = U + sol_x[i]*myBasis[i]

    return U


def max_entropy_T(p: ScalarFloatArray, n_iter: int = 10_000,
                  tol: float = 1e-14) -> Tuple[ScalarFloatArray, dict]:
    """Maximum-entropy-rate zero-diagonal transition matrix for a target
    stationary distribution.

    Computes the row-stochastic matrix :math:`T^*` with zero diagonal (no
    self-transitions) whose entropy rate is maximal among all zero-diagonal
    chains with stationary distribution exactly *p*. There is no closed
    form: :math:`T^*` corresponds to the maximum-entropy coupling
    :math:`\\pi_{ij} = p_i T_{ij}` of *p* with itself, supported off the
    diagonal, found via iterative proportional fitting (Sinkhorn scaling) —
    starting from any positive matrix on that support and alternately
    rescaling rows and columns to match *p* converges geometrically to the
    unique product-form fixed point :math:`\\pi_{ij} = c_i c_j`, for any *p*
    with :math:`\\max_i p_i \\le 1/2` (the necessary and sufficient
    feasibility condition).

    Useful as a maximum-entropy null model for microstate transition syntax:
    unlike the randomization test null hypothesis (Lehmann et al., 2005), which 
    normalizes each row of *p* without reproducing *p* as its own stationary 
    distribution), 
    :math:`T^*` reproduces the empirical marginal exactly while otherwise 
    maximizing randomness of the transition structure.

    Parameters
    ----------
    p : array_like of float, shape (K,)
        Target stationary distribution (renormalized internally if it
        doesn't already sum to 1). Requires ``K >= 3`` and
        ``max(p) <= 0.5``.
    n_iter : int, optional
        Maximum number of Sinkhorn iterations (default 10000).
    tol : float, optional
        Convergence tolerance on the maximum row/column marginal error
        (default 1e-14).

    Returns
    -------
    T : ndarray of float, shape (K, K)
        Maximum-entropy zero-diagonal row-stochastic transition matrix with
        stationary distribution *p*.
    info : dict
        ``{'n_iter': int, 'marginal_error': float}`` — iterations used and
        the final maximum marginal error.

    Raises
    ------
    ValueError
        If ``K < 3`` (a zero-diagonal 2-state chain is forced to the swap
        matrix, whose stationary distribution is always uniform), or if
        ``max(p) > 0.5`` (infeasible: inflow into the dominant state is
        capped at ``1 - max(p)``).

    Notes
    -----
    Convergence is geometric away from the feasibility boundary
    :math:`\\max_i p_i = 1/2`, but degrades to sublinear right at that
    boundary (the coupling polytope collapses to a single point there); pass
    a larger *n_iter* if *p* is close to that edge and
    ``info['marginal_error']`` is not small enough. Emits a ``UserWarning``
    if *tol* is not reached within *n_iter* iterations.
    """
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    K = len(p)

    if K < 3:
        raise ValueError("K>=3 required: a zero-diagonal K=2 chain has no "
                         "off-diagonal freedom (T is forced to the swap "
                         "matrix, whose stationary distribution is always "
                         "uniform)")
    if p.max() > 0.5 + 1e-12:
        raise ValueError(f"infeasible: max(p)={p.max():.6f} > 1/2 -- no "
                         f"zero-diagonal chain can have this p as its "
                         f"stationary distribution (inflow into the "
                         f"dominant state is capped at 1-max(p))")

    off_diag = ~np.eye(K, dtype=bool)
    pi = off_diag.astype(float)

    n_used = n_iter
    err = np.inf
    for it in range(n_iter):
        pi *= (p / pi.sum(axis=1))[:, None]
        pi *= (p / pi.sum(axis=0))[None, :]
        err = max(np.abs(pi.sum(axis=1) - p).max(),
                 np.abs(pi.sum(axis=0) - p).max())
        if err < tol:
            n_used = it + 1
            break

    if err >= tol:
        warnings.warn(f"max_entropy_T: did not reach tol={tol:.1e} within "
                      f"{n_iter} iterations (marginal error={err:.2e}); "
                      f"p is likely close to the max_i p_i=1/2 feasibility "
                      f"boundary where convergence is sublinear -- increase "
                      f"n_iter")

    T = pi / p[:, None]
    np.fill_diagonal(T, 0.0)
    return T, {"n_iter": n_used, "marginal_error": err}

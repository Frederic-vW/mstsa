"""mstsa.stats — statistical tests for symbolic sequences.

Kullback-Technometrics Markov-chain tests, geometric-distribution tests,
homogeneity tests, and the multi-subject transition-matrix syntax test.
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.stats import chi2

from typing import Generic, Tuple, TypeVar

from .mstsa import (
    ScalarFloatArray,
    ScalarIntArray,
    embedded_process,
    sojourn_time_histograms,
    sojourn_times_unordered,
)

from .markov import (
    mc_sample_path,
    pmf,
    tpm_cond,
    tpm_joint,
    tpm_joint_exp,
)

@jit(nopython=True, cache=True)
def _count_transitions(x: ScalarIntArray, n: int, K: int) -> ScalarFloatArray:
    """Count joint transition frequencies f[i,j] = #{t: x[t]=i, x[t+1]=j}."""
    f = np.zeros((K, K))
    for t in range(n - 1):
        f[x[t], x[t + 1]] += 1.0
    return f


@jit(nopython=True, cache=True)
def _count_markov1(x: ScalarIntArray, n: int, K: int) -> ScalarFloatArray:
    """Count 3-gram frequencies f[i,j,k]."""
    f = np.zeros((K, K, K))
    for t in range(n - 2):
        f[x[t], x[t + 1], x[t + 2]] += 1.0
    return f


@jit(nopython=True, cache=True)
def _count_markov2(x: ScalarIntArray, n: int, K: int) -> ScalarFloatArray:
    """Count 4-gram frequencies f[i,j,k,l]."""
    f = np.zeros((K, K, K, K))
    for t in range(n - 3):
        f[x[t], x[t + 1], x[t + 2], x[t + 3]] += 1.0
    return f


@jit(nopython=True, cache=True)
def _count_block_symbols(x: ScalarIntArray, K: int, r: int, l: int) -> ScalarFloatArray:
    """Count per-block symbol frequencies f[block, symbol]."""
    f = np.zeros((r, K))
    for i in range(r):
        for k in range(l):
            f[i, x[i * l + k]] += 1.0
    return f


@jit(nopython=True, cache=True)
def _count_block_transitions(x: ScalarIntArray, K: int, r: int, l: int) -> ScalarFloatArray:
    """Count per-block transition frequencies f[block, from, to]."""
    f = np.zeros((r, K, K))
    for i in range(r):
        for ii in range(l - 1):
            f[i, x[i * l + ii], x[i * l + ii + 1]] += 1.0
    return f


@jit(nopython=True, cache=True)
def _permutation_null(D: ScalarFloatArray, p_sum: ScalarFloatArray, q_sum: ScalarFloatArray,
                      n_files: int, odi_0: ScalarIntArray, odi_1: ScalarIntArray,
                      n_perm: int) -> ScalarFloatArray:
    """Permutation null distribution for the transition syntax test.

    For each permutation, each subject's observed/expected labels are
    independently swapped with probability 0.5 (Bernoulli), then the
    chi-square distance between group means is recomputed.  Uses the identity

        p_mean_perm = (p_sum - D[S].sum(0)) / n_files
        q_mean_perm = (q_sum + D[S].sum(0)) / n_files

    where D = p_arr - q_arr and S is the random swap set, avoiding any
    per-iteration array copies.
    """
    n_odi = len(odi_0)
    K     = p_sum.shape[0]
    t0    = np.zeros(n_perm)
    D_S   = np.zeros((K, K))
    n     = float(n_files)
    for perm in range(n_perm):
        for i in range(K):
            for j in range(K):
                D_S[i, j] = 0.0
        for s in range(n_files):
            if np.random.random() < 0.5:
                for i in range(K):
                    for j in range(K):
                        D_S[i, j] += D[s, i, j]
        stat = 0.0
        for od in range(n_odi):
            ii = odi_0[od]; jj = odi_1[od]
            p_m = (p_sum[ii, jj] - D_S[ii, jj]) / n
            q_m = (q_sum[ii, jj] + D_S[ii, jj]) / n
            stat += (p_m - q_m) ** 2 / q_m
        t0[perm] = stat
    return t0


def multiple_comparisons(p_values: ScalarFloatArray, method: str) -> ScalarFloatArray:
    """Apply a multiple-comparisons correction to an array of p-values.

    Parameters
    ----------
    p_values : array_like of float
        Uncorrected p-values.
    method : str
        Correction method.  Supported values:

        ``'bonferroni'``
            Bonferroni correction: multiply each p-value by the number of
            tests and clip to 1.
        ``'fdr_bh'``
            Benjamini-Hochberg false discovery rate correction.

    Returns
    -------
    pvals_corrected : ndarray of float
        Corrected p-values (same order as input).

    Raises
    ------
    ValueError
        If *method* is not one of the supported strings.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size

    if method == 'bonferroni':
        return np.clip(p * m, 0.0, 1.0)

    if method == 'fdr_bh':
        order = np.argsort(p)
        p_sorted = p[order]
        # p_adjusted[k] = p_sorted[k] * m / (k+1)
        ranks = np.arange(1, m + 1)
        adjusted = p_sorted * m / ranks
        # enforce monotonicity: working from right, each value is the min
        # of itself and all values to its right
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        # restore original order
        result = np.empty(m)
        result[order] = adjusted
        return result

    raise ValueError(
        f"Unknown method {method!r}. Supported: 'bonferroni', 'fdr_bh'."
    )


def test_cond_homogeneity(x: ScalarIntArray, K: int, l: int, alpha: float,
                          verbose: bool = True) -> float:
    """Test conditional homogeneity (stationarity of transitions) of a sequence.

    Splits *x* into non-overlapping blocks of length *l* and tests whether
    the transition probabilities are homogeneous across blocks using a
    likelihood-ratio (G) statistic against a chi-squared distribution.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    l : int
        Block length (samples).
    alpha : float
        Significance level (not used for the return value; kept for API
        consistency).
    verbose : bool, optional
        Print the test statistic, degrees of freedom, and p-value
        (default True).

    Returns
    -------
    p : float
        p-value of the conditional homogeneity test.

    References
    ----------
    .. [1] Kullback, S. (1962). Tests for stationarity in the Markov chain.
           *Technometrics*, Table 9.1.
    """
    n = len(x)
    r = int(np.floor(float(n)/float(l))) # number of blocks
    nl = r*l
    f_ijk = _count_block_transitions(x, K, r, l)
    f_ij  = f_ijk.sum(axis=2)
    f_jk  = f_ijk.sum(axis=0)
    f_i   = f_ij.sum(axis=1)
    f_j   = f_jk.sum(axis=1)

    # conditional homogeneity (Markovianity stationarity)
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        # conditional homogeneity
        f = f_ijk[i,j,k]*f_j[j]*f_ij[i,j]*f_jk[j,k]
        if (f > 0):
            num_ = f_ijk[i,j,k]*f_j[j]
            den_ = f_ij[i,j]*f_jk[j,k]
            T += (f_ijk[i,j,k]*np.log(num_/den_))
    T *= 2.0
    df = (r-1)*(K-1)*K
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"[+] Conditional homogeneity test: l = {l:d}")
        print(f"\tdata split in r={r:d} blocks of length {l:d}")
        print(f"\tp: {p:.5f} | t: {T:.3f} | df: {df:.1f}")
    return p


def test_geometric_dist(tau_dict: dict, K: int, fs: float = None,
                        verbose: bool = False,
                        doplot: bool = False) -> ScalarFloatArray:
    """Test whether sojourn-time distributions are geometric.

    For each symbol, compares the observed sojourn-time histogram against
    a fitted geometric distribution using a likelihood-ratio (G) test.

    Parameters
    ----------
    tau_dict : dict
        Sojourn durations; keys ``0..K-1``, values are lists of individual
        durations.  Pass durations in milliseconds together with ``fs``, or
        durations in samples with ``fs=None``.
    K : int
        Number of distinct symbols.
    fs : float, optional
        Sampling rate in Hz.  When provided, durations in ``tau_dict`` are
        interpreted as milliseconds and converted to samples.  When ``None``
        (default), durations are treated as integer sample counts and ``fs``
        is not required for the statistical test; plots label axes in samples.
    verbose : bool, optional
        Print per-symbol test results (default False).
    doplot : bool, optional
        Display bar charts of observed vs. expected distributions (default False).

    Returns
    -------
    p_values : ndarray of float, shape (K,)
        p-value for each symbol's geometric-distribution test.
    """
    if verbose:
        print(f"[+] Geometric distribution test:")

    dt_ms = 1000.0 / fs if fs is not None else 1.0
    duration_unit = 'ms' if fs is not None else 'samples'

    tau_mean_obs = [np.mean(tau_dict[i]) for i in range(K)]

    tau_dist = []
    for k in range(K):
        lens = np.array([int(t / dt_ms) for t in tau_dict[k]], dtype=np.int32)
        tau_dist.append(np.bincount(lens - 1).astype(float).tolist())

    p_values = np.zeros(K)
    if doplot:
        q_list = [] # contains (q_obs, q_exp) as tuples
    for i in range(K): # test for each symbol
        if verbose:
            print(f"\tTesting the distribution of symbol # {i:d}")
        # m = max_tau:
        m = len(tau_dist[i])
        # observed lifetime distribution:
        q_obs = np.zeros(m)
        # theoretical lifetime distribution:
        q_exp = np.zeros(m)
        mu = tau_mean_obs[i]
        T_ii = (mu-1)/mu # reconstruct T_ii
        for j in range(m):
            # observed frequency of lifetime j+1 for state s
            q_obs[j] = tau_dist[i][j]
            # expected frequency
            q_exp[j] = (1-T_ii) * T_ii**j
        q_exp *= sum(q_obs)

        if doplot:
            q_list.append((q_obs, q_exp))

        t = 0.0 # chi2 statistic
        for j in range(m):
            if ((q_obs[j] > 0) & (q_exp[j] > 0)):
                t += (q_obs[j]*np.log(q_obs[j]/q_exp[j]))
        t *= 2.0
        df = m-1
        #p0 = chi2test(t, df, alpha)
        p0 = chi2.sf(t, df, loc=0, scale=1)

        ''' alternative: Chi2 contingency test
        g1, p1, dof1, expctd1 = chi2_contingency(np.vstack((q_obs,q_exp)), \
                                                 lambda_='log-likelihood')
        if verbose:
            print((f"\tG-test (log-likelihood) p: {p1:.5f}, g: {g1:.3f}, "
                   f"df: {dof1:.1f}"))

        # Pearson's Chi2 test
        g2, p2, dof2, expctd2 = chi2_contingency( np.vstack((q_obs,q_exp)) )
        if verbose:
            print((f"\tG-test (Pearson Chi2) p: {p2:.5f}, g: {g2:.3f}, "
                   f"df: {dof2:.1f}"))
        '''

        if verbose:
            print(f"\tp: {p0:.5f} | t: {t:.3f} | df: {df:.1f}")
        p_values[i] = p0

    if doplot:
        w = dt_ms/2 # 0.4
        wh = w/2
        fsize = 14
        fig, ax = plt.subplots(1, K, figsize=(4*K,4), 
                               sharex=True, sharey=True)
        for i in range(K):
            # mean duration
            tau_i = tau_mean_obs[i]
            # observed and expected lifetime histograms for microstate class i
            q_obs = q_list[i][0]
            q_exp = q_list[i][1]
            # normalize histograms
            q_obs /= q_obs.sum()
            q_exp /= q_exp.sum()
            m = len(q_obs)
            x = np.arange(1,m+1)*dt_ms # x-axis
            ax[i].bar(x-wh, q_obs, width=w, color='k', alpha=0.5, 
                      label='observed durations')
            ax[i].bar(x+wh, q_exp, width=w, color='b', alpha=0.5, 
                      label='expected durations')
            ax[i].axvline(tau_i, color='k')
            #ax[i].axvline(tau_i, color='b') # same as obs 1/(1-T_ii)
            ax[i].set_xlabel("duration " + r"$\tau$" + f" ({duration_unit})", fontsize=fsize)
            ax[i].set_ylabel(r"$p\left( \tau \right)$", fontsize=fsize)
            ax[i].text(0.5, 0.4,
                       r"$\tau_m=$"+f"{tau_i:.1f} {duration_unit}\np={p_values[i]:.3f}",
                       color='k',
                       fontsize=fsize,
                       transform=ax[i].transAxes,
                       bbox=dict(facecolor='#dddddd',
                                 edgecolor='black',
                                 alpha=1.0))
        ax[0].legend(loc='upper right', fontsize=fsize)
        plt.suptitle("Geometric/exponential distribution test",
                     fontsize=fsize, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return p_values


def sojourn_distribution_fit(x: ScalarIntArray, K: int, xmin: float = None,
                             discrete: bool = True,
                             distributions: tuple = ("exponential",
                                                     "stretched_exponential",
                                                     "lognormal")) -> list:
    """Fit and contrast candidate sojourn-time distributions via the `powerlaw` package.

    For each symbol, fits a power-law distribution together with the
    requested alternative distributions to the sojourn-time data (sharing
    the same ``x_min``), using the Clauset et al. (2009) maximum-likelihood
    method as implemented by the `powerlaw` package. Each alternative is
    contrasted against the power-law fit with a log-likelihood-ratio test.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    xmin : float, optional
        Lower cutoff for the fitted tail. If ``None`` (default), the
        optimal ``x_min`` is found automatically per symbol (Clauset method).
    discrete : bool, optional
        Treat durations as discrete (integer-valued) data (default True),
        appropriate for sojourn times measured in samples.
    distributions : tuple of str, optional
        Alternative distributions to fit and contrast against
        ``'power_law'``. Any name supported by `powerlaw.Fit` may be used,
        e.g. ``'exponential'``, ``'stretched_exponential'`` (Weibull),
        ``'lognormal'``, ``'truncated_power_law'``
        (default ``('exponential', 'stretched_exponential', 'lognormal')``).

    Returns
    -------
    results : list of dict, length K
        ``results[s]`` has keys:

        ``xmin`` : float
            Fitted (or supplied) lower cutoff.
        ``n_tail`` : int
            Number of observations at or above ``xmin``.
        ``params`` : dict
            Fitted parameters per distribution name, including
            ``'power_law'`` (``alpha``) and each name in ``distributions``.
        ``comparisons`` : dict
            For each name in ``distributions``, ``{'R': float, 'p': float}``
            from ``distribution_compare('power_law', name)``: positive
            ``R`` favors the power law, negative favors the alternative;
            ``p`` is the two-sided significance of that preference.

        A symbol with fewer than 2 sojourns yields ``{'error': ...}``.

    References
    ----------
    .. [1] Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law
           distributions in empirical data. *SIAM Review*, 51(4), 661-703.
    .. [2] Alstott, J., Bullmore, E., & Plenz, D. (2014). powerlaw: A Python
           package for analysis of heavy-tailed distributions. *PLOS ONE*,
           9(1), e85777.
    """
    import powerlaw as pl

    durations_per_symbol = sojourn_times_unordered(x, K)
    results = []

    for durations in durations_per_symbol:
        y = np.asarray(durations, dtype=float)
        if len(y) < 2:
            results.append({"error": f"Too few sojourns ({len(y)})"})
            continue

        fit = pl.Fit(y, xmin=xmin, discrete=discrete, verbose=False)

        params = {"power_law": {"alpha": float(fit.power_law.alpha)}}
        comparisons = {}
        for name in distributions:
            dist_obj = getattr(fit, name)
            params[name] = {
                attr: float(getattr(dist_obj, attr))
                for attr in ("Lambda", "alpha", "beta", "mu", "sigma")
                if hasattr(dist_obj, attr) and getattr(dist_obj, attr) is not None
            }
            R, p = fit.distribution_compare("power_law", name, normalized_ratio=True)
            comparisons[name] = {"R": float(R), "p": float(p)}

        results.append({
            "xmin": float(fit.xmin),
            "n_tail": int(np.sum(y >= fit.xmin)),
            "params": params,
            "comparisons": comparisons,
        })

    return results


def test_geometric_seq(x: ScalarIntArray, K: int, fs: float = None,
                       verbose: bool = False,
                       doplot: bool = False) -> ScalarFloatArray:
    """Test whether sojourn-time distributions are geometric (sequence variant).

    Like :func:`test_geometric_dist` but accepts a raw symbolic sequence
    instead of a pre-computed sojourn dictionary.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    fs : float, optional
        Sampling rate in Hz.  Required only for the plotting section, where
        durations are displayed in milliseconds.  When ``None`` (default),
        the statistical test runs unchanged and plots label axes in samples.
    verbose : bool, optional
        Print per-symbol test results (default False).
    doplot : bool, optional
        Display bar charts of observed vs. expected distributions (default False).

    Returns
    -------
    p_values : ndarray of float, shape (K,)
        p-value for each symbol's geometric-distribution test.
    """
    if verbose:
        print(f"[+] Geometric distribution test:")
    T = tpm_cond(x, K) # (conditional) transition probability matrix

    dt_ms = 1000.0 / fs if fs is not None else 1.0
    duration_unit = 'ms' if fs is not None else 'samples'

    tau_dist = sojourn_time_histograms(x, K)
    tau_mean_obs = [
        np.dot(np.arange(1, len(h)+1), np.array(h)) / np.sum(h) * dt_ms
        for h in tau_dist
    ]

    p_values = np.zeros(K)
    if doplot:
        q_list = [] # contains (q_obs, q_exp) as tuples
    for i in range(K): # test for each symbol
        if verbose:
            print(f"\tTesting the distribution of symbol # {i:d}")
        # m = max_tau:
        m = len(tau_dist[i])
        # observed lifetime distribution:
        q_obs = np.zeros(m)
        # theoretical lifetime distribution:
        q_exp = np.zeros(m)
        for j in range(m):
            # observed frequency of lifetime j+1 for state s
            q_obs[j] = tau_dist[i][j]
            # expected frequency
            q_exp[j] = (1-T[i,i]) * T[i,i]**j
        q_exp *= sum(q_obs)

        if doplot:
            q_list.append((q_obs, q_exp))

        t = 0.0 # chi2 statistic
        for j in range(m):
            if ((q_obs[j] > 0) & (q_exp[j] > 0)):
                t += (q_obs[j]*np.log(q_obs[j]/q_exp[j]))
        t *= 2.0
        df = m-1
        #p0 = chi2test(t, df, alpha)
        p0 = chi2.sf(t, df, loc=0, scale=1)

        ''' alternative: Chi2 contingency test
        g1, p1, dof1, expctd1 = chi2_contingency(np.vstack((q_obs,q_exp)), \
                                                 lambda_='log-likelihood')
        if verbose:
            print((f"\tG-test (log-likelihood) p: {p1:.5f}, g: {g1:.3f}, "
                   f"df: {dof1:.1f}"))

        # Pearson's Chi2 test
        g2, p2, dof2, expctd2 = chi2_contingency( np.vstack((q_obs,q_exp)) )
        if verbose:
            print((f"\tG-test (Pearson Chi2) p: {p2:.5f}, g: {g2:.3f}, "
                   f"df: {dof2:.1f}"))
        '''

        if verbose:
            print(f"\tp: {p0:.5f} | t: {t:.3f} | df: {df:.1f}")
        p_values[i] = p0

    if doplot:
        w = dt_ms/2 # 0.4
        wh = w/2
        fsize = 14
        fig, ax = plt.subplots(1, K, figsize=(4*K,4), 
                               sharex=True, sharey=True)
        for i in range(K):
            # mean duration
            tau_i = tau_mean_obs[i]
            # observed and expected lifetime histograms for microstate class i
            q_obs = q_list[i][0]
            q_exp = q_list[i][1]
            # normalize histograms
            q_obs /= q_obs.sum()
            q_exp /= q_exp.sum()
            m = len(q_obs)
            x = np.arange(1,m+1)*dt_ms # x-axis
            ax[i].bar(x-wh, q_obs, width=w, color='k', alpha=0.5, 
                      label='observed durations')
            ax[i].bar(x+wh, q_exp, width=w, color='b', alpha=0.5, 
                      label='expected durations')
            ax[i].axvline(tau_i, color='k')
            #ax[i].axvline(tau_i, color='b') # same as obs 1/(1-T_ii)
            ax[i].set_xlabel("duration " + r"$\tau$" + f" ({duration_unit})", fontsize=fsize)
            ax[i].set_ylabel(r"$p\left( \tau \right)$", fontsize=fsize)
            ax[i].text(0.5, 0.4,
                       r"$\tau_m=$"+f"{tau_i:.1f} {duration_unit}\np={p_values[i]:.3f}",
                       color='k',
                       fontsize=fsize,
                       transform=ax[i].transAxes,
                       bbox=dict(facecolor='#dddddd',
                                 edgecolor='black',
                                 alpha=1.0))
        ax[0].legend(loc='upper right', fontsize=fsize)
        plt.suptitle("Geometric/exponential distribution test",
                     fontsize=fsize, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return p_values


def test_j_homogeneity(x: ScalarIntArray, K: int, l: int, alpha: float,
                       verbose: bool = True) -> float:
    """Test marginal (j-)homogeneity of a symbolic sequence.

    Splits *x* into non-overlapping blocks of length *l* and tests whether
    the marginal symbol distribution is stationary across blocks (G-test on a
    two-way contingency table).

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    l : int
        Block length (samples).
    alpha : float
        Significance level (kept for API consistency; not used in the return
        value).
    verbose : bool, optional
        Print test details (default True).

    Returns
    -------
    p : float
        p-value of the j-homogeneity test.

    References
    ----------
    .. [1] Kullback, S. (1962). Tests for stationarity in the Markov chain.
           *Technometrics*, Table 3.4.
    """
    n = len(x)
    r = int(np.floor(n/l)) # number of blocks
    nl = r*l # total number of data points in the blocked data
    f_ij = _count_block_symbols(x, K, r, l)
    f_j  = f_ij.sum(axis=0)

    # j-homogeneity
    t = 0.0
    for i in range(r):
        for j in range(K):
            f = f_ij[i][j]*f_j[j]
            if  ( f != 0.0 ):
                t += ( f_ij[i][j] * np.log( (nl*f_ij[i][j]) / (l*f_j[j]) ) )
    t *= 2.0
    df = (r-1)*(K-1)
    #p = chi2test(t, df, alpha)
    p = chi2.sf(t,df,loc=0,scale=1)
    if verbose:
        print("[+] j-homogeneity G-test (two-way table):")
        print(f"\tData split in r = {r:d} blocks of length {l:d}")
        print(f"\tp: {p:.5f} | t: {t:.3f} | df: {df:d}")
    return p


def test_jk_homogeneity(x: ScalarIntArray, K: int, l: int, alpha: float,
                        verbose: bool = True) -> float:
    """Test joint (jk-)homogeneity of transitions in a symbolic sequence.

    Splits *x* into non-overlapping blocks of length *l* and tests whether
    the joint transition distribution is stationary across blocks (G-test on a
    three-way contingency table).

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    l : int
        Block length (samples).
    alpha : float
        Significance level (kept for API consistency; not used in the return
        value).
    verbose : bool, optional
        Print test details (default True).

    Returns
    -------
    p : float
        p-value of the jk-homogeneity test.

    References
    ----------
    .. [1] Kullback, S. (1962). Tests for stationarity in the Markov chain.
           *Technometrics*, Table 3.4.
    """
    n = len(x)
    r = int(np.floor(n/l)) # number of blocks
    nl = r*l # total number of data points in the blocked data
    f_ijk = _count_block_transitions(x, K, r, l)
    f_jk  = f_ijk.sum(axis=0)
    f_i   = f_ijk.sum(axis=(1, 2))

    # j,k-homogeneity
    t = 0.0
    for i in range(r):
        for j in range(K):
            for k in range(K):
                f = f_ijk[i][j][k]*f_jk[j][k]*f_i[i]
                if  ( f != 0.0 ):
                    t += ( f_ijk[i][j][k] * \
                    np.log( (nl*f_ijk[i][j][k]) / (f_i[i]*f_jk[j][k]) ) )
    t *= 2.0
    df = (r-1)*(K*K-1)
    #p = chi2test(t, df, alpha)
    p = chi2.sf(t,df,loc=0,scale=1)
    if verbose:
        print("[+] j,k-homogeneity G-test (three-way table):")
        print(f"\tData split in r = {r:d} blocks of length {l:d}")
        print(f"\tp: {p:.3f} | t: {t:.3f} | df: {df:d}")
    return p


def test_markov0(x: ScalarIntArray, K: int, verbose: bool = False) -> float:
    """Test zero-order Markovianity (i.i.d. assumption).

    Tests whether successive symbols are independent (i.e. whether the
    sequence is i.i.d.) using a likelihood-ratio (G) statistic against a
    chi-squared distribution.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    verbose : bool, optional
        Print the test statistic, degrees of freedom, and p-value
        (default False).

    Returns
    -------
    p : float
        p-value.  Small values reject the i.i.d. null hypothesis.

    Notes
    -----
    H₀: :math:`P(X_{t+1} = j) = P(X_{t+1} = j \mid X_t = i)`
    (successive symbols are independent).

    References
    ----------
    .. [1] Kullback, S. (1962). Tests for stationarity in the Markov chain.
           *Technometrics*.
    """
    n = len(x)
    f_ij = _count_transitions(x, n, K)
    f_i  = f_ij.sum(axis=1)
    f_j  = f_ij.sum(axis=0)
    T = 0.0 # statistic
    for i, j in np.ndindex(f_ij.shape):
        f = f_ij[i,j]*f_i[i]*f_j[j]
        if (f > 0):
            num_ = n*f_ij[i,j]
            den_ = f_i[i]*f_j[j]
            T += (f_ij[i,j] * np.log(num_/den_))
    T *= 2.0
    df = (K-1.0) * (K-1.0)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print("[+] Markov test (zero-order):")
        print(f"\tp: {p:.5f} | t: {T:.3f} | df: {df:.1f}")
    return p


def test_markov1(x: ScalarIntArray, K: int, verbose: bool = False) -> float:
    """Test first-order Markovianity of a symbolic sequence.

    Tests whether the sequence satisfies the first-order Markov property
    (i.e. whether two-step transitions are consistent with one-step
    transitions) using a likelihood-ratio (G) statistic.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    verbose : bool, optional
        Print test details (default False).

    Returns
    -------
    p : float
        p-value.  Small values reject the first-order Markov null hypothesis.

    Notes
    -----
    H₀: :math:`P(X_{t+1} \mid X_t) = P(X_{t+1} \mid X_t, X_{t-1})`.

    References
    ----------
    .. [1] Kullback, S. (1962). Tests for stationarity in the Markov chain.
           *Technometrics*, Tables 8.1, 8.2, 8.6.
    """
    n = len(x)
    f_ijk = _count_markov1(x, n, K)
    f_ij  = f_ijk.sum(axis=2)
    f_jk  = f_ijk.sum(axis=0)
    f_j   = f_jk.sum(axis=1)
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        f = f_ijk[i][j][k]*f_j[j]*f_ij[i][j]*f_jk[j][k]
        if (f > 0):
            num_ = f_ijk[i,j,k]*f_j[j]
            den_ = f_ij[i,j]*f_jk[j,k]
            T += (f_ijk[i,j,k]*np.log(num_/den_))
    T *= 2.0
    df = K*(K-1)*(K-1)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print("[+] Markov test (first-order):")
        print(f"\tp: {p:.5f} | t: {T:.3f} | df: {df:.1f}")
    return p


def test_markov2(x: ScalarIntArray, K: int, verbose: bool = False) -> float:
    """Test second-order Markovianity of a symbolic sequence.

    Tests whether a first-order Markov chain is sufficient, or whether a
    second-order model is needed, using a likelihood-ratio (G) statistic.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    verbose : bool, optional
        Print test details (default False).

    Returns
    -------
    p : float
        p-value.  Small values reject the first-order Markov null hypothesis
        in favour of a second-order model.

    Notes
    -----
    H₀: :math:`P(X_{t+1} \mid X_t, X_{t-1}) = P(X_{t+1} \mid X_t, X_{t-1}, X_{t-2})`.

    References
    ----------
    .. [1] Kullback, S. (1962). Tests for stationarity in the Markov chain.
           *Technometrics*, Table 10.2.
    """
    n = len(x)
    f_ijkl = _count_markov2(x, n, K)
    f_ijk  = f_ijkl.sum(axis=3)
    f_jkl  = f_ijkl.sum(axis=0)
    f_jk   = f_jkl.sum(axis=2)
    T = 0.0
    for i, j, k, l in np.ndindex(f_ijkl.shape):
        f = f_ijkl[i,j,k,l]*f_ijk[i,j,k]*f_jkl[j,k,l]*f_jk[j,k]
        if (f > 0):
            num_ = f_ijkl[i,j,k,l]*f_jk[j,k]
            den_ = f_ijk[i,j,k]*f_jkl[j,k,l]
            T += (f_ijkl[i,j,k,l]*np.log(num_/den_))
    T *= 2.0
    df = K*K*(K-1)*(K-1)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print("[+] Markov test (second-order):")
        print(f"\tp: {p:.5f} | t: {T:.3f} | df: {df:.1f}")
    return p


def test_symmetry(x: ScalarIntArray, K: int, verbose: bool = True) -> float:
    """Test symmetry of the transition matrix.

    Tests whether :math:`P(X_{t+1}=j \mid X_t=i) = P(X_{t+1}=i \mid X_t=j)`
    for all *i, j* using a likelihood-ratio (G) statistic.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    verbose : bool, optional
        Print test details (default True).

    Returns
    -------
    p : float
        p-value.  Small values reject the symmetry (detailed balance) null.

    References
    ----------
    .. [1] Kullback, S. (1962). Tests for stationarity in the Markov chain.
           *Technometrics*.
    """
    n = len(x)
    f_ij = _count_transitions(x, n, K)
    T = 0.0
    for i, j in np.ndindex(f_ij.shape):
        if (i != j):
            f = f_ij[i,j]*f_ij[j,i]
            if (f > 0):
                num_ = 2*f_ij[i,j]
                den_ = f_ij[i,j]+f_ij[j,i]
                T += (f_ij[i,j]*np.log(num_/den_))
    T *= 2.0
    df = K*(K-1)/2
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"[+] Symmetry test:")
        print(f"\tp: {p:.5f} | t: {T:.3f} | df: {df:.1f}")
    return p


def test_transition_matrix(x: ScalarIntArray, K: int, T_ref: ScalarIntArray,
                           verbose: bool = True) -> float:
    """Test an empirical transition matrix against a reference matrix.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    T_ref : ndarray of float, shape (K, K)
        Reference (expected) transition matrix.
    verbose : bool, optional
        Print test details (default True).

    Returns
    -------
    p : float
        p-value.  Small values indicate the empirical matrix deviates
        significantly from *T_ref*.

    References
    ----------
    .. [1] Kullback, S. (1962). Tests for stationarity in the Markov chain.
           *Technometrics*, Eq. 7.2, Table 7.2.
    """
    n = len(x)
    f_ij = _count_transitions(x, n, K)
    f_i  = f_ij.sum(axis=1)
    T = 0.0
    for i in range(K):
        for j in range(K):
            if (T_ref[i,j] > 0):
                T += (f_ij[i,j]*np.log(f_ij[i,j]/(f_i[i]*T_ref[i,j])))
    T *= 2.0
    df = K*(K-1)/2
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"[+] Symmetry test:")
        print(f"\tp: {p:.5f} | t: {T:.3f} | df: {df:.1f}")
    return p


def transition_syntax_test(file_list: list,
                    K: int,
                    n_perm: int = 5000,
                    verbose: bool = True) -> Tuple[float, float]:
    """Microstate transition-matrix syntax test across subjects.

    For each sequence in *file_list*, estimates the empirical joint transition
    matrix *p* and the expected matrix *q* (under the independence model), then
    aggregates across subjects and tests whether *p* deviates from *q* using a
    chi-squared statistic with a permutation null distribution.

    Parameters
    ----------
    file_list : list of str
        Paths to ``.npy`` files, each containing a 1-D integer microstate
        sequence.
    K : int
        Number of distinct symbols (assumed identical across all sequences).
    n_perm : int, optional
        Number of permutations for the null distribution (default 5000).
    verbose : bool, optional
        Print progress and summary statistics (default True).

    References
    ----------
    .. [1] Lehmann, D. et al. (2005). EEG microstate duration and syntax in
           attention-deficit hyperactivity disorder.
           *Psychiatry Research: Neuroimaging*, 138, 141-152.
    """
    alpha = 0.05 # significance level
    n_files = len(file_list)
    if n_files < 2:
        raise ValueError(("transition_syntax_test requires at least 2 files (subjects)"
                          " to form a permutation null distribution."))
    p_arr = np.zeros((n_files,K,K)) # 1: observed transition matrices
    q_arr = np.zeros((n_files,K,K)) # 2, 3: expected transition matrices
    # get off-diagonal matrix indices to avoid division by zero along the diag.
    uti = np.triu_indices(K, 1) # upper triangle indices
    lti = np.tril_indices(K,-1) # lower triangle indices
    odi = (np.hstack((uti[0], lti[0])), 
           np.hstack((uti[1], lti[1]))) # off-diagonal indices
    for i, f in enumerate(file_list):
        if verbose:
            print(f"\tFile {i+1:d}/{n_files:d}", end="\r")
        x = np.load(f)
        y, _ = embedded_process(x)
        p = pmf(y, K)  # frequency-weighted marginal from jump process
        p_arr[i,:,:] = tpm_joint(y, K)
        q_arr[i,:,:] = tpm_joint_exp(p)
    if verbose:
        print("")
    p_mean = p_arr.mean(axis=0)
    q_mean = q_arr.mean(axis=0)
    p_q = np.zeros((K, K))
    p_q[odi] = (p_mean[odi] - q_mean[odi]) ** 2 / q_mean[odi]
    t = np.sum(p_q[odi])
    if verbose:
        print("Computing null distribution from random obs/exp assignments")
    D     = p_arr - q_arr
    p_sum = p_arr.sum(axis=0)
    q_sum = q_arr.sum(axis=0)
    t0 = _permutation_null(D, p_sum, q_sum, n_files,
                           odi[0].astype(np.int64), odi[1].astype(np.int64), n_perm)
    t_thr = np.percentile(t0, 100 * (1 - alpha))
    if verbose:
        print(f"Test statistic: {t:.2e}")
        print(f"Chi2 threshold: {t_thr:.2e}")
        if t > t_thr:
            print("Permutation test IS significant.\n"
                  "Some transition rate(s) are apparently not random.")
        else:
            print("Permutation test is NOT significant.\n"
                  "All transition rates are within random range.")
    return t, t_thr

"""pytest test suite for the mstsa package.

Run with:  pytest tests/test_mstsa.py -v
"""

import numpy as np
import pytest

import sys
sys.path.insert(0, '..')

from mstsa import (
    # entropy
    h1, entropy_rate, renyi_entropy, tsallis_entropy,
    # information
    aif,
    # sequence utilities
    embedded_process, sojourn_time_histograms,
    partitions, randomwalk,
    # scaling
    dfa, rescaled_range, hurst_exponents_dfa, hurst_exponents_rs,
    # complexity
    lz76,
    # Markov chain
    tpm_cond, tpm_joint, tpm_joint_exp,
    p_stationary, generator_matrix,
    mc_sample_path, pmf, p_ngram_mc,
    max_entropy_T,
    # ACD / duration analysis
    simulate_acd, fit_acd, select_acd_order, acd_analysis,
    # sojourn-time distribution fitting
    sojourn_distribution_fit,
)

K   = 4
RNG = np.random.default_rng(42)
SEQ = RNG.integers(0, K, size=10_000)   # reusable i.i.d. sequence


# ── 1. entropy: known analytic values ────────────────────────────────────────

def test_h1_uniform():
    """Perfectly uniform sequence → H1 = log2(K)."""
    x = np.tile(np.arange(K), 2500)        # exactly 2500 of each symbol
    assert h1(x, K) == pytest.approx(np.log2(K), rel=1e-6)


def test_h1_constant():
    """Constant sequence → H1 = 0."""
    x = np.zeros(1000, dtype=int)
    assert h1(x, K) == pytest.approx(0.0, abs=1e-10)


def test_renyi_uniform_any_order():
    """Rényi entropy of uniform distribution = log2(K) for any order a ≠ 1."""
    p = np.ones(K) / K
    for a in [0.5, 2.0, 5.0]:
        assert renyi_entropy(p, a) == pytest.approx(np.log2(K), rel=1e-6), f"failed at a={a}"


def test_renyi_degenerate():
    """Rényi entropy of a degenerate distribution = 0 for any order."""
    p = np.array([1.0, 0.0, 0.0, 0.0])
    for a in [0.5, 2.0, 5.0]:
        assert renyi_entropy(p, a) == pytest.approx(0.0, abs=1e-10), f"failed at a={a}"


def test_tsallis_q1_equals_shannon():
    """Tsallis entropy at q=1 reduces to Shannon entropy."""
    p = np.array([0.1, 0.4, 0.3, 0.2])
    h_shannon = -np.sum(p * np.log2(p))
    assert tsallis_entropy(p, q=1.0) == pytest.approx(h_shannon, rel=1e-6)


def test_tsallis_degenerate():
    """Tsallis entropy of a degenerate distribution = 0 for any order."""
    p = np.array([1.0, 0.0, 0.0, 0.0])
    for q in [0.5, 2.0, 5.0]:
        assert tsallis_entropy(p, q=q) == pytest.approx(0.0, abs=1e-10), f"failed at q={q}"


def test_entropy_rate_le_h1():
    """Entropy rate ≤ H1 for any sequence (conditioning reduces entropy)."""
    er, _ = entropy_rate(SEQ, K, kmax=4, doplot=False)
    assert er <= h1(SEQ, K) + 1e-10


# ── 2. algebraic invariants ───────────────────────────────────────────────────

def test_tpm_cond_rows_sum_to_one():
    T = tpm_cond(SEQ, K)
    np.testing.assert_allclose(T.sum(axis=1), np.ones(K), atol=1e-12)


def test_p_stationary_is_fixed_point():
    """Stationary distribution satisfies π T = π."""
    T = tpm_cond(SEQ, K)
    p = p_stationary(T)
    np.testing.assert_allclose(p @ T, p, atol=1e-10)


def test_generator_matrix_rows_sum_to_zero():
    """Every row of a CTMC generator matrix sums to zero."""
    Q = generator_matrix(x=SEQ)
    np.testing.assert_allclose(Q.sum(axis=1), np.zeros(K), atol=1e-12)


def test_max_entropy_T_reproduces_stationary_dist():
    """T* is zero-diagonal, row-stochastic, and has stationary distribution p."""
    p = np.array([0.318, 0.230, 0.262, 0.191])
    T, info = max_entropy_T(p)
    np.testing.assert_allclose(np.diag(T), np.zeros(len(p)), atol=1e-10)
    np.testing.assert_allclose(T.sum(axis=1), np.ones(len(p)), atol=1e-10)
    np.testing.assert_allclose(p @ T, p, atol=1e-10)
    assert info["marginal_error"] < 1e-10


def test_max_entropy_T_raises_on_infeasible_or_small_K():
    with pytest.raises(ValueError):
        max_entropy_T([0.6, 0.2, 0.2])  # max(p) > 1/2
    with pytest.raises(ValueError):
        max_entropy_T([0.5, 0.5])  # K < 3


def test_tpm_joint_jump_process_sums_to_one():
    """Joint transition matrix of a jump process (no self-transitions) sums to 1."""
    y, _ = embedded_process(SEQ)
    assert tpm_joint(y, K).sum() == pytest.approx(1.0, abs=1e-10)


def test_tpm_joint_exp_sums_to_one():
    """Expected joint transition matrix sums to 1 for any valid marginal."""
    p = np.array([0.15, 0.35, 0.25, 0.25])
    assert tpm_joint_exp(p).sum() == pytest.approx(1.0, abs=1e-10)


# ── 3. structural / shape checks ──────────────────────────────────────────────

def test_partitions_k4_count_and_structure():
    """partitions(4) returns the 3 balanced binary partitions of {0,1,2,3}."""
    parts = partitions(K)
    assert len(parts) == 3
    for part in parts:
        assert len(part) == 2
        assert len(part[0]) + len(part[1]) == K
        assert set(part[0]) | set(part[1]) == set(range(K))
        assert set(part[0]) & set(part[1]) == set()


def test_randomwalk_values_and_shape():
    """`randomwalk` returns a ±1 array of the same length as the input."""
    part = partitions(K)[0]
    rw = randomwalk(SEQ, part)
    assert rw.shape == SEQ.shape
    assert set(np.unique(rw)).issubset({-1.0, 1.0})


def test_sojourn_time_histograms_known_sequence():
    """Exact histogram counts on a hand-crafted sequence."""
    x = np.array([0, 0, 1, 1, 1, 0, 2, 2])
    hists = sojourn_time_histograms(x, K=3)
    # k=0: two runs, lengths 2 and 1 → one of length-1, one of length-2
    assert hists[0] == [1.0, 1.0]
    # k=1: one run of length 3
    assert hists[1] == [0.0, 0.0, 1.0]
    # k=2: one run of length 2
    assert hists[2] == [0.0, 1.0]


def test_embedded_process_no_consecutive_repeats():
    """`embedded_process` output contains no consecutive repeated symbols."""
    y, _ = embedded_process(SEQ)
    assert np.all(np.diff(y) != 0)


def test_hurst_exponents_dfa_shape():
    h = hurst_exponents_dfa(SEQ, lmin=10, lmax=500, fitmin=10, fitmax=500, nsteps=15)
    assert h.shape == (K,)


def test_hurst_exponents_rs_shape():
    h = hurst_exponents_rs(SEQ)
    assert h.shape == (K,)


# ── 4. ordering / monotonicity ────────────────────────────────────────────────

def test_lz76_constant_less_than_random():
    """LZ76 complexity of a constant sequence < random sequence."""
    x_const = np.zeros(1000, dtype=int)
    x_rand  = RNG.integers(0, K, 1000)
    assert lz76(x_const) < lz76(x_rand)


def test_dfa_white_noise_hurst_near_half():
    """DFA on i.i.d. Gaussian increments → H ≈ 0.5."""
    x = np.random.default_rng(0).standard_normal(10_000)
    h = dfa(x, lmin=10, lmax=1000, fitmin=10, fitmax=1000, nsteps=20)
    assert h == pytest.approx(0.5, abs=0.1)


def test_rescaled_range_white_noise_hurst_near_half():
    """R/S on i.i.d. Gaussian increments → H ≈ 0.5."""
    x = np.random.default_rng(1).standard_normal(10_000)
    h = rescaled_range(x)
    assert h == pytest.approx(0.5, abs=0.15)


# ── 5. round-trip / consistency ───────────────────────────────────────────────

def test_mc_sample_path_recovers_tpm():
    """TPM estimated from a long Markov surrogate ≈ generating TPM."""
    T_true = tpm_cond(SEQ, K)
    p      = p_stationary(T_true)
    x_mc   = mc_sample_path(T=T_true, p=p, n=100_000)
    T_est  = tpm_cond(x_mc, K)
    np.testing.assert_allclose(T_est, T_true, atol=0.02)


def test_p_ngram_mc_in_unit_interval():
    """p_ngram_mc returns values in [0, 1] for all 2-grams."""
    T = tpm_cond(SEQ, K)
    p = p_stationary(T)
    for i in range(K):
        for j in range(K):
            val = p_ngram_mc(np.array([i, j]), p, T)
            assert 0.0 <= val <= 1.0


def test_aif_lag0_equals_h1():
    """AIF at lag 0 = H1 (mutual information of X with itself = self-entropy)."""
    aif_vals = aif(SEQ, K, kmax=5)
    assert aif_vals[0] == pytest.approx(h1(SEQ, K), rel=1e-6)


# ── 9. ACD / duration analysis ───────────────────────────────────────────────

def test_acd_fit_recovers_known_persistence():
    """Fitting simulated ACD(1,1) durations recovers alpha+beta persistence."""
    omega, alpha1, beta1 = 1.0, 0.15, 0.6
    durations = simulate_acd(omega, alpha1, beta1, n=3000, dist="exponential",
                             rng=np.random.default_rng(0))
    model, residuals = fit_acd(durations, p=1, q=1, dist="exponential")
    assert model.converged_
    assert (model.alpha[0] + model.beta[0]) == pytest.approx(alpha1 + beta1, abs=0.1)


def test_select_acd_order_returns_valid_order():
    """select_acd_order picks an order within the searched grid with finite IC."""
    durations = simulate_acd(1.0, 0.1, 0.7, n=1500, dist="exponential",
                             rng=np.random.default_rng(1))
    result = select_acd_order(durations, p_max=2, q_max=2, dist="exponential")
    assert "error" not in result
    assert 1 <= result["p"] <= 2
    assert 1 <= result["q"] <= 2
    assert np.isfinite(result["AIC"])
    assert np.isfinite(result["BIC"])


def test_acd_analysis_shape():
    """acd_analysis returns one result dict per symbol."""
    results = acd_analysis(SEQ, K, p=1, q=1, dist="exponential")
    assert len(results) == K
    for r in results:
        assert "n_sojourns" in r or "error" in r


# ── 10. sojourn-time distribution fitting (powerlaw) ─────────────────────────

def test_sojourn_distribution_fit_returns_expected_structure():
    """sojourn_distribution_fit returns one result dict per symbol with fitted params."""
    pytest.importorskip("powerlaw")
    results = sojourn_distribution_fit(SEQ, K)
    assert len(results) == K
    for r in results:
        if "error" in r:
            continue
        assert r["xmin"] > 0
        assert "power_law" in r["params"]
        assert set(r["comparisons"]) == {"exponential", "stretched_exponential", "lognormal"}

# mstsa — Microstate Time Series Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

`mstsa` is a Python package for information-theoretic and statistical analysis
of discrete symbolic time series, with a particular focus on EEG microstate
sequences.

The source code is available at [GitHub](https://github.com/Frederic-vW/mstsa).

---

## Features

### Symbol statistics
- Empirical probability mass function (`pmf`)
- Transition probability matrices — conditional (`tpm_cond`), joint
  (`tpm_joint`), expected under independence (`tpm_joint_exp`)
- Sojourn-time distributions (`sojourn_time_histograms`,
  `sojourn_times_unordered`)
- Microstate duration, occurrence, and coverage (`dur_occ_cov`)
- Embedded jump process extraction (`embedded_process`)

### Information-theoretic measures
- Shannon entropy (`h1`), joint entropy (`h2`), *k*-gram entropy (`hk`)
- Rényi entropy (`renyi_entropy`), Tsallis entropy (`tsallis_entropy`)
- Topological entropy (`topological_entropy`), dispersion entropy
  (`dispersion_entropy`), diffusion entropy analysis of microstate
  indicator functions (`diffusion_entropy`)
- Entropy rate and excess entropy by linear regression (`entropy_rate`)
- Active information storage (`ais`)
- Auto-information function — empirical (`aif`) and Markov-predicted (`aif_mc`)
- Partial auto-information function (`paif`)
- Sample entropy for continuous signals (`sample_entropy_cont`,
  `sample_entropy_fast_cont_py`) and discrete sequences (`sample_entropy_disc`,
  `sample_entropy_fast_disc_py`)
- Lempel-Ziv complexity (`lz76`)

### Markov chain analysis
- First-order Markov chain surrogate generation (`mc_sample_path`)
- Theoretical entropy rate under Markov assumption (`entropy_rate_mc`, `entropy_rates_mc`)
- Theoretical auto-information function of a Markov chain (`aif_mc`)
- Expected sample entropy under a Markov model (`sample_entropy_mc`, `sample_entropies_mc`)
- Theoretical duration, occurrence, and coverage under a Markov model (`dur_occ_cov_mc`)
- Joint and trajectory probabilities under a Markov model (`p_joint_mc`, `p_ngram_mc`)
- Continuous-time Markov chain generator matrix (`generator_matrix`)
- Stationary distribution (`p_stationary`)
- Relaxation time / spectral gap (`relaxation_time`)
- Nearest reversible Markov chain (`nearest_reversible_mc`, requires `cvxopt`)
- Maximum-entropy-rate zero-diagonal transition matrix for a target
  stationary distribution, via Sinkhorn scaling (`max_entropy_T`)

### Duration / ACD analysis (requires `mstsa[acd]`)
- Autoregressive Conditional Duration model, ACD(p, q), with exponential,
  Weibull, or lognormal innovations (`ACD`, `fit_acd`)
- Ljung-Box residual autocorrelation test (`test_acd_residuals`)
- AIC/BIC-based ACD order selection (`select_acd_order`)
- ACD(p, q) duration simulation / surrogate generation (`simulate_acd`)
- End-to-end per-symbol ACD analysis of a microstate sequence (`acd_analysis`)
- ACF/PACF diagnostic plot of sojourn-duration series (`plot_duration_acf_pacf`)

### Statistical tests (Kullback-Technometrics framework)
- Zero-order Markovianity / i.i.d. test (`test_markov0`)
- First-order Markovianity test (`test_markov1`)
- Second-order Markovianity test (`test_markov2`)
- Conditional homogeneity / stationarity (`test_cond_homogeneity`)
- Marginal homogeneity (`test_j_homogeneity`)
- Joint transition homogeneity (`test_jk_homogeneity`)
- Transition-matrix symmetry / detailed balance (`test_symmetry`)
- Geometric (exponential) sojourn-time test (`test_geometric_seq`, `test_geometric_dist`)
- Sojourn-time distribution fitting and contrast — exponential, stretched
  exponential (Weibull), lognormal, and power-law, via the `powerlaw`
  package (`sojourn_distribution_fit`, requires `mstsa[acd]`)
- Test against reference transition matrix (`test_transition_matrix`)
- Microstate syntax / transition test across subjects (`transition_syntax_test`)

### Other analyses
- Detrended fluctuation analysis (`dfa`) and rescaled-range analysis
  (`rescaled_range`), with Hurst exponents of microstate indicator functions
  (`hurst_exponents_dfa`, `hurst_exponents_rs`)
- Detailed balance statistics (`detailed_balance`, `detailed_balance_cond`,
  `detailed_balance_joint`)
- Random walk construction from symbolic sequence (`randomwalk`, `partitions`)
- Power spectral densities of characteristic microstate functions (`spectra`)
- EEG complexity measures Ω, Σ, Φ, LC (`complexity_ospl`)
- Multiple-comparisons correction (`multiple_comparisons`)
- Microstate syntax analysis (`Microsynt`)

### Optional C extensions
For improved performance on entropy computations (`hk`), sample entropy, and
Lempel-Ziv complexity, `mstsa` builds optional CFFI-linked C extensions
(`_c_entropy`, `_se`, `_lz76`) automatically during installation, provided a
C compiler is available on the system (e.g. `gcc`/`build-essential` on
Linux, Xcode Command Line Tools on macOS, MSVC Build Tools on Windows). No
manual build step is required.

If a compiler is unavailable or a compiled extension otherwise fails to
import, `mstsa` falls back automatically to pure-Python / Numba
implementations and emits a `UserWarning` noting the slower fallback is in
use.

---

## Installation

### From PyPI (once published)

```bash
pip install mstsa
```

### From source

```bash
git clone https://github.com/Frederic-vW/mstsa.git
cd mstsa/packaging
pip install -e .
```

### Optional dependencies

```bash
pip install mstsa[qp]    # nearest_reversible_mc (requires cvxopt)
pip install mstsa[mc_tests]  # multiple_comparisons (requires statsmodels)
pip install mstsa[acd]   # ACD models and sojourn_distribution_fit (requires statsmodels, powerlaw)
```

---

## Quick start

```python
import numpy as np
from mstsa import pmf, tpm_cond, h1, entropy_rate, aif, test_markov0, test_markov1

# Load or create a symbolic sequence
rng = np.random.default_rng(42)
x = rng.integers(0, 4, size=10000)
nc = 4  # number of symbols

# Symbol distribution and transition matrix
p = pmf(x, nc)
T = tpm_cond(x, nc)

# Shannon entropy and entropy rate
print(f"Shannon entropy: {h1(x, nc):.3f} bits")
er, ee = entropy_rate(x, nc, kmax=8)
print(f"Entropy rate: {er:.3f} bits/sample")

# Auto-information function
aif_vals = aif(x, nc, kmax=20)

# Markov order tests
p0 = test_markov0(x, nc, verbose=True)
p1 = test_markov1(x, nc, verbose=True)
```

---

## Citation

If you use `mstsa` in your research, please cite:

> von Wegner, F. (2026). mstsa: Microstate Time Series Analysis.
> *Journal of Open Source Software* (submitted).

---

## License

MIT — see [LICENSE](LICENSE).

"""mstsa: Microstate Time Series Analysis.

A Python package for information-theoretic and statistical analysis
of discrete symbolic time series, with a focus on EEG microstate sequences.
"""

from .mstsa import (
    # Entropy
    h1,
    h2,
    hk,
    entropy_rate,
    renyi_entropy,
    tsallis_entropy,
    topological_entropy,
    dispersion_entropy,
    diffusion_entropy,
    # Information / auto-information
    aif,
    ais,
    paif,
    # Sample entropy (discrete)
    sample_entropy_disc,
    sample_entropy_fast_disc_py,
    # Sequence utilities
    sojourn_time_histograms,
    sojourn_times_unordered,
    dur_occ_cov,
    embedded_process,
    randomwalk,
    spectra,
    dfa,
    rescaled_range,
    hurst_exponents_dfa,
    hurst_exponents_rs,
    lz76,
    partitions,
)

from .markov import (
    # Entropy
    entropy_rate_mc,
    entropy_rates_mc,
    # Information / auto-information
    aif_mc,
    # Sample entropy
    sample_entropy_mc,
    sample_entropies_mc,
    # Markov chain
    mc_sample_path,
    dur_occ_cov_mc,
    generator_matrix,
    p_joint_mc,
    p_stationary,
    p_ngram_mc,
    relaxation_time,
    nearest_reversible_mc,
    max_entropy_T,
    tpm_cond,
    tpm_joint,
    tpm_joint_exp,
    # Detailed balance
    detailed_balance,
    detailed_balance_cond,
    detailed_balance_joint,
    # Sequence utilities
    pmf,
)

from .eeg import (
    complexity_ospl,
    # Sample entropy (continuous)
    sample_entropy_cont,
    sample_entropy_fast_cont_py,
)

from .microsynt import Microsynt

from .stats import (
    multiple_comparisons,
    test_cond_homogeneity,
    test_geometric_dist,
    test_geometric_seq,
    test_j_homogeneity,
    test_jk_homogeneity,
    test_markov0,
    test_markov1,
    test_markov2,
    test_symmetry,
    test_transition_matrix,
    transition_syntax_test,
    sojourn_distribution_fit,
)

from .acd import (
    ACD,
    fit_acd,
    test_acd_residuals,
    select_acd_order,
    simulate_acd,
    acd_analysis,
    plot_duration_acf_pacf,
)

__version__ = "0.4.2"
__author__ = "Frederic von Wegner"
__email__ = "fvw.github@gmail.com"

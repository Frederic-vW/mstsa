---
title: 'mstsa: A Python Library for Information-Theoretic and Statistical Analysis of EEG Microstate Sequences'
tags:
  - Python
  - EEG
  - neuroscience
  - information theory
  - symbolic dynamics
  - time series
  - microstates
authors:
  - name: Frederic von Wegner
    orcid: 0000-0002-6779-9959
    corresponding: true
    affiliation: 1
affiliations:
  - name: School of Biomedical Sciences, University of New South Wales (UNSW)
    index: 1
date: 30 May 2026
bibliography: paper.bib
---

# Summary

EEG microstate analysis decomposes a multichannel EEG recording into a
discrete symbolic time series: each sample is assigned to one of a small
number of prototypical scalp topographies (the microstates), typically at
sampling rates of 250–1000 symbols per second [@lehmann1998brain;
@pascual1995segmentation; @michel2018review].  Once the segmentation step is
complete, the scientific question shifts to the temporal structure of the
resulting sequence: Is it Markovian? Stationary? How complex is it, and how
does that complexity compare to a random null model?

`mstsa` (Microstate Time Series Analysis) is a Python library that addresses
these questions.  It provides a comprehensive set of information-theoretic,
Markov-chain, and statistical tools for the analysis of discrete symbolic
sequences, with EEG microstate data as the primary application.  The library
exposes six modules:

- **`mstsa`** (core): Shannon, Rényi, and Tsallis entropies; block, topological
  [@adler1965topological], and dispersion [@rostaghi2016dispersion] entropies;
  entropy rate and excess entropy [@cover2006elements]; auto-information
  function (AIF) and partial auto-information function (PAIF)
  [@vonwegner2016fluctuation; @vonwegner2018partial]; active information storage
  (AIS) [@lizier2014jidt]; sample entropy for discrete sequences, using
  exact-match comparison and a runs-tracking algorithm that computes estimates
  for all template lengths in a single O(N²) pass
  [@richman2000physiological; @lake2002sample]; Lempel–Ziv complexity
  [@lempel1976complexity; @kaspar1987complexity]; detrended fluctuation
  analysis (DFA), rescaled-range (R/S) analysis, and diffusion entropy analysis
  (DEA) [@scafetta2002diffusion], applicable to both binary-partition random
  walks and to indicator functions (one-hot encodings) of individual
  microstate classes [@peng1994mosaic; @hurst1951storage];
  microstate duration, occurrence, and coverage; power spectral densities of
  indicator functions; and random-walk construction from binary partitions
  [@vandevill2010scalefree; @vonwegner2016fluctuation].
- **`markov`**: Conditional and joint transition probability matrices,
  stationary distribution, continuous-time generator matrix, AIF and sample
  entropy under first-order Markov models [@vonwegner2026comparison], surrogate
  sequence generation [@vonwegner2025higherorder], nearest-reversible Markov
  chain projection
  [@nielsen2015nearest], and detailed-balance statistics
  [@roldan2010dissipation; @lynn2021broken].
- **`stats`**: Likelihood-ratio (G) tests for zero-, first-, and second-order
  Markovianity; j- and jk-homogeneity (stationarity); conditional homogeneity;
  transition-matrix symmetry [@kullback1962tests]; geometric sojourn-time
  distributions; sojourn-time distribution fitting that contrasts power-law,
  exponential, stretched-exponential, and lognormal candidates via the
  `powerlaw` package [@clauset2009powerlaw; @alstott2014powerlaw]; and a
  permutation-based multi-subject transition syntax test [@lehmann2005eeg].
- **`acd`**: Autoregressive Conditional Duration (ACD) models
  [@engle1998autoregressive] of sojourn-time sequences, with exponential,
  Weibull, or lognormal innovations, AIC/BIC-based order selection, a
  Ljung–Box residual autocorrelation test, ACD surrogate-sequence simulation,
  and ACF/PACF diagnostic plots — quantifying serial clustering between
  consecutive sojourn durations, complementary to the marginal-distribution
  fitting in `stats`.
- **`eeg`**: Wackermann/Palus multichannel EEG complexity measures — effective
  dimensionality Ω, effective voltage Σ, generalised frequency Φ, and Palus
  complexity coefficient [@wackermann1996complexity; @palus1992spatio]; and
  continuous sample entropy (`sample_entropy_cont`), using tolerance-based
  matching [@richman2000physiological; @lake2002sample].
- **`microsynt`**: Python implementation of the Microsynt syntactic analysis
  method [@artoni2023microsynt].

Computationally demanding routines are accelerated by Numba JIT compilation
with persistent disk caching [@numba] and optional CFFI-linked C extensions
for block entropy, Lempel–Ziv complexity, and continuous and discrete sample
entropy.

# Statement of Need

EEG microstate research spans five decades and has produced a rich body of
results linking microstate dynamics to cognition, sleep, anaesthesia, and
neurological disorders [@michel2018review].  The standard analysis pipeline
has two stages: (1) identifying the prototypical maps by clustering, and (2)
analysing the resulting symbolic sequence.

Stage 1 is well covered by existing software.  CARTOOL [@brunet2011cartool],
the EEGLAB microstate toolbox [@poulsen2018microstate], MICROSTATELAB
[@kalburgi2023microstatelab], and the Python library Pycrostates
[@ferat2022pycrostates] all provide mature clustering and segmentation tools.
Pycrostates in particular integrates cleanly with the MNE-Python ecosystem
and supports both subject- and group-level analyses.

Stage 2 is far less systematically served.  The conventional summary statistics
— mean duration, occurrence rate, fractional coverage, and transition
probabilities — have been available for decades, but they characterise the
*marginal* and *first-order* structure of the sequence only.  The
information-theoretic characterisation that reveals *higher-order* temporal
dependencies — entropy rate, excess entropy, auto-information structure, and
Lempel–Ziv complexity — and, critically, their Markov-chain reference values
that establish significance baselines, has only been formalised recently
[@vonwegner2017info; @vonwegner2024complexity; @vonwegner2026comparison].
These methods are currently scattered across individual publications.  An
earlier Python package implementing an initial subset of these methods was
published alongside the foundational information-theoretic analyses
[@vonwegner2018python]; `mstsa` extends and maintains that work with a
substantially expanded function set, C-accelerated implementations, and
Numba JIT compilation.

`mstsa` fills this gap.  It is the only Python library that (i) implements the
full set of information-theoretic complexity measures for symbolic sequences
alongside their first-order Markov-chain predictions, enabling direct
quantification of non-Markovian excess structure; (ii) provides the
Kullback-information likelihood-ratio tests for Markovianity, stationarity,
and transition-matrix symmetry [@kullback1962tests]; (iii) includes a
permutation-based multi-subject test for non-random microstate transition
syntax [@lehmann2005eeg]; and (iv) models serial dependence between
consecutive sojourn durations directly, via Autoregressive Conditional
Duration models [@engle1998autoregressive], rather than only characterising
each duration's marginal distribution in isolation.  The ability to compare
empirical measures to analytically derived Markov-chain benchmarks — rather than relying solely on
computationally intensive surrogate ensembles — is a distinctive feature of
the library [@vonwegner2026comparison].

The library is standalone: it does not require MNE-Python and operates on any
integer-valued symbolic sequence.  This makes it applicable beyond EEG to any
domain in which discrete time series arise, such as fMRI parcellation sequences,
animal-behaviour state data, or genomic symbol strings.  An earlier version
(0.2.0) has been used in a series of peer-reviewed publications characterising
microstate dynamics during resting state, propofol anaesthesia, and epilepsy
[@vonwegner2016fluctuation; @vonwegner2017info; @vonwegner2018invariant;
@vonwegner2024complexity; @vonwegner2026comparison; @hermann2024propofol].

# Acknowledgements
This work did not receive external funding.

# References

"""mstsa.acd — Autoregressive Conditional Duration (ACD) analysis of sojourn times.

Models serial dependence (clustering) between consecutive sojourn durations of
a symbolic sequence, following Engle & Russell (1998). Complements the static
sojourn-time distribution tools in :mod:`mstsa.stats`.
"""

import numpy as np

from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn

from typing import Tuple

from .mstsa import ScalarFloatArray, ScalarIntArray, sojourn_times_unordered


# -------------------------------------------------
# ACD model
# -------------------------------------------------

class ACD:
    """Autoregressive Conditional Duration model, ACD(p, q).

    .. math::
        \\psi_i = \\omega + \\sum_{j=1}^{q} \\alpha_j x_{i-j}
                          + \\sum_{k=1}^{p} \\beta_k \\psi_{i-k}

    Supported innovation distributions:
        ``'exponential'`` -- :math:`\\mathrm{Exp}(1)`, simplest, analytic log-likelihood.
        ``'weibull'``      -- Weibull with estimated shape, mean constrained to 1.
        ``'lognormal'``    -- Lognormal with estimated shape (sigma), mean constrained to 1.

    Parameters
    ----------
    p : int, optional
        Number of :math:`\\beta` (GARCH-type) lags (default 1).
    q : int, optional
        Number of :math:`\\alpha` (ARCH-type) lags (default 1).
    dist : {'exponential', 'weibull', 'lognormal'}, optional
        Innovation distribution (default ``'exponential'``).

    References
    ----------
    .. [1] Engle, R. F., & Russell, J. R. (1998). Autoregressive conditional
           duration: A new model for irregularly spaced transaction data.
           *Econometrica*, 66(5), 1127-1162.
    """

    def __init__(self, p: int = 1, q: int = 1, dist: str = "exponential"):
        if p < 1 or q < 1:
            raise ValueError("p and q must be >= 1")
        if dist not in ("exponential", "weibull", "lognormal"):
            raise ValueError("dist must be 'exponential', 'weibull', or 'lognormal'")
        self.p = p
        self.q = q
        self.dist = dist

        self.params_ = None
        self.psi_ = None
        self.residuals_ = None
        self.loglik_ = None
        self.converged_ = False

    def _unpack(self, params):
        omega = params[0]
        alpha = params[1 : 1 + self.q]
        beta = params[1 + self.q : 1 + self.q + self.p]
        shape = params[-1] if self.dist in ("weibull", "lognormal") else None
        return omega, alpha, beta, shape

    def _compute_psi(self, params, x):
        omega, alpha, beta, _ = self._unpack(params)
        n = len(x)
        m = max(self.p, self.q)
        psi = np.empty(n)

        mu0 = x.mean()
        psi[:m] = mu0

        for i in range(m, n):
            psi[i] = (
                omega
                + np.dot(alpha, x[i - self.q : i][::-1])
                + np.dot(beta, psi[i - self.p : i][::-1])
            )

        return psi

    def _neg_loglik(self, params, x):
        omega, alpha, beta, shape = self._unpack(params)

        if omega <= 0 or np.any(alpha < 0) or np.any(beta < 0):
            return 1e12
        if self.dist == "weibull" and (shape is None or shape <= 0):
            return 1e12

        psi = self._compute_psi(params, x)
        m = max(self.p, self.q)
        psi_e = psi[m:]
        x_e = x[m:]

        if np.any(psi_e <= 0):
            return 1e12

        if alpha.sum() + beta.sum() >= 1.0:
            return 1e12

        eps = x_e / psi_e

        if np.any(eps <= 0):
            return 1e12

        if self.dist == "exponential":
            ll = -np.sum(np.log(psi_e) + eps)

        elif self.dist == "weibull":
            a = shape
            c = gamma_fn(1.0 + 1.0 / a)
            log_f = (
                np.log(a) + np.log(c)
                + (a - 1) * np.log(eps)
                - (c * eps) ** a
            )
            ll = np.sum(log_f - np.log(psi_e))

        else:  # lognormal
            sigma = shape
            if sigma <= 0:
                return 1e12
            s2 = sigma ** 2
            log_x = np.log(x_e)
            log_psi = np.log(psi_e)
            ll = np.sum(
                -log_x
                - 0.5 * np.log(2 * np.pi * s2)
                - (log_x - log_psi + 0.5 * s2) ** 2 / (2 * s2)
            )

        return -ll

    def fit(self, x: ScalarFloatArray) -> "ACD":
        """Fit the ACD model by maximum likelihood.

        Parameters
        ----------
        x : array_like of float, shape (N,)
            Strictly positive sojourn durations.

        Returns
        -------
        self : ACD
            The fitted model (also stores ``params_``, ``psi_``,
            ``residuals_``, ``loglik_``, ``converged_``).
        """
        x = np.asarray(x, dtype=float)
        if np.any(x <= 0):
            raise ValueError("All durations must be strictly positive")

        mu0 = x.mean()
        a0 = 0.1 / self.q
        b0 = 0.7 / self.p
        alpha0 = np.full(self.q, a0)
        beta0 = np.full(self.p, b0)
        omega0 = mu0 * (1.0 - alpha0.sum() - beta0.sum())

        ub = 1.0 - 1e-6
        if self.dist == "exponential":
            x0 = np.concatenate([[omega0], alpha0, beta0])
            bounds = [(1e-8, None)] + [(1e-8, ub)] * (self.q + self.p)
        elif self.dist == "weibull":
            x0 = np.concatenate([[omega0], alpha0, beta0, [1.5]])
            bounds = (
                [(1e-8, None)]
                + [(1e-8, ub)] * (self.q + self.p)
                + [(0.1, 20.0)]
            )
        else:  # lognormal
            sigma0 = np.std(np.log(x + 1e-8)) * 0.8
            sigma0 = np.clip(sigma0, 0.1, 3.0)
            x0 = np.concatenate([[omega0], alpha0, beta0, [sigma0]])
            bounds = (
                [(1e-8, None)]
                + [(1e-8, ub)] * (self.q + self.p)
                + [(0.01, 5.0)]
            )

        result = minimize(
            self._neg_loglik,
            x0,
            args=(x,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        self.params_ = result.x
        self.converged_ = result.success
        self.loglik_ = -result.fun
        self.psi_ = self._compute_psi(result.x, x)
        self.residuals_ = x / self.psi_

        return self

    @property
    def omega(self) -> float:
        return self.params_[0]

    @property
    def alpha(self) -> ScalarFloatArray:
        return self.params_[1 : 1 + self.q]

    @property
    def beta(self) -> ScalarFloatArray:
        return self.params_[1 + self.q : 1 + self.q + self.p]

    @property
    def shape(self):
        return self.params_[-1] if self.dist in ("weibull", "lognormal") else None

    @property
    def unconditional_mean(self) -> float:
        """Theoretical :math:`E[x] = \\omega / (1 - \\sum\\alpha - \\sum\\beta)`."""
        denom = 1.0 - self.alpha.sum() - self.beta.sum()
        if denom <= 0:
            return np.inf
        return self.omega / denom


def fit_acd(durations: ScalarFloatArray, p: int = 1, q: int = 1,
            dist: str = "exponential") -> Tuple[ACD, ScalarFloatArray]:
    """Fit an ACD(p, q) model to a sequence of durations.

    Parameters
    ----------
    durations : array_like of float, shape (N,)
        Strictly positive sojourn durations.
    p, q : int, optional
        ACD order (default 1, 1).
    dist : {'exponential', 'weibull', 'lognormal'}, optional
        Innovation distribution (default ``'exponential'``).

    Returns
    -------
    model : ACD
        The fitted model.
    residuals : ndarray of float, shape (N,)
        Standardized residuals :math:`x_i / \\psi_i`.
    """
    model = ACD(p=p, q=q, dist=dist).fit(durations)
    return model, model.residuals_


def test_acd_residuals(residuals: ScalarFloatArray, lags: int = 10) -> dict:
    """Ljung-Box test for residual autocorrelation after fitting an ACD model.

    Parameters
    ----------
    residuals : array_like of float, shape (N,)
        Standardized ACD residuals (e.g. from :func:`fit_acd`).
    lags : int, optional
        Number of lags to test jointly (default 10).

    Returns
    -------
    result : dict
        ``{'lb_stat': float, 'p_value': float}``. A small ``p_value``
        indicates significant residual autocorrelation, i.e. the fitted
        ACD order is insufficient.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return {
        "lb_stat": float(lb["lb_stat"].iloc[0]),
        "p_value": float(lb["lb_pvalue"].iloc[0]),
    }


def select_acd_order(durations: ScalarFloatArray, p_max: int = 3, q_max: int = 4,
                      dist: str = "exponential", lb_lags: int = 10,
                      ic: str = "bic") -> dict:
    """Select the ACD order (p, q) that minimizes an information criterion.

    Grid-searches ``p in 1..p_max``, ``q in 1..q_max``, fits each ACD(p, q)
    model, and returns the winner together with a Ljung-Box residual test.

    Parameters
    ----------
    durations : array_like of float, shape (N,)
        Strictly positive sojourn durations.
    p_max, q_max : int, optional
        Maximum lag orders to search (default 3, 4).
    dist : {'exponential', 'weibull', 'lognormal'}, optional
        Innovation distribution (default ``'exponential'``).
    lb_lags : int, optional
        Number of lags for the Ljung-Box test on the winning model's
        residuals (default 10).
    ic : {'bic', 'aic'}, optional
        Information criterion used for order selection (default ``'bic'``).

    Returns
    -------
    result : dict
        Keys: ``p``, ``q``, ``ic_used``, ``omega``, ``alpha`` (list),
        ``beta`` (list), ``shape`` (if applicable), ``loglik``, ``AIC``,
        ``BIC``, ``lb_stat``, ``lb_pvalue``, ``autocorr_present``,
        ``converged``. Returns ``{'error': ...}`` if no candidate model
        converged to a finite log-likelihood.
    """
    x = np.asarray(durations, dtype=float)
    n = len(x)

    best_ic_val = np.inf
    best_result = None

    for p in range(1, p_max + 1):
        for q in range(1, q_max + 1):
            try:
                model, residuals = fit_acd(x, p=p, q=q, dist=dist)
                if not np.isfinite(model.loglik_):
                    continue
                k = 1 + p + q
                aic = -2 * model.loglik_ + 2 * k
                bic = -2 * model.loglik_ + k * np.log(n)
                ic_val = aic if ic == "aic" else bic
                if ic_val < best_ic_val:
                    best_ic_val = ic_val
                    best_result = (p, q, model, residuals, aic, bic)
            except Exception:
                continue

    if best_result is None:
        return {"error": "all models failed"}

    p_sel, q_sel, model, residuals, aic, bic = best_result
    lb = test_acd_residuals(residuals, lags=lb_lags)

    result = {
        "p": p_sel,
        "q": q_sel,
        "ic_used": ic,
        "omega": float(model.omega),
        "alpha": model.alpha.tolist(),
        "beta": model.beta.tolist(),
        "unconditional_mean": float(model.unconditional_mean),
        "loglik": float(model.loglik_),
        "AIC": aic,
        "BIC": bic,
        "lb_stat": lb["lb_stat"],
        "lb_pvalue": lb["p_value"],
        "autocorr_present": lb["p_value"] < 0.05,
        "converged": bool(model.converged_),
    }
    if dist in ("weibull", "lognormal"):
        result["shape"] = float(model.shape)

    return result


def simulate_acd(omega: float, alpha: ScalarFloatArray, beta: ScalarFloatArray,
                  n: int, dist: str = "exponential", shape: float = None,
                  rng: np.random.Generator = None, burn_in: int = 500) -> ScalarFloatArray:
    """Simulate durations from an ACD(p, q) process.

    Parameters
    ----------
    omega : float
        ACD intercept.
    alpha : array_like of float, shape (q,)
        ARCH-type coefficients (or a scalar for ACD(*, 1)).
    beta : array_like of float, shape (p,)
        GARCH-type coefficients (or a scalar for ACD(1, *)).
    n : int
        Number of durations to generate (after discarding burn-in).
    dist : {'exponential', 'weibull', 'lognormal'}, optional
        Innovation distribution, matching :class:`ACD` (default ``'exponential'``).
    shape : float, optional
        Shape parameter for ``'weibull'`` or ``'lognormal'`` innovations
        (required for those distributions).
    rng : numpy.random.Generator, optional
        Random generator; a fresh default generator is created if ``None``.
    burn_in : int, optional
        Number of initial samples discarded to remove start-up transients
        (default 500).

    Returns
    -------
    x : ndarray of float, shape (n,)
        Simulated durations.
    """
    if rng is None:
        rng = np.random.default_rng()
    if dist in ("weibull", "lognormal") and shape is None:
        raise ValueError("shape is required for 'weibull' and 'lognormal'")

    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    p, q = len(beta), len(alpha)
    m = max(p, q)

    total = n + burn_in
    psi = np.empty(total)
    x = np.empty(total)

    denom = 1.0 - alpha.sum() - beta.sum()
    psi0 = omega / denom if denom > 0 else omega
    psi[:m] = psi0

    def _draw(size):
        if dist == "exponential":
            return rng.exponential(1.0, size=size)
        elif dist == "weibull":
            c = gamma_fn(1.0 + 1.0 / shape)
            return rng.weibull(shape, size=size) / c
        else:  # lognormal
            s2 = shape ** 2
            return rng.lognormal(mean=-0.5 * s2, sigma=shape, size=size)

    x[:m] = _draw(m) * psi[:m]

    for i in range(m, total):
        psi[i] = omega + np.dot(alpha, x[i - q : i][::-1]) + np.dot(beta, psi[i - p : i][::-1])
        x[i] = _draw(1)[0] * psi[i]

    return x[burn_in:]


def acd_analysis(x: ScalarIntArray, K: int, p: int = 1, q: int = 1,
                  dist: str = "exponential", lb_lags: int = 10,
                  min_obs: int = 10) -> list:
    """End-to-end ACD(p, q) analysis of every symbol's sojourn durations.

    For each symbol: extracts sojourn durations (:func:`mstsa.sojourn_times_unordered`),
    fits an ACD(p, q) model (:func:`fit_acd`), and tests the residuals for
    remaining autocorrelation (:func:`test_acd_residuals`).

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    p, q : int, optional
        ACD order (default 1, 1).
    dist : {'exponential', 'weibull', 'lognormal'}, optional
        Innovation distribution (default ``'exponential'``).
    lb_lags : int, optional
        Number of lags for the Ljung-Box test (default 10).
    min_obs : int, optional
        Minimum number of sojourns required to attempt fitting a symbol
        (default 10).

    Returns
    -------
    results : list of dict, length K
        ``results[s]`` holds the ACD fit summary for symbol *s*, or
        ``{'error': ...}`` if too few sojourns were observed.
    """
    durations_per_symbol = sojourn_times_unordered(x, K)
    results = []

    for durations in durations_per_symbol:
        if len(durations) < min_obs:
            results.append({"error": f"Too few sojourns ({len(durations)} < {min_obs})"})
            continue

        try:
            model, residuals = fit_acd(durations, p=p, q=q, dist=dist)
            lb = test_acd_residuals(residuals, lags=lb_lags)

            result = {
                "n_sojourns": len(durations),
                "mean_duration": float(np.mean(durations)),
                "omega": float(model.omega),
                "alpha": model.alpha.tolist(),
                "beta": model.beta.tolist(),
                "unconditional_mean": float(model.unconditional_mean),
                "loglik": float(model.loglik_),
                "converged": bool(model.converged_),
                "lb_stat": lb["lb_stat"],
                "p_value": lb["p_value"],
                "autocorrelation_present": lb["p_value"] < 0.05,
            }
            if dist in ("weibull", "lognormal"):
                result["shape"] = float(model.shape)

        except Exception as e:
            result = {"error": str(e)}

        results.append(result)

    return results


def plot_duration_acf_pacf(x: ScalarIntArray, K: int, lags: int = 40,
                           min_obs: int = 10, alpha: float = 0.05,
                           savepath: str = None) -> None:
    """Plot ACF and PACF of each symbol's sojourn-duration series.

    Box-Jenkins-style diagnostic to identify the serial-dependence structure
    (ARCH/GARCH-type lags) before fitting an ACD model.

    Parameters
    ----------
    x : array_like of int, shape (N,)
        Symbolic sequence with integer labels in ``[0, K)``.
    K : int
        Number of distinct symbols.
    lags : int, optional
        Number of lags to display (default 40).
    min_obs : int, optional
        Skip symbols with fewer sojourns than this (default 10).
    alpha : float, optional
        Significance level for the confidence bands (default 0.05).
    savepath : str, optional
        If given, save the figure to this path instead of showing it.
    """
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    durations_per_symbol = sojourn_times_unordered(x, K)
    symbols = [s for s, d in enumerate(durations_per_symbol) if len(d) >= min_obs]
    n = len(symbols)

    if n == 0:
        print("No symbols with enough sojourns to plot.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(12, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, symbol in enumerate(symbols):
        y = np.asarray(durations_per_symbol[symbol], dtype=float)

        ax_acf = axes[row, 0]
        ax_pacf = axes[row, 1]

        plot_acf(y, lags=lags, alpha=alpha, ax=ax_acf,
                 title=f"ACF — symbol {symbol}  (n={len(y)})")
        plot_pacf(y, lags=lags, alpha=alpha, ax=ax_pacf,
                  title=f"PACF — symbol {symbol}", method="ywm")

        for ax in (ax_acf, ax_pacf):
            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")

    fig.suptitle("Sojourn duration series — serial correlation structure",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
        print(f"Saved to {savepath}")
    else:
        plt.show()

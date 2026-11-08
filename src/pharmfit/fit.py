"""Nonlinear least-squares fitting of concentration-response curves."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import optimize, stats

from .models import hill, operational, operational_ec50, operational_emax


@dataclass
class FitResult:
    """Result of a curve fit.

    ``params`` holds the fitted (and fixed) parameter values, ``stderr`` the standard
    errors of the fitted parameters (NaN for fixed ones or when there are no residual
    degrees of freedom) and ``ci95`` the corresponding t-based 95% confidence intervals.
    """

    model: str
    params: Dict[str, float]
    stderr: Dict[str, float]
    ci95: Dict[str, Tuple[float, float]]
    n_obs: int
    dof: int
    rss: float
    r2: float
    fixed: Dict[str, float] = field(default_factory=dict)

    @property
    def ec50(self) -> float:
        """EC50 in molar units (4PL: 10**logec50)."""
        if self.model == "4pl":
            return 10.0 ** self.params["logec50"]
        return float(
            operational_ec50(self.params["log_ka"], self.params["log_tau"], self.fixed.get("n", 1.0))
        )

    @property
    def span(self) -> float:
        """Top minus bottom (4PL only)."""
        return self.params["top"] - self.params["bottom"]


def _prepare(conc, resp, drop_nonpositive: bool = True):
    conc = np.asarray(conc, dtype=float)
    resp = np.asarray(resp, dtype=float)
    if conc.shape != resp.shape:
        raise ValueError("conc and resp must have the same length")
    mask = np.isfinite(conc) & np.isfinite(resp)
    if drop_nonpositive:
        mask &= conc > 0
    if mask.sum() == 0:
        raise ValueError("no valid data points")
    return np.log10(conc[mask]), resp[mask]


def _mean_by_x(x, y):
    ux = np.unique(x)
    my = np.array([y[x == u].mean() for u in ux])
    return ux, my


def _p0_4pl(x, y) -> Dict[str, float]:
    ux, my = _mean_by_x(x, y)
    bottom, top = float(my[0]), float(my[-1])
    if bottom == top:
        top = bottom + 1.0
    mid = (bottom + top) / 2.0
    # first crossing of the half-way response, linearly interpolated
    logec50 = float(np.median(ux))
    for i in range(len(ux) - 1):
        lo, hi = my[i], my[i + 1]
        if (lo - mid) * (hi - mid) <= 0 and lo != hi:
            logec50 = float(ux[i] + (mid - lo) / (hi - lo) * (ux[i + 1] - ux[i]))
            break
    return {"bottom": bottom, "top": top, "logec50": logec50, "slope": 1.0}


def _summarise(model, names, popt, pcov, x, y, predict, fixed):
    resid = y - predict
    rss = float(np.sum(resid**2))
    dof = len(y) - len(names)
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    if dof > 0 and np.all(np.isfinite(pcov)):
        se = np.sqrt(np.diag(pcov))
        tcrit = stats.t.ppf(0.975, dof)
    else:
        se = np.full(len(names), np.nan)
        tcrit = np.nan
    params = dict(fixed)
    params.update(dict(zip(names, map(float, popt))))
    stderr = {k: float("nan") for k in fixed}
    stderr.update(dict(zip(names, map(float, se))))
    ci = {k: (v, v) for k, v in fixed.items()}
    ci.update({n: (float(p - tcrit * s), float(p + tcrit * s)) for n, p, s in zip(names, popt, se)})
    return FitResult(model, params, stderr, ci, len(y), dof, rss, r2, dict(fixed))


def fit_4pl(
    conc,
    resp,
    *,
    fix_bottom: Optional[float] = None,
    fix_top: Optional[float] = None,
    fix_slope: Optional[float] = None,
    drop_nonpositive: bool = True,
) -> FitResult:
    """Fit a four-parameter logistic curve to molar concentrations and responses.

    Parameters can be held fixed (for example ``fix_bottom=0`` for normalised data or
    ``fix_slope=1`` for a mass-action-like curve). Raises ``ValueError`` if there are
    fewer observations than free parameters and ``RuntimeError`` if the fit fails.
    """
    x, y = _prepare(conc, resp, drop_nonpositive)
    all_names = ["bottom", "top", "logec50", "slope"]
    fixed = {
        k: v
        for k, v in zip(all_names, [fix_bottom, fix_top, None, fix_slope])
        if v is not None
    }
    free = [k for k in all_names if k not in fixed]
    if len(y) < len(free):
        raise ValueError(f"need at least {len(free)} points to fit {free}")

    p0_all = _p0_4pl(x, y)
    lower = {"bottom": -np.inf, "top": -np.inf, "logec50": x.min() - 2.0, "slope": 0.2}
    upper = {"bottom": np.inf, "top": np.inf, "logec50": x.max() + 2.0, "slope": 4.0}
    p0 = [float(np.clip(p0_all[k], lower[k] + 1e-9, upper[k] - 1e-9)) for k in free]

    def model(xx, *vals):
        p = dict(fixed)
        p.update(zip(free, vals))
        return hill(xx, p["bottom"], p["top"], p["logec50"], p["slope"])

    try:
        popt, pcov = optimize.curve_fit(
            model,
            x,
            y,
            p0=p0,
            bounds=([lower[k] for k in free], [upper[k] for k in free]),
            maxfev=20000,
        )
    except (RuntimeError, optimize.OptimizeWarning) as exc:  # pragma: no cover - rare
        raise RuntimeError(f"4PL fit did not converge: {exc}") from exc
    return _summarise("4pl", free, popt, pcov, x, y, model(x, *popt), fixed)


def fit_operational(
    conc, resp, *, em: float = 1.0, n: float = 1.0, drop_nonpositive: bool = True
) -> FitResult:
    """Fit the Black & Leff operational model for ``log_ka`` and ``log_tau``.

    ``em`` (maximal system response, in the units of ``resp``) and the transducer slope
    ``n`` are held fixed, because they cannot be estimated from a single partial-agonist
    curve. Use responses normalised to the system maximum (for example a full agonist or
    forskolin) together with ``em=1``.
    """
    x, y = _prepare(conc, resp, drop_nonpositive)
    names = ["log_ka", "log_tau"]
    if len(y) < len(names):
        raise ValueError("need at least 2 points")

    def model(xx, log_ka, log_tau):
        return operational(10.0**xx, log_ka, log_tau, em=em, n=n)

    p0 = [float(np.median(x)), 0.0]
    popt, pcov = optimize.curve_fit(
        model,
        x,
        y,
        p0=p0,
        bounds=([x.min() - 4.0, -3.0], [x.max() + 4.0, 3.0]),
        maxfev=20000,
    )
    return _summarise("operational", names, popt, pcov, x, y, model(x, *popt), {"em": em, "n": n})


def operational_summary(result: FitResult) -> Dict[str, float]:
    """Derived quantities (tau, KA, Emax fraction, EC50) from an operational fit."""
    if result.model != "operational":
        raise ValueError("expected an operational-model fit")
    em, n = result.fixed.get("em", 1.0), result.fixed.get("n", 1.0)
    log_ka, log_tau = result.params["log_ka"], result.params["log_tau"]
    return {
        "tau": 10.0**log_tau,
        "ka": 10.0**log_ka,
        "emax": float(operational_emax(log_tau, em, n)),
        "ec50": float(operational_ec50(log_ka, log_tau, n)),
    }


def relative_efficacy(test: FitResult, reference: FitResult) -> Dict[str, float]:
    """Relative maximal effect (alpha) and concentration ratio (CR) of two 4PL fits.

    alpha = (top - bottom) of the test ligand divided by that of the reference;
    CR = EC50(test) / EC50(reference). With N/OFQ as the reference and normalised data
    this reproduces the alpha and CR columns reported in NOP receptor studies.
    """
    if test.model != "4pl" or reference.model != "4pl":
        raise ValueError("relative_efficacy needs 4PL fits")
    return {"alpha": test.span / reference.span, "cr": test.ec50 / reference.ec50}


def bootstrap_relative(
    conc_test,
    resp_test,
    conc_ref,
    resp_ref,
    *,
    n_boot: int = 500,
    seed: Optional[int] = 0,
    **fit_kwargs,
) -> Dict[str, Tuple[float, float]]:
    """Residual bootstrap 95% intervals for alpha and CR.

    Residuals of each fitted curve are resampled (rescaled for degrees of freedom),
    added to the fitted values and the curves are refitted. Failed refits are skipped.
    Returns percentile intervals; raises ``RuntimeError`` if fewer than 20 refits work.
    """
    rng = np.random.default_rng(seed)
    curves = []
    for conc, resp in ((conc_test, resp_test), (conc_ref, resp_ref)):
        x, y = _prepare(conc, resp)
        fit = fit_4pl(10.0**x, y, **fit_kwargs)
        pred = hill(x, fit.params["bottom"], fit.params["top"], fit.params["logec50"], fit.params["slope"])
        resid = y - pred
        scale = np.sqrt(len(y) / max(fit.dof, 1))
        curves.append((x, pred, resid * scale))

    alphas, crs = [], []
    for _ in range(n_boot):
        fits = []
        try:
            for x, pred, resid in curves:
                y_b = pred + rng.choice(resid, size=len(resid), replace=True)
                fits.append(fit_4pl(10.0**x, y_b, **fit_kwargs))
        except (RuntimeError, ValueError):
            continue
        rel = relative_efficacy(fits[0], fits[1])
        if np.isfinite(rel["alpha"]) and np.isfinite(rel["cr"]):
            alphas.append(rel["alpha"])
            crs.append(rel["cr"])
    if len(alphas) < 20:
        raise RuntimeError("too few successful bootstrap refits")
    return {
        "alpha": (float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))),
        "cr": (float(np.percentile(crs, 2.5)), float(np.percentile(crs, 97.5))),
    }

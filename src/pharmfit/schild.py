"""Competitive-antagonism analysis (Gaddum / Schild)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy import stats


def concentration_ratio(logec50_with_antagonist: float, logec50_control: float) -> float:
    """Dose ratio: EC50 in the presence of antagonist divided by control EC50."""
    return float(10.0 ** (logec50_with_antagonist - logec50_control))


def pa2_single(cr: float, antagonist_conc: float) -> float:
    """pA2 from one antagonist concentration, assuming a Schild slope of 1.

    pA2 = log10(CR - 1) - log10([B]) with [B] in molar. Requires CR > 1, i.e. a
    rightward shift of the agonist curve.
    """
    if cr <= 1.0:
        raise ValueError("concentration ratio must be > 1 for a rightward shift")
    if antagonist_conc <= 0:
        raise ValueError("antagonist concentration must be positive")
    return float(np.log10(cr - 1.0) - np.log10(antagonist_conc))


@dataclass
class SchildResult:
    """Result of a Schild regression."""

    slope: float
    slope_ci95: Tuple[float, float]
    pa2_free: float
    pa2_constrained: float
    pa2_constrained_ci95: Tuple[float, float]
    slope_differs_from_unity: bool
    n: int


def schild_regression(antagonist_concs: Sequence[float], crs: Sequence[float]) -> SchildResult:
    """Schild regression of log10(CR - 1) on log10([B]).

    Reports the free-slope pA2 (intercept / slope), the pA2 obtained with the slope
    constrained to 1 (mean of log10(CR - 1) - log10[B]) with a t-based 95% interval, and
    whether the free slope differs from 1 at the 5% level. Needs at least 3 antagonist
    concentrations with CR > 1.
    """
    b = np.asarray(antagonist_concs, dtype=float)
    cr = np.asarray(crs, dtype=float)
    keep = (b > 0) & (cr > 1.0)
    b, cr = b[keep], cr[keep]
    if len(b) < 3:
        raise ValueError("need at least 3 antagonist concentrations with CR > 1")
    x = np.log10(b)
    y = np.log10(cr - 1.0)
    reg = stats.linregress(x, y)
    dof = len(x) - 2
    tcrit = stats.t.ppf(0.975, dof)
    slope_ci = (reg.slope - tcrit * reg.stderr, reg.slope + tcrit * reg.stderr)
    pa2_free = reg.intercept / reg.slope
    per_point = y - x
    mean = per_point.mean()
    sem = per_point.std(ddof=1) / np.sqrt(len(per_point))
    tcrit_c = stats.t.ppf(0.975, len(per_point) - 1)
    return SchildResult(
        slope=float(reg.slope),
        slope_ci95=(float(slope_ci[0]), float(slope_ci[1])),
        pa2_free=float(pa2_free),
        pa2_constrained=float(mean),
        pa2_constrained_ci95=(float(mean - tcrit_c * sem), float(mean + tcrit_c * sem)),
        slope_differs_from_unity=bool(not (slope_ci[0] <= 1.0 <= slope_ci[1])),
        n=int(len(x)),
    )

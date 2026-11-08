import numpy as np
import pytest

from pharmfit import (
    bootstrap_relative,
    fit_4pl,
    fit_operational,
    hill,
    operational,
    operational_summary,
    relative_efficacy,
)
from pharmfit.simulate import default_concentrations

CONCS = default_concentrations()
LOGC = np.log10(CONCS)


def _noisy(fn, sd, reps, seed):
    rng = np.random.default_rng(seed)
    conc = np.tile(CONCS, reps)
    y = np.tile(fn(), reps) + rng.normal(0, sd, size=len(conc))
    return conc, y


def test_fit_4pl_recovers_parameters():
    conc, y = _noisy(lambda: hill(LOGC, 0.0, 1.0, -8.0, 1.0), 0.02, 5, seed=1)
    fit = fit_4pl(conc, y)
    assert fit.params["logec50"] == pytest.approx(-8.0, abs=0.1)
    assert fit.params["top"] == pytest.approx(1.0, abs=0.05)
    assert fit.params["bottom"] == pytest.approx(0.0, abs=0.05)
    assert fit.params["slope"] == pytest.approx(1.0, abs=0.2)
    assert fit.r2 > 0.99
    lo, hi = fit.ci95["logec50"]
    assert lo < fit.params["logec50"] < hi
    # the estimate sits within a few standard errors of the truth
    assert abs(fit.params["logec50"] + 8.0) < 4 * fit.stderr["logec50"]


def test_ci95_coverage_over_many_simulated_datasets():
    """About 95% of intervals should contain the true logEC50 (checked loosely)."""
    hits = 0
    runs = 60
    for seed in range(runs):
        conc, y = _noisy(lambda: hill(LOGC, 0.0, 1.0, -8.0, 1.0), 0.05, 3, seed=100 + seed)
        lo, hi = fit_4pl(conc, y).ci95["logec50"]
        hits += lo <= -8.0 <= hi
    assert 0.85 <= hits / runs <= 1.0


def test_fit_4pl_inhibition_curve():
    conc, y = _noisy(lambda: hill(LOGC, 1.0, 0.1, -7.0, 1.2), 0.02, 4, seed=2)
    fit = fit_4pl(conc, y)
    assert fit.params["top"] < fit.params["bottom"]
    assert fit.params["logec50"] == pytest.approx(-7.0, abs=0.15)


def test_fit_4pl_with_fixed_parameters():
    conc, y = _noisy(lambda: hill(LOGC, 0.0, 1.0, -8.0, 1.0), 0.02, 3, seed=3)
    fit = fit_4pl(conc, y, fix_bottom=0.0, fix_slope=1.0)
    assert fit.params["bottom"] == 0.0
    assert fit.params["slope"] == 1.0
    assert np.isnan(fit.stderr["bottom"])
    assert fit.dof == len(y) - 2


def test_fit_4pl_ignores_zero_concentrations():
    conc, y = _noisy(lambda: hill(LOGC, 0.0, 1.0, -8.0, 1.0), 0.02, 3, seed=4)
    conc = np.append(conc, 0.0)
    y = np.append(y, 0.0)
    fit = fit_4pl(conc, y)
    assert fit.n_obs == len(y) - 1


def test_fit_4pl_needs_enough_points():
    with pytest.raises(ValueError):
        fit_4pl([1e-8, 1e-7], [0.1, 0.5])


def test_relative_efficacy_matches_construction():
    c1, y1 = _noisy(lambda: hill(LOGC, 0.0, 1.0, -9.0, 1.0), 0.015, 5, seed=5)
    c2, y2 = _noisy(lambda: hill(LOGC, 0.0, 0.4, -7.0, 1.0), 0.015, 5, seed=6)
    rel = relative_efficacy(fit_4pl(c2, y2), fit_4pl(c1, y1))
    assert rel["alpha"] == pytest.approx(0.4, abs=0.05)
    assert rel["cr"] == pytest.approx(100.0, rel=0.3)


def test_bootstrap_interval_contains_truth():
    c1, y1 = _noisy(lambda: hill(LOGC, 0.0, 1.0, -9.0, 1.0), 0.02, 5, seed=7)
    c2, y2 = _noisy(lambda: hill(LOGC, 0.0, 0.4, -7.0, 1.0), 0.02, 5, seed=8)
    ci = bootstrap_relative(c2, y2, c1, y1, n_boot=150, seed=0)
    assert ci["alpha"][0] < 0.4 < ci["alpha"][1]
    assert ci["cr"][0] < 100.0 < ci["cr"][1]


def test_operational_fit_recovers_ka_and_tau():
    truth_ka, truth_tau = -6.5, np.log10(2.0)
    conc, y = _noisy(lambda: operational(CONCS, truth_ka, truth_tau), 0.01, 6, seed=9)
    fit = fit_operational(conc, y, em=1.0, n=1.0)
    assert fit.params["log_ka"] == pytest.approx(truth_ka, abs=0.2)
    assert fit.params["log_tau"] == pytest.approx(truth_tau, abs=0.15)
    summary = operational_summary(fit)
    assert summary["emax"] == pytest.approx(2.0 / 3.0, abs=0.05)
    assert summary["ec50"] == pytest.approx(10.0**truth_ka / 3.0, rel=0.3)


def test_operational_summary_rejects_other_models():
    conc, y = _noisy(lambda: hill(LOGC, 0.0, 1.0, -8.0, 1.0), 0.02, 3, seed=10)
    with pytest.raises(ValueError):
        operational_summary(fit_4pl(conc, y))

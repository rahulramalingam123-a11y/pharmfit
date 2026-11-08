import numpy as np
import pytest

from pharmfit import hill, operational, operational_ec50, operational_emax


def test_hill_limits_and_midpoint():
    logc = np.array([-20.0, -8.0, 20.0])
    y = hill(logc, bottom=0.1, top=0.9, logec50=-8.0, slope=1.0)
    assert y[0] == pytest.approx(0.1, abs=1e-9)
    assert y[1] == pytest.approx(0.5, abs=1e-9)
    assert y[2] == pytest.approx(0.9, abs=1e-9)


def test_hill_inhibition_curve_has_top_below_bottom():
    y = hill(np.array([-12.0, -4.0]), bottom=1.0, top=0.2, logec50=-8.0, slope=1.0)
    assert y[0] > y[1]


@pytest.mark.parametrize("log_tau", [-1.0, 0.0, 0.5, 2.0])
@pytest.mark.parametrize("n", [0.8, 1.0, 1.5])
def test_operational_emax_matches_high_concentration_limit(log_tau, n):
    y = operational(1e12, log_ka=-6.0, log_tau=log_tau, em=1.0, n=n)
    assert y == pytest.approx(operational_emax(log_tau, 1.0, n), rel=1e-6)


@pytest.mark.parametrize("log_tau", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("n", [0.8, 1.0, 1.5])
def test_operational_response_at_ec50_is_half_of_emax(log_tau, n):
    log_ka = -7.0
    ec50 = operational_ec50(log_ka, log_tau, n)
    y = operational(ec50, log_ka, log_tau, em=1.0, n=n)
    assert y == pytest.approx(operational_emax(log_tau, 1.0, n) / 2.0, rel=1e-9)


def test_operational_ec50_reduces_to_ka_over_one_plus_tau_for_n_equal_one():
    assert operational_ec50(-7.0, np.log10(4.0), 1.0) == pytest.approx(1e-7 / 5.0)


def test_higher_tau_gives_higher_emax():
    assert operational_emax(0.5) > operational_emax(-0.5)

import numpy as np
import pytest

from pharmfit import concentration_ratio, pa2_single, schild_regression


def test_concentration_ratio():
    assert concentration_ratio(-7.0, -8.0) == pytest.approx(10.0)


def test_pa2_single_known_value():
    # KB = 1 nM -> CR = 1 + [B]/KB; for [B] = 10 nM, CR = 11 and pA2 = 9
    assert pa2_single(11.0, 10e-9) == pytest.approx(9.0)


@pytest.mark.parametrize("cr, b", [(1.0, 1e-8), (0.5, 1e-8), (5.0, 0.0)])
def test_pa2_single_rejects_invalid_input(cr, b):
    with pytest.raises(ValueError):
        pa2_single(cr, b)


def test_schild_regression_recovers_unit_slope_and_pa2():
    kb = 10 ** -8.5
    b = np.array([1e-8, 3e-8, 1e-7, 3e-7, 1e-6])
    cr = 1.0 + b / kb
    res = schild_regression(b, cr)
    assert res.slope == pytest.approx(1.0, abs=1e-6)
    assert res.pa2_free == pytest.approx(8.5, abs=1e-6)
    assert res.pa2_constrained == pytest.approx(8.5, abs=1e-6)
    assert not res.slope_differs_from_unity


def test_schild_regression_flags_non_unit_slope():
    b = np.array([1e-8, 3e-8, 1e-7, 3e-7, 1e-6])
    cr = 1.0 + (b / 1e-8) ** 0.5  # slope 0.5 on the Schild plot
    rng = np.random.default_rng(0)
    cr = cr * np.exp(rng.normal(0, 0.01, size=len(cr)))
    res = schild_regression(b, cr)
    assert res.slope == pytest.approx(0.5, abs=0.1)
    assert res.slope_differs_from_unity


def test_schild_regression_needs_three_valid_points():
    with pytest.raises(ValueError):
        schild_regression([1e-8, 1e-7], [3.0, 12.0])

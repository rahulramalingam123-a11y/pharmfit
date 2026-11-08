"""Mathematical models for concentration-response data.

All concentrations are molar and all logarithms are base 10.
"""
from __future__ import annotations

import numpy as np


def hill(logc, bottom, top, logec50, slope):
    """Four-parameter logistic (Hill) curve in log10 concentration.

    ``slope`` must be positive. Inhibition curves are described by ``top < bottom``.
    """
    logc = np.asarray(logc, dtype=float)
    return bottom + (top - bottom) / (1.0 + 10.0 ** ((logec50 - logc) * slope))


def operational(conc, log_ka, log_tau, em=1.0, n=1.0):
    """Black & Leff operational model of agonism.

    E = Em * tau^n * [A]^n / ( ([A] + KA)^n + tau^n * [A]^n )

    ``log_ka`` is log10 of the agonist dissociation constant, ``log_tau`` is log10 of
    the transducer ratio tau (receptor density divided by the transducer constant,
    i.e. efficacy in a given system), ``em`` is the maximal system response and ``n``
    is the slope of the transducer function.
    """
    a = np.asarray(conc, dtype=float)
    ka = 10.0**log_ka
    tau = 10.0**log_tau
    num = em * tau**n * a**n
    den = (a + ka) ** n + tau**n * a**n
    return num / den


def operational_emax(log_tau, em=1.0, n=1.0):
    """Maximal response of an agonist under the operational model."""
    tau_n = (10.0**log_tau) ** n
    return em * tau_n / (1.0 + tau_n)


def operational_ec50(log_ka, log_tau, n=1.0):
    """EC50 (molar) of an agonist under the operational model.

    Setting E = Emax / 2 gives EC50 = KA / ((2 + tau^n)^(1/n) - 1); for n = 1 this
    reduces to the familiar KA / (1 + tau).
    """
    ka = 10.0**log_ka
    tau_n = (10.0**log_tau) ** n
    return ka / ((2.0 + tau_n) ** (1.0 / n) - 1.0)

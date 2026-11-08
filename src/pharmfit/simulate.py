"""Simulated assay data for tests, demos and teaching.

The data are generated from the Black & Leff operational model. They are simulated, not
experimental, and are meant to show how one ligand can look like a weak partial agonist
in a system with little signal amplification and a near-full agonist in a system with a
lot of it.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import operational

#: (log10 KA, log10 tau) in a reference system with a coupling factor of 1.
DEMO_LIGANDS: Dict[str, Tuple[float, float]] = {
    "FullAgonist": (-8.0, np.log10(50.0)),
    "PartialAgonist": (-6.5, np.log10(0.4)),
}

#: Multiplier on tau, standing in for receptor reserve / amplification of each assay.
DEMO_ASSAYS: Dict[str, float] = {
    "low amplification": 1.0,
    "medium amplification": 3.0,
    "high amplification": 10.0,
}


def default_concentrations(low_log: float = -12.0, high_log: float = -4.0, step: float = 0.5):
    """Half-log concentration series in molar units."""
    return 10.0 ** np.arange(low_log, high_log + step / 2, step)


def simulate_assay(
    ligands: Dict[str, Tuple[float, float]],
    coupling: float,
    concs: Sequence[float],
    *,
    em: float = 1.0,
    n: float = 1.0,
    sd: float = 0.04,
    n_experiments: int = 5,
    rng: Optional[np.random.Generator] = None,
    assay: str = "assay",
) -> pd.DataFrame:
    """Simulate replicate experiments for several ligands in one assay.

    ``coupling`` multiplies each ligand's tau (more amplification means a higher tau and
    therefore a higher apparent efficacy). Gaussian noise with standard deviation
    ``sd * em`` is added independently to every point.
    """
    rng = rng or np.random.default_rng()
    concs = np.asarray(concs, dtype=float)
    rows = []
    for ligand, (log_ka, log_tau) in ligands.items():
        mean = operational(concs, log_ka, log_tau + np.log10(coupling), em=em, n=n)
        for exp in range(1, n_experiments + 1):
            noisy = mean + rng.normal(0.0, sd * em, size=len(concs))
            for c, r in zip(concs, noisy):
                rows.append((assay, ligand, exp, c, r))
    return pd.DataFrame(rows, columns=["assay", "ligand", "experiment", "conc", "response"])


def make_demo_dataset(seed: int = 7, n_experiments: int = 5, sd: float = 0.04) -> pd.DataFrame:
    """Three assays of increasing amplification, each with a full and a partial agonist."""
    rng = np.random.default_rng(seed)
    concs = default_concentrations()
    parts = [
        simulate_assay(
            DEMO_LIGANDS,
            factor,
            concs,
            sd=sd,
            n_experiments=n_experiments,
            rng=rng,
            assay=name,
        )
        for name, factor in DEMO_ASSAYS.items()
    ]
    return pd.concat(parts, ignore_index=True)

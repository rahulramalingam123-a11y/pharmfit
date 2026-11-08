"""Batch analysis: fit every ligand in every assay and compare with a reference."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .fit import bootstrap_relative, fit_4pl, relative_efficacy


def fit_table(
    df: pd.DataFrame,
    *,
    ligand: str = "ligand",
    conc: str = "conc",
    response: str = "response",
    assay: Optional[str] = None,
    reference: Optional[str] = None,
    n_boot: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit a 4PL curve to each ligand (within each assay) and tabulate the results.

    One row per assay and ligand with bottom, top, logEC50 (and standard error and 95%
    interval), Hill slope, R-squared and the number of points. When ``reference`` names
    a ligand, each ligand also gets alpha (relative maximal effect) and CR (concentration
    ratio of EC50 values) against that reference in the same assay; ``n_boot > 0`` adds
    residual-bootstrap 95% intervals for both. Ligands whose fit fails get NaN values.
    """
    for col in (ligand, conc, response) + ((assay,) if assay else ()):
        if col not in df.columns:
            raise KeyError(f"column '{col}' not found; available: {list(df.columns)}")

    groups = [("all", df)] if assay is None else list(df.groupby(assay, sort=False))
    rows = []
    for assay_name, sub in groups:
        fits, data = {}, {}
        for lig, s in sub.groupby(ligand, sort=False):
            data[lig] = (s[conc].to_numpy(), s[response].to_numpy())
            row = {"assay": assay_name, "ligand": lig, "n_obs": len(s)}
            try:
                fit = fit_4pl(*data[lig])
            except (RuntimeError, ValueError):
                rows.append(row)
                continue
            fits[lig] = fit
            lo, hi = fit.ci95["logec50"]
            row.update(
                bottom=fit.params["bottom"],
                top=fit.params["top"],
                logec50=fit.params["logec50"],
                se_logec50=fit.stderr["logec50"],
                logec50_low=lo,
                logec50_high=hi,
                slope=fit.params["slope"],
                r2=fit.r2,
            )
            rows.append(row)
        if reference is None:
            continue
        if reference not in fits:
            raise ValueError(f"reference ligand '{reference}' could not be fitted in assay '{assay_name}'")
        ref_fit = fits[reference]
        for row in rows:
            if row["assay"] != assay_name or row["ligand"] not in fits:
                continue
            lig = row["ligand"]
            rel = relative_efficacy(fits[lig], ref_fit)
            row["alpha"], row["cr"] = rel["alpha"], rel["cr"]
            if n_boot > 0 and lig != reference:
                try:
                    ci = bootstrap_relative(*data[lig], *data[reference], n_boot=n_boot, seed=seed)
                    row["alpha_low"], row["alpha_high"] = ci["alpha"]
                    row["cr_low"], row["cr_high"] = ci["cr"]
                except RuntimeError:
                    pass

    out = pd.DataFrame(rows)
    if "logec50" in out:
        out["ec50_nM"] = 10.0 ** out["logec50"] * 1e9
    return out

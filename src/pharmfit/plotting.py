"""Plots of concentration-response data with fitted curves."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import hill


def plot_curves(
    df: pd.DataFrame,
    table: pd.DataFrame,
    *,
    ligand: str = "ligand",
    conc: str = "conc",
    response: str = "response",
    assay: Optional[str] = None,
    outfile: Optional[str] = None,
    title: Optional[str] = None,
):
    """One panel per assay: mean +/- SD of the data and the fitted 4PL curve per ligand.

    ``table`` is the output of :func:`pharmfit.analysis.fit_table`.
    """
    assays = ["all"] if assay is None else list(dict.fromkeys(df[assay]))
    fig, axes = plt.subplots(1, len(assays), figsize=(4.2 * len(assays), 3.8), sharey=True, squeeze=False)
    ligands = list(dict.fromkeys(df[ligand]))
    colors = plt.get_cmap("tab10")
    for ax, a in zip(axes[0], assays):
        sub = df if assay is None else df[df[assay] == a]
        for i, lig in enumerate(ligands):
            s = sub[sub[ligand] == lig]
            if s.empty:
                continue
            g = s.groupby(conc)[response].agg(["mean", "std"]).reset_index()
            x = np.log10(g[conc].to_numpy())
            ax.errorbar(x, g["mean"], yerr=g["std"].fillna(0), fmt="o", ms=4, color=colors(i), capsize=2)
            row = table[(table["assay"] == a) & (table["ligand"] == lig)]
            if row.empty or pd.isna(row.iloc[0].get("top")):
                continue
            p = row.iloc[0]
            xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 200)
            label = lig
            if "alpha" in p and pd.notna(p["alpha"]):
                label = f"{lig} (alpha {p['alpha']:.2f})"
            ax.plot(xs, hill(xs, p["bottom"], p["top"], p["logec50"], p["slope"]), color=colors(i), label=label)
        ax.set_title(a if assay is not None else (title or ""), fontsize=10)
        ax.set_xlabel("log10 [ligand] (M)")
        ax.legend(fontsize=8, frameon=False)
    axes[0][0].set_ylabel(response)
    if title and assay is not None:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=160)
    return fig

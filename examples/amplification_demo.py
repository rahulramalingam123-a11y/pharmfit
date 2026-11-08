"""Same ligand, three assays: apparent efficacy depends on signal amplification.

Run from the repository root:

    python examples/amplification_demo.py

All data here are SIMULATED from the Black & Leff operational model. A partial agonist
(tau = 0.4 in the least amplified system) is compared with a full agonist (tau = 50) in
assays whose amplification is 1x, 3x and 10x. The 4PL fit reports the partial agonist's
alpha (relative maximal effect) rising from about 0.3 to about 0.8 although its
intrinsic properties never change.
"""
from pathlib import Path

import pandas as pd

from pharmfit import fit_table, make_demo_dataset
from pharmfit.plotting import plot_curves

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "examples" / "data").mkdir(parents=True, exist_ok=True)
(ROOT / "docs").mkdir(exist_ok=True)

df = make_demo_dataset(seed=7)
df.to_csv(ROOT / "examples" / "data" / "simulated_assays.csv", index=False)

table = fit_table(df, assay="assay", reference="FullAgonist", n_boot=300, seed=0)
table.to_csv(ROOT / "docs" / "amplification_results.csv", index=False)

cols = ["assay", "ligand", "logec50", "top", "alpha", "alpha_low", "alpha_high", "cr"]
with pd.option_context("display.width", 140, "display.max_columns", 20):
    print(table[cols].round(2).to_string(index=False))

plot_curves(
    df,
    table,
    assay="assay",
    outfile=str(ROOT / "docs" / "amplification_demo.png"),
    title="Simulated data: one partial agonist, three levels of signal amplification",
)
print("saved docs/amplification_demo.png")

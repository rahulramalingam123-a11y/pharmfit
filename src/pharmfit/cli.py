"""Command-line interface: ``pharmfit fit data.csv --reference N/OFQ``."""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import matplotlib
import pandas as pd

from .analysis import fit_table
from .plotting import plot_curves
from .simulate import make_demo_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pharmfit",
        description="Fit concentration-response curves and compare ligands with a reference.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit 4PL curves from a CSV file")
    fit.add_argument("csv", help="CSV with one row per measurement")
    fit.add_argument("--ligand", default="ligand", help="ligand column (default: ligand)")
    fit.add_argument("--conc", default="conc", help="molar concentration column (default: conc)")
    fit.add_argument("--response", default="response", help="response column (default: response)")
    fit.add_argument("--assay", default=None, help="optional column that splits the data into assays")
    fit.add_argument("--reference", default=None, help="reference ligand for alpha and CR")
    fit.add_argument("--bootstrap", type=int, default=0, metavar="N", help="residual bootstrap replicates for alpha and CR")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--out", default=None, help="write the results table to this CSV")
    fit.add_argument("--plot", default=None, help="save a figure of the data and fits (PNG)")

    demo = sub.add_parser("demo", help="write a simulated 3-assay dataset to CSV")
    demo.add_argument("out", help="output CSV path")
    demo.add_argument("--seed", type=int, default=7)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        make_demo_dataset(seed=args.seed).to_csv(args.out, index=False)
        print(f"wrote {args.out}")
        return 0

    df = pd.read_csv(args.csv)
    try:
        table = fit_table(
            df,
            ligand=args.ligand,
            conc=args.conc,
            response=args.response,
            assay=args.assay,
            reference=args.reference,
            n_boot=args.bootstrap,
            seed=args.seed,
        )
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    shown = table.round(3)
    print(shown.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
    if args.plot:
        matplotlib.use("Agg")
        plot_curves(
            df,
            table,
            ligand=args.ligand,
            conc=args.conc,
            response=args.response,
            assay=args.assay,
            outfile=args.plot,
        )
        print(f"saved {args.plot}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

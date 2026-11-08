import pandas as pd
import pytest

from pharmfit import fit_table, make_demo_dataset
from pharmfit.cli import main


@pytest.fixture(scope="module")
def demo():
    return make_demo_dataset(seed=7)


def test_alpha_increases_with_amplification(demo):
    table = fit_table(demo, assay="assay", reference="FullAgonist")
    partial = table[table["ligand"] == "PartialAgonist"].set_index("assay")
    alphas = [partial.loc[a, "alpha"] for a in ["low amplification", "medium amplification", "high amplification"]]
    assert alphas[0] < alphas[1] < alphas[2]
    assert alphas[0] == pytest.approx(0.29, abs=0.08)
    assert alphas[2] == pytest.approx(0.80, abs=0.08)


def test_reference_has_unit_alpha_and_cr(demo):
    table = fit_table(demo, assay="assay", reference="FullAgonist")
    ref = table[table["ligand"] == "FullAgonist"]
    assert len(ref) == 3
    assert all(v == pytest.approx(1.0) for v in ref["alpha"])
    assert all(v == pytest.approx(1.0) for v in ref["cr"])


def test_missing_column_gives_clear_error(demo):
    with pytest.raises(KeyError, match="not found"):
        fit_table(demo, response="nope")


def test_unknown_reference_raises(demo):
    with pytest.raises(ValueError, match="reference ligand"):
        fit_table(demo, assay="assay", reference="Nonexistent")


def test_bootstrap_columns_present(demo):
    sub = demo[demo["assay"] == "low amplification"]
    table = fit_table(sub, reference="FullAgonist", n_boot=40, seed=1)
    row = table[table["ligand"] == "PartialAgonist"].iloc[0]
    assert row["alpha_low"] < row["alpha"] < row["alpha_high"]


def test_cli_roundtrip(tmp_path, capsys):
    data = tmp_path / "data.csv"
    out = tmp_path / "table.csv"
    png = tmp_path / "fig.png"
    assert main(["demo", str(data)]) == 0
    code = main(["fit", str(data), "--assay", "assay", "--reference", "FullAgonist", "--out", str(out), "--plot", str(png)])
    assert code == 0
    table = pd.read_csv(out)
    assert {"alpha", "cr", "logec50"} <= set(table.columns)
    assert png.stat().st_size > 1000
    assert "PartialAgonist" in capsys.readouterr().out


def test_cli_reports_bad_column(tmp_path, capsys):
    data = tmp_path / "data.csv"
    main(["demo", str(data)])
    assert main(["fit", str(data), "--response", "missing"]) == 2
    assert "not found" in capsys.readouterr().err

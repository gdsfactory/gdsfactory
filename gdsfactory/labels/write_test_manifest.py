"""Converts CSV of test site labels into a CSV test manifest."""

import csv
import pathlib

import gdsfactory as gf
from gdsfactory.samples.sample_reticle import sample_reticle
from gdsfactory.typings import Iterable


def write_test_manifest(
    c: gf.Component,
    csvpath: str | pathlib.Path,
    cell_name_prefixes: Iterable[str] = ("spiral*",),
    analysis: str = "[power_envelope]",
    analysis_parameters: str = '[{"n": 10, "wvl_of_interest_nm": 1550}]',
) -> None:
    """Converts CSV of test site labels into a CSV test manifest.

    Args:
        c: the component to write the test manifest for.
        csvpath: the path to the CSV file to write.
        cell_name_prefixes: the prefixes of the cells to include in the test manifest.
        analysis: the analysis to run on the cells.
        analysis_parameters: the parameters to use for the analysis.
    """
    with open(csvpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cell",
                "x",
                "y",
                "analysis",
                "analysis_parameters",
            ]
        )

        for prefix in cell_name_prefixes:
            ci = c._kdb_cell.begin_instances_rec()
            ci.targets = prefix
            while not ci.at_end():
                _c = c.kcl[ci.inst_cell().cell_index()]
                _disp = (ci.trans() * ci.inst_trans()).disp
                writer.writerow(
                    [
                        _c.name,
                        _disp.x,
                        _disp.y,
                        analysis,
                        analysis_parameters,
                    ]
                )
                ci.next()


if __name__ == "__main__":
    import pandas as pd

    c = sample_reticle()
    c.show()
    gdspath = c.write_gds()
    csvpath = gdspath.with_suffix(".csv")
    write_test_manifest(c, csvpath)
    df = pd.read_csv(csvpath)
    print(df)

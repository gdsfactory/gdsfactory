"""Converts CSV of test site labels into a CSV test manifest."""

import csv
import json
import pathlib

import gdsfactory as gf
from gdsfactory.samples.sample_reticle import sample_reticle
from gdsfactory.typings import Iterable


def write_test_manifest(
    component: gf.Component,
    csvpath: str | pathlib.Path,
    cell_name_prefixes: Iterable[str] | None = None,
    analysis: str = "[power_envelope]",
    analysis_parameters: str = '[{"n": 10, "wvl_of_interest_nm": 1550}]',
) -> None:
    """Converts CSV of test site labels into a CSV test manifest.

    Args:
        component: the component to write the test manifest for.
        csvpath: the path to the CSV file to write.
        cell_name_prefixes: the prefixes of the cells to include in the test manifest.
        analysis: list of analysis to run on the cells.
        analysis_parameters: list of parameters to use for the analysis.
    """
    cell_name_prefixes = cell_name_prefixes or []
    cell_name_prefixes = list(cell_name_prefixes)
    c = component

    if not cell_name_prefixes:
        for cell_index in c.each_child_cell():
            ci = c.kcl[cell_index]
            cell_name_prefixes.append(f"{ci.name}*")

    with open(csvpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cell",
                "x",
                "y",
                "info",
                "analysis",
                "analysis_parameters",
            ]
        )

        for prefix in cell_name_prefixes:
            ci = c._kdb_cell.begin_instances_rec()
            ci.targets = prefix
            for _ci in ci.each():
                _c = c.kcl[_ci.inst_cell().cell_index()]
                disp = (_ci.trans() * _ci.inst_trans()).disp

                writer.writerow(
                    [
                        _c.name,
                        disp.x,
                        disp.y,
                        json.dumps(_c.info.model_dump()),
                        analysis,
                        analysis_parameters,
                    ]
                )


if __name__ == "__main__":
    import pandas as pd

    c = sample_reticle()
    # cit = c.begin_instances_rec()
    # cit.min_depth = 1
    # cit.max_depth = 1

    # for i in cit.each():
    #     cell_index = i.cell_index()
    #     ci = c.kcl[cell_index]
    #     print(ci.name)

    # for cell_index in c.each_child_cell():
    #     ci = c.kcl[cell_index]
    #     print(ci.name)

    # while not cit.at_end():
    #     ci = c.kcl[cit.inst_cell().cell_index()]
    #     print(ci.name)
    # c.show()

    # iter = c.begin_instances_rec()
    # iter.min_depth=1
    # iter.max_depth=1
    # for _iter in iter.each():
    #     cell_index = iter.cell_index()

    gdspath = c.write_gds()
    csvpath = gdspath.with_suffix(".csv")
    write_test_manifest(c, csvpath)
    df = pd.read_csv(csvpath)
    print(df)

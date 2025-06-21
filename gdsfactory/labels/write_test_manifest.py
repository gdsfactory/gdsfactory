"""Converts CSV of test site labels into a CSV test manifest."""

import csv
import json
import pathlib
from collections.abc import Iterable, Sequence

import gdsfactory as gf


def write_test_manifest(
    component: gf.Component,
    csvpath: str | pathlib.Path,
    search_strings: Iterable[str] | None = None,
    parameters: Sequence[str] = (
        "doe",
        "analysis",
        "analysis_parameters",
        "measurement",
        "measurement_parameters",
        "ports_optical",
        "ports_electrical",
    ),
    warn_if_missing: bool = True,
) -> None:
    """Converts CSV of test site labels into a CSV test manifest.

    It only includes cells that have a "doe" key in their info dictionary.

    Args:
        component: the component to write the test manifest for.
        csvpath: the path to the CSV file to write.
        search_strings: the search_strings of the cells to include in the test manifest.
            If None, all cells one level below top cell are included.
        parameters: the parameters to include in the test manifest as columns.
        warn_if_missing: if True, warn if a parameter is missing from a cell's info dictionary.
    """
    search_strings = search_strings or []
    search_strings = list(search_strings)
    c = component

    with open(csvpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cell",
                "x",
                "y",
                "info",
                "ports",
                "settings",
            ]
            + list(parameters)
        )

        ci = c.kdb_cell.begin_instances_rec()
        if search_strings:
            ci.targets = "{" + ",".join(search_strings) + "}"  # type: ignore[assignment]
        else:
            ci.targets = c.called_cells()

        for _ci in ci.each():
            cell = c.kcl[_ci.inst_cell().cell_index()]

            if cell.info.get("doe"):
                disp = (_ci.trans() * _ci.inst_trans()).disp
                dtrans = _ci.dtrans() * _ci.inst_dtrans()
                ports = {
                    p.name: gf.port.to_dict(p.copy(trans=dtrans)) for p in cell.ports
                }

                values = [cell.info.get(key, "") for key in parameters]
                if warn_if_missing:
                    for key in parameters:
                        if key not in cell.info:
                            print(f"Warning: {key!r} missing from {cell.name!r}")

                writer.writerow(
                    [
                        cell.name,
                        disp.x * c.kcl.dbu,
                        disp.y * c.kcl.dbu,
                        json.dumps(cell.info.model_dump(exclude=set(parameters))),
                        json.dumps(ports),
                        cell.settings.model_dump_json(),
                    ]
                    + values
                )


if __name__ == "__main__":
    import pandas as pd

    from gdsfactory.samples.sample_reticle import sample_reticle

    c = sample_reticle()

    gdspath = c.write_gds()
    csvpath = gdspath.with_suffix(".csv")
    write_test_manifest(c, csvpath, search_strings=["ring_10"])
    df = pd.read_csv(csvpath)
    c.show()

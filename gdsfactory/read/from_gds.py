"""Read component GDS, JSON metadata and CSV ports."""
import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Union

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.import_gds import import_gds


def read_metadata(component: Component, gdspath: Union[str, Path]) -> None:
    """Add Component ports (CSV) and settings (JSON) into Component."""

    gdspath = Path(gdspath)
    if not gdspath.exists():
        raise FileNotFoundError(f"No such file '{gdspath}'")

    ports_filepath = gdspath.with_suffix(".ports")
    metadata_filepath = gdspath.with_suffix(".json")

    if ports_filepath.exists():
        with open(str(ports_filepath), newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")
            for r in reader:
                layer_type = int(r[5].strip().strip("("))
                data_type = int(r[6].strip().strip(")"))
                component.add_port(
                    name=r[0],
                    midpoint=[float(r[1]), float(r[2])],
                    orientation=int(r[3]),
                    width=float(r[4]),
                    layer=(layer_type, data_type),
                )

    if metadata_filepath.exists():
        with open(metadata_filepath) as f:
            data = json.load(f)
        cell_settings = data["cells"][component.name]
        component.settings.update(cell_settings)


@lru_cache(maxsize=None)
def from_gds(gdspath: Union[str, Path], **kwargs) -> Component:
    """Returns Component with ports (CSV) and metadata (JSON) info (if any).
    Args:
        gdspath: path of GDS file
        cellname: cell of the name to import (None) imports top cell
        flatten: if True returns flattened (no hierarchy)
        snap_to_grid_nm: snap to different nm grid (does not snap if False)

    """

    gdspath = Path(gdspath)
    if not gdspath.exists():
        raise FileNotFoundError(f"No such file '{gdspath}'")
    component = import_gds(gdspath)

    read_metadata(component=component, gdspath=gdspath)
    return component


if __name__ == "__main__":
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c = gf.read.from_gds(gdspath)

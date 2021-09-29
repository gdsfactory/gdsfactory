"""Read component GDS, JSON metadata and CSV ports."""
from functools import lru_cache
from pathlib import Path
from typing import Union

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.import_gds import import_gds


@lru_cache(maxsize=None)
def gds(gdspath: Union[str, Path], **kwargs) -> Component:
    """Returns Component with ports (CSV) and metadata (JSON) info (if any).
    Args:
        gdspath: path of GDS file
        cellname: cell of the name to import (None) imports top cell
        flatten: if True returns flattened (no hierarchy)
        snap_to_grid_nm: snap to different nm grid (does not snap if False)

    """

    if not gdspath.exists():
        raise FileNotFoundError(f"No such file '{gdspath}'")
    component = import_gds(gdspath)

    gf.read.read_ports(component=component, gdspath=gdspath)
    return component


if __name__ == "__main__":
    from gdsfactory.tests.test_load_component import test_load_component_gds

    test_load_component_gds()

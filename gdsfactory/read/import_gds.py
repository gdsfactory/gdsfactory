from __future__ import annotations

import warnings
from pathlib import Path

from kfactory import KCLayout

from gdsfactory.component import Component


def import_gds(
    gdspath: str | Path,
    cellname: str | None = None,
    **kwargs,
) -> Component:
    """Reads a GDS file and returns a Component.

    Args:
        gdspath: path to GDS file.
        cellname: name of the cell to return. Defaults to top cell.
    """

    if kwargs:
        warnings.warn(f"kwargs {kwargs} are not used")

    temp_kcl = KCLayout(name=str(gdspath))
    temp_kcl.read(gdspath)
    cellname = cellname or temp_kcl.top_cell().name
    kcell = temp_kcl[cellname]

    c = Component()
    c.name = cellname
    c._kdb_cell.copy_tree(kcell._kdb_cell)

    c.ports = kcell.ports
    c._settings = kcell.settings.model_copy()
    c.info = kcell.info.model_copy()
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    gdspath = c.write_gds()

    c = import_gds(gdspath)
    c.show()

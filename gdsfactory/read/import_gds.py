from __future__ import annotations

import warnings
from pathlib import Path

import kfactory as kf
from kfactory import KCLayout


def import_gds(
    gdspath: str | Path,
    cellname: str | None = None,
    **kwargs,
) -> kf.KCell:
    """Reads a GDS file and returns a KLayout cell.

    Args:
        gdspath: path to GDS file.
        cellname: name of the cell to return. Defaults to top cell.
    """

    if kwargs:
        warnings.warn(f"kwargs {kwargs} are not used")

    kcl = KCLayout(name=str(gdspath))
    kcl.read(gdspath)
    return kcl[kcl.top_cell().name] if cellname is None else kcl[cellname]


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    gdspath = c.write_gds()

    c = import_gds(gdspath)
    c.show()

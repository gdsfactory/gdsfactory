import pathlib
from typing import Iterable

from pp.component import Component
from pp.import_gds import import_gds
from pp.types import ComponentOrPath, PathType


def gdspaths(cells: Iterable[ComponentOrPath]) -> Component:
    """Combine all GDS files or gdsfactory components into a gdsfactory component.

    Args:
        cells: List of gdspaths or Components
    """
    c = Component()

    for cell in cells:
        if not isinstance(cell, Component):
            cell = import_gds(cell)
        c << cell

    return c


def gdsdir(dirpath: PathType) -> Component:
    """Merges GDS cells from a directory into a single Component"""
    dirpath = pathlib.Path(dirpath)
    return gdspaths(dirpath.glob("*.gds"))


if __name__ == "__main__":
    from pp.config import diff_path

    # c = gdspaths([pp.components.straight(), pp.components.bend_circular()])
    # leave these two lines to end up tests showing the diff
    c = gdspaths(diff_path.glob("*.gds"))
    c.show()

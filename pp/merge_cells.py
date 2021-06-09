import pathlib
from typing import Iterable

from pp.component import Component
from pp.import_gds import import_gds
from pp.types import ComponentOrPath, PathType


def merge_cells(cells: Iterable[ComponentOrPath]) -> Component:
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


def merge_cells_from_directory(dirpath: PathType) -> Component:
    """Merges GDS cells from a directory into a single Component"""
    dirpath = pathlib.Path(dirpath)
    return merge_cells(dirpath.glob("*.gds"))


if __name__ == "__main__":
    from pp.config import diff_path

    # c = merge_cells([pp.components.straight(), pp.components.bend_circular()])
    # leave these two lines to end up tests showing the diff
    c = merge_cells(diff_path.glob("*.gds"))
    c.show()

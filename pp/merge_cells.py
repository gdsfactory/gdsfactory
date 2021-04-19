from typing import Iterable

from pp.component import Component
from pp.import_gds import import_gds
from pp.types import ComponentOrPath


def merge_cells(cells: Iterable[ComponentOrPath]) -> Component:
    c = Component()

    for cell in cells:
        if not isinstance(cell, Component):
            cell = import_gds(cell)
        c << cell

    return c


if __name__ == "__main__":
    from pp.config import diff_path

    # c = merge_cells([pp.components.straight(), pp.components.bend_circular()])
    # leave these two lines to end up tests showing the diff
    c = merge_cells(diff_path.glob("*.gds"))
    c.show()

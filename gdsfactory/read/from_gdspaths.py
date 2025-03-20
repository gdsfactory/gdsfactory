from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from gdsfactory import logger
from gdsfactory.component import Component
from gdsfactory.read.import_gds import import_gds

if TYPE_CHECKING:
    from gdsfactory.typings import ComponentOrPath, PathType


def from_gdspaths(cells: "Sequence[ComponentOrPath]") -> Component:
    """Combine all GDS files or gf.components into a gf.component.

    Args:
        cells: List of gdspaths or Components.

    """
    component = Component(name="merged")

    for c in cells:
        if isinstance(c, str | pathlib.Path):
            logger.info(f"Loading {c!r}")
            c = import_gds(c)

        assert isinstance(c, Component)
        _ = component << c

    return component


def from_gdsdir(dirpath: "PathType") -> Component:
    """Merges GDS cells from a directory into a single Component."""
    dirpath = pathlib.Path(dirpath)
    assert dirpath.exists(), f"{dirpath} does not exist"
    return from_gdspaths(list(dirpath.glob("*.gds")))


if __name__ == "__main__":
    from gdsfactory.config import diff_path

    # c = gdspaths([gf.components.straight(), gf.components.bend_circular()])
    # leave these two lines to end up tests showing the diff
    c = from_gdspaths(list(diff_path.glob("*.gds")))
    c.show()

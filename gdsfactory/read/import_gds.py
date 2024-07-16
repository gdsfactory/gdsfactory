from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import kfactory as kf
from kfactory import KCLayout

from gdsfactory.component import Component


def import_gds(
    gdspath: str | Path,
    cellname: str | None = None,
    post_process: Callable[[Component], Component] | None = None,
    **kwargs,
) -> Component:
    """Reads a GDS file and returns a Component.

    Args:
        gdspath: path to GDS file.
        cellname: name of the cell to return. Defaults to top cell.
        post_process: function to run after reading the GDS file.
        kwargs: deprecated and ignored.
    """
    if kwargs:
        for k in kwargs:
            warnings.warn(f"kwargs {k!r} is deprecated and ignored")

    temp_kcl = KCLayout(name=str(gdspath))
    temp_kcl.read(gdspath)
    cellname = cellname or temp_kcl.top_cell().name
    kcell = temp_kcl[cellname]
    c = kcell_to_component(kcell)

    if post_process:
        post_process(c)
    return c


def kcell_to_component(kcell: kf.KCell) -> Component:
    c = Component()
    c._kdb_cell.copy_tree(kcell._kdb_cell)
    c.rebuild()
    c.ports = kcell.ports
    c._settings = kcell.settings.model_copy()
    c.info = kcell.info.model_copy()
    c.name = kcell.name
    return c


def import_gds_with_conflicts(
    gdspath: str | Path,
    cellname: str | None = None,
    name: str | None = None,
    **kwargs,
) -> Component:
    """Reads a GDS file and returns a Component.

    Args:
        gdspath: path to GDS file.
        cellname: name of the cell to return. Defaults to top cell.
        name: optional name.
        kwargs: deprecated and ignored.

    Modes:
        AddToCell: Add content to existing cell. Content of new cells is simply added to existing cells with the same name.
        OverwriteCell: The old cell is overwritten entirely (including child cells which are not used otherwise)
        RenameCell: The new cell will be renamed to become unique
        SkipNewCell: The new cell is skipped entirely (including child cells which are not used otherwise)
    """
    if kwargs:
        for k in kwargs:
            warnings.warn(f"kwargs {k!r} is deprecated and ignored")

    read_options = kf.kcell.load_layout_options()
    read_options.cell_conflict_resolution = (
        kf.kdb.LoadLayoutOptions.CellConflictResolution.RenameCell
    )
    top_cells = set(kf.kcl.top_cells())
    kf.kcl.read(gdspath, read_options, test_merge=True)
    new_top_cells = set(kf.kcl.top_cells()) - top_cells
    if len(new_top_cells) != 1:
        raise ValueError(f"Expected 1 new top cell, got {len(new_top_cells)}")

    if cellname is None:
        cellname = new_top_cells.pop().name if new_top_cells else kf.kcl.top_kcell.name
    kcell = kf.kcl[cellname]
    c = Component()
    c._kdb_cell.copy_tree(kcell._kdb_cell)
    c.ports = kcell.ports
    c._settings = kcell.settings.model_copy()
    c.info = kcell.info.model_copy()
    name = name or kcell.name
    c.name = name
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    gdspath = c.write_gds()

    c = import_gds(gdspath)
    c.show()

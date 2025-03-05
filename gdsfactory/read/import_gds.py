from __future__ import annotations

from pathlib import Path
from typing import Any

import kfactory as kf
from kfactory import KCLayout

from gdsfactory.component import Component
from gdsfactory.typings import PostProcesses


def import_gds(
    gdspath: str | Path,
    cellname: str | None = None,
    post_process: PostProcesses | None = None,
    rename_duplicated_cells: bool = False,
) -> Component:
    """Reads a GDS file and returns a Component.

    Args:
        gdspath: path to GDS file.
        cellname: name of the cell to return. Defaults to top cell.
        post_process: function to run after reading the GDS file.
        rename_duplicated_cells: if True, rename duplicated cells.
    """
    temp_kcl = KCLayout(name=str(gdspath))
    options = kf.utilities.load_layout_options()
    options.warn_level = 0
    temp_kcl.read(gdspath, options=options)
    cellname = cellname or temp_kcl.layout.top_cell().name
    kcell = temp_kcl[cellname]
    if rename_duplicated_cells:
        read_options = kf.utilities.load_layout_options()
        read_options.cell_conflict_resolution = (
            kf.kdb.LoadLayoutOptions.CellConflictResolution.RenameCell
        )

    if hasattr(temp_kcl, "cross_sections"):
        for cross_section in temp_kcl.cross_sections.cross_sections.values():
            kf.kcl.get_symmetrical_cross_section(cross_section)

    c = kcell_to_component(kcell)
    for pp in post_process or []:
        pp(c)

    temp_kcl.library.delete()
    del kf.layout.kcls[temp_kcl.name]
    return c


def kcell_to_component(kcell: kf.kcell.ProtoTKCell[Any]) -> Component:
    c = Component()
    c.kdb_cell.copy_tree(kcell.kdb_cell)
    c.add_ports(kcell.ports)
    c.settings = kcell.settings.model_copy()
    c.info = kcell.info.model_copy()
    c.name = kcell.name
    return c


def import_gds_with_conflicts(
    gdspath: str | Path,
    cellname: str | None = None,
) -> Component:
    """Reads a GDS file and returns a Component.

    Args:
        gdspath: path to GDS file.
        cellname: name of the cell to return. Defaults to top cell.

    Modes:
        AddToCell: Add content to existing cell. Content of new cells is simply added to existing cells with the same name.
        OverwriteCell: The old cell is overwritten entirely (including child cells which are not used otherwise)
        RenameCell: The new cell will be renamed to become unique
        SkipNewCell: The new cell is skipped entirely (including child cells which are not used otherwise)
    """
    return import_gds(gdspath, cellname=cellname, rename_duplicated_cells=True)


if __name__ == "__main__":
    from gdsfactory.components import mzi

    c = mzi()
    c.pprint_ports()
    gdspath = c.write_gds()

    c = import_gds(gdspath)
    c.pprint_ports()
    c.show()

from __future__ import annotations

from pathlib import Path
from typing import Any

import kfactory as kf
import kfactory.utilities
from kfactory import KCLayout

from gdsfactory.component import Component
from gdsfactory.typings import PostProcesses


def import_gds(
    gdspath: str | Path,
    cellname: str | None = None,
    post_process: PostProcesses | None = None,
    rename_duplicated_cells: bool = False,
    skip_new_cells: bool = False,
) -> Component:
    """Reads a GDS file and returns a Component.

    Args:
        gdspath: path to GDS file.
        cellname: name of the cell to return. Defaults to top cell.
        post_process: function to run after reading the GDS file.
        rename_duplicated_cells: if True, rename duplicated cells. By default appends $n to the cell name.
        skip_new_cells: if True, skip new cells that conflict with existing ones.

    """
    temp_kcl = KCLayout(name=str(gdspath))
    options = kf.utilities.load_layout_options()
    options.warn_level = 0

    if skip_new_cells:
        options.cell_conflict_resolution = (
            kf.kdb.LoadLayoutOptions.CellConflictResolution.SkipNewCell
        )
    elif rename_duplicated_cells:
        options.cell_conflict_resolution = (
            kf.kdb.LoadLayoutOptions.CellConflictResolution.RenameCell
        )

    temp_kcl.read(gdspath, options=options)

    if cellname is None:
        if len(temp_kcl.layout.top_cells()) > 1:
            raise ValueError(
                "GDS file has multiple top cells. Use cellname to select a specific one, or use gf.read.import_gds_multiple_top_cells instead.\n"
                + f"Top cells: {[c.name for c in temp_kcl.layout.top_cells()]}"
            )
        cellname = temp_kcl.layout.top_cell().name

    kcell = temp_kcl[cellname]

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
    kcell.set_meta_data()

    for ci in kcell.called_cells():
        kcell.kcl[ci].set_meta_data()

    c = Component()
    c.name = kcell.name
    c.kdb_cell.copy_tree(kcell.kdb_cell)
    c.copy_meta_info(kcell.kdb_cell)
    c.get_meta_data()

    for ci in c.called_cells():
        c.kcl[ci].get_meta_data()

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


def import_gds_multiple_top_cells(
    gdspath: str | Path,
    cellnames: list[str] | None = None,
    post_process: PostProcesses | None = None,
    rename_duplicated_cells: bool = False,
    skip_new_cells: bool = False,
) -> dict[str, Component]:
    """Reads a GDS file and returns a dictionary of its top cells as Components.

    Args:
        gdspath: path to GDS file.
        cellnames: list of names of the cells to return. Defaults to all top cells.

    Returns:
        dict of cellname to Component.

    Raises:
        ValueError: if any of the provided cellnames are not found among the top cells.

    """
    temp_kcl = KCLayout(name=str(gdspath))
    options = kf.utilities.load_layout_options()
    options.warn_level = 0

    if skip_new_cells:
        options.cell_conflict_resolution = (
            kf.kdb.LoadLayoutOptions.CellConflictResolution.SkipNewCell
        )
    elif rename_duplicated_cells:
        options.cell_conflict_resolution = (
            kf.kdb.LoadLayoutOptions.CellConflictResolution.RenameCell
        )

    temp_kcl.read(gdspath, options=options)

    components = {}

    kcells = temp_kcl.layout.top_cells()

    if cellnames is not None:
        # Validate provided cellnames and surface all invalid names at once
        available_cellnames = {kcell.name for kcell in kcells}
        missing = set(cellnames) - available_cellnames
        if missing:
            raise ValueError(
                "Unknown cellnames requested. These names are not present in the GDS top cells: "
                + ", ".join(sorted(missing))
                + ".\n"
                + f"Available top cells: {sorted(available_cellnames)}"
            )
        # Filter kcells to include only those specified in cellnames
        kcells = [kcell for kcell in kcells if kcell.name in cellnames]

    for kcell in kcells:
        components[kcell.name] = kcell_to_component(
            temp_kcl[kcell.name]
        )  # Convert each kcell to Component class and store in dictionary using its name as the key

    for pp in post_process or []:
        for c in components.values():
            pp(c)

    temp_kcl.library.delete()
    del kf.layout.kcls[temp_kcl.name]
    return components

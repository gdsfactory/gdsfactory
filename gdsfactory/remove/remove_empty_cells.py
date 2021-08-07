import sys
from pathlib import Path
from typing import Union

from gdsfactory.import_gds import import_gds


def remove_cell(component, cell_name) -> None:
    """Removes cell from a component"""
    all_cells = component.get_dependencies(recursive=True)
    all_cell_names = set([c.name for c in all_cells])
    if cell_name in all_cell_names:
        for c in all_cells:
            to_remove = []
            for e in c.references:
                to_remove += [e]
            for e in to_remove:
                # print("removing", e)
                c.remove(e)


def is_empty(c):
    # print(len(c.polygons), len(c.get_dependencies()))
    return len(c.polygons) == 0 and len(c.get_dependencies()) == 0


def remove_empty_cells(
    gds: Union[str, Path], recursive: bool = True, recurse_depth: int = 0
):
    """Returns the list cells to cells."""
    if isinstance(gds, (str, Path)):
        gds = import_gds(gds)

    cells = gds.get_dependencies(recursive=True)
    cells_to_remove = []
    for c in cells:
        if is_empty(c):
            cells_to_remove += [c]

    for c in cells_to_remove:
        print(c.name)
        remove_cell(gds, c.name)

    if recursive:
        while (len(cells_to_remove)) > 0:
            print(recurse_depth)
            sys.stdout.flush()
            cells_to_remove = remove_empty_cells(
                gds, recursive=True, recurse_depth=recurse_depth + 1
            )

    return cells_to_remove


def remove_empty_cells_from_gds_file(gdspath):
    component = import_gds(gdspath)
    remove_empty_cells(component)
    component.write_gds(gdspath[:-4] + "_cleaned.gds")


if __name__ == "__main__":
    remove_empty_cells_from_gds_file()

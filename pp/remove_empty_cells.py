from pp.import_gds import import_gds
from pp.write_component import write_gds
import os
from pp.config import CONFIG
import sys


def remove_cell(device, cell_name):
    all_cells = device.get_dependencies(recursive=True)
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


def remove_empty_cells(gds, recursive=True, recurse_depth=0):
    """
    returns the list of removed cells
    """
    if type(gds) == str:
        gds = import_gds(gds)

    cells = gds.get_dependencies(recursive=True)
    _to_remove = []
    for c in cells:
        if is_empty(c):
            _to_remove += [c]

    for c in _to_remove:
        print(c.name)
        remove_cell(gds, c.name)

    if recursive:
        while (len(_to_remove)) > 0:
            print(recurse_depth)
            sys.stdout.flush()
            _to_remove = remove_empty_cells(
                gds, recursive=True, recurse_depth=recurse_depth + 1
            )

    return _to_remove


def remove_empty_cells_from_gds_file(gdspath):
    gds = import_gds(gdspath)
    remove_empty_cells(gds)
    write_gds(gds, gdspath[:-4] + "_cleaned.gds")


def clean_teg1():
    path_teg1 = os.path.join(CONFIG["mask_directory"], "teg1.gds")
    remove_empty_cells_from_gds_file(path_teg1)


if __name__ == "__main__":
    clean_teg1()

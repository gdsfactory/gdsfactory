"""Returns Component from OASIS file.

based on https://github.com/atait/phidl.git

"""

import os
from typing import Optional

from gdsfactory.read.import_gds import import_gds
from gdsfactory.types import Component, PathType


def import_oas(
    filename: PathType, cellname: Optional[str] = None, flatten: bool = False
) -> Component:
    """Import Component from OASIS.

    Args:
        filename: filepath for oasis file.
        cellname: to import. Defaults to TOP cell.
        flatten: flattens hierarchy.

    """
    filename = str(filename)

    if filename.lower().endswith(".gds"):
        # you are looking for import_gds
        retval = import_gds(filename, cellname=cellname, flatten=flatten)
        return retval
    try:
        import klayout.db as pya
    except ImportError as err:
        err.args = (
            "you need klayout package to read OASIS"
            "pip install klayout\n" + err.args[0],
        ) + err.args[1:]
        raise
    if not filename.lower().endswith(".oas"):
        filename += ".oas"
    fileroot = os.path.splitext(filename)[0]
    tempfilename = f"{fileroot}-tmp.gds"

    layout = pya.Layout()
    layout.read(filename)

    # We want to end up with one Component. If the imported layout has multiple top cells,
    # a new toplevel is created, and they go into the second level
    if len(layout.top_cells()) > 1:
        topcell = layout.create_cell("toplevel")
        rot_DTrans = pya.DTrans.R0
        origin = pya.DPoint(0, 0)
        for childcell in layout.top_cells():
            if childcell == topcell:
                continue
            topcell.insert(
                pya.DCellInstArray(
                    childcell.cell_index(), pya.DTrans(rot_DTrans, origin)
                )
            )
    else:
        topcell = layout.top_cell()
    topcell.write(tempfilename)
    retval = import_gds(tempfilename, cellname=cellname, flatten=flatten)
    os.remove(tempfilename)
    return retval


if __name__ == "__main__":
    c = import_oas(filename="/home/jmatres/demo/a.oas")
    c.show()

from pathlib import Path
from typing import Union


def check_duplicated_cells(gdspath: Union[Path, str]):
    """Reads cell and checks for duplicated cells.

    Args:
        gdspath: path to GDS or Component

    """
    import klayout.db as pya

    from gdsfactory.component import Component

    if isinstance(gdspath, Component):
        gdspath.flatten()
        gdspath = gdspath.write_gds()
    layout = pya.Layout()
    layout.read(str(gdspath))
    return layout.top_cell()

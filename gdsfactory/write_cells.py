import pathlib
from typing import Optional

import gdspy

from gdsfactory.component import Component
from gdsfactory.config import CONFIG
from gdsfactory.name import clean_name
from gdsfactory.read.import_gds import import_gds
from gdsfactory.types import PathType

script_prefix = """
from pathlib import PosixPath
import gdsfactory as gf

# Only umcomment 1 line from the next 3 lines
decorator = gf.add_ports.add_ports_from_markers_inside
# decorator = gf.add_ports.add_ports_from_markers_center
# decorator = None
"""


def get_import_gds_script(dirpath: PathType) -> str:
    """Returns import_gds script from a directory with all the GDS files."""
    dirpath = pathlib.Path(dirpath)
    script = [script_prefix]
    script += [f"gdsdir = {dirpath.absolute()!r}\n"]
    script += [
        f"{clean_name(cell.stem)} = gf.import_gds(gdsdir / {cell.stem + cell.suffix!r}, decorator=decorator)"
        for cell in dirpath.glob("*.gds")
    ]
    return "\n".join(script)


def write_cells(
    gdspath: PathType, dirpath: Optional[PathType] = None, recursively: bool = True
) -> None:
    """Writes cells into separate GDS files.

    Args:
        gdspath: to read cells from
        dirpath: directory path to store gds cells
        recursively: writes all subcells
    """
    gdspath = pathlib.Path(gdspath)
    dirpath = dirpath or gdspath.parent / "gds"

    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(gdspath)
    top_level_cells = gdsii_lib.top_level()
    cellnames = [c.name for c in top_level_cells]

    for cellname in cellnames:
        component = import_gds(gdspath, cellname=cellname)
        component.write_gds(f"{pathlib.Path(dirpath)/cellname}.gds")
        if recursively:
            write_cells_from_component(component=component, dirpath=dirpath)


def write_cells_from_component(
    component: Component, dirpath: Optional[PathType] = None
) -> None:
    """Writes all Component cells.

    Args:
        component:
        dirpath: directory path to write GDS (defaults to CWD)
    """
    dirpath = dirpath or pathlib.Path.cwd()
    if component.references:
        for ref in component.references:
            component = ref.parent
            component.write_gds(f"{pathlib.Path(dirpath)/component.name}.gds")
            write_cells_from_component(component=component, dirpath=dirpath)
    else:
        component.write_gds(f"{pathlib.Path(dirpath)/component.name}.gds")


if __name__ == "__main__":
    import gdsfactory as gf

    gdspath = CONFIG["gdsdir"] / "mzi2x2.gds"
    gf.show(gdspath)

    write_cells(gdspath=gdspath, dirpath="extra/gds")

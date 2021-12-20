import pathlib
from typing import Optional

import gdspy

from gdsfactory.component import Component
from gdsfactory.config import CONFIG
from gdsfactory.read.import_gds import import_gds
from gdsfactory.types import PathType


def write_top_cells(gdspath: PathType, **kwargs) -> None:
    """Writes each top level cell into a separate GDS file."""
    gdspath = pathlib.Path(gdspath)
    dirpath = gdspath.parent
    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(gdspath)
    top_level_cells = gdsii_lib.top_level()
    cellnames = [c.name for c in top_level_cells]

    for cellname in cellnames:
        component = import_gds(gdspath, cellname=cellname, **kwargs)
        component.write_gds(f"{dirpath/cellname}.gds")


def write_cells(
    gdspath: PathType,
    dirpath: Optional[PathType] = None,
) -> None:
    """Writes each top level cell into separate GDS file.

    Args:
        gdspath: to read cells from
        dirpath: directory path to store gds cells
    """
    gdspath = gdspath or pathlib.Path(gdspath)
    dirpath = dirpath or gdspath.parent / "gds"

    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(gdspath)
    top_level_cells = gdsii_lib.top_level()
    cellnames = [c.name for c in top_level_cells]

    for cellname in cellnames:
        component = import_gds(gdspath, cellname=cellname)
        component.write_gds(f"{pathlib.Path(dirpath)/cellname}.gds")
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

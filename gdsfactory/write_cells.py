import datetime
import pathlib
from typing import Optional

import gdspy

from gdsfactory.component import _timestamp2019
from gdsfactory.config import CONFIG, logger
from gdsfactory.name import clean_name
from gdsfactory.types import PathType

script_prefix = """
from pathlib import PosixPath
from functools import partial
import gdsfactory as gf

add_ports_optical = gf.partial(gf.add_ports.add_ports_from_markers_inside, pin_layer=(1, 0), port_layer=(2, 0))
add_ports_electrical = gf.partial(gf.add_ports.add_ports_from_markers_inside, pin_layer=(41, 0), port_layer=(1, 0))
add_ports = gf.compose(add_ports_optical, add_ports_electrical)

"""


def get_import_gds_script(dirpath: PathType) -> str:
    """Returns import_gds script from a directory with all the GDS files."""
    dirpath = pathlib.Path(dirpath)
    script = [script_prefix]
    script += [f"gdsdir = {dirpath.absolute()!r}\n"]
    script += [
        "import_gds = partial(gf.import_gds, gdsdir=gdsdir, decorator=add_ports)\n"
    ]

    cells = [
        f"{clean_name(cell.stem)} = partial(import_gds, "
        f"{cell.stem + cell.suffix!r})"
        for cell in dirpath.glob("*.gds")
    ]
    script += sorted(cells)
    return "\n".join(script)


def write_cells_recursively(
    cell: gdspy.Cell,
    unit: float = 1e-6,
    precision: float = 1e-9,
    timestamp: Optional[datetime.datetime] = _timestamp2019,
    dirpath: pathlib.Path = Optional[pathlib.Path],
):
    """Write gdspy cells recursively

    Args:
        cell: gdspy cell
        unit: unit size for objects in library. 1um by default.
        precision: for object dimensions in the library (m). 1nm by default.
        timestamp: Defaults to 2019-10-25. If None uses current time.
        dirpath: directory for the GDS file
    """
    dirpath = dirpath or pathlib.Path.cwd()

    for cell in cell.get_dependencies():
        gdspath = f"{pathlib.Path(dirpath)/cell.name}.gds"
        lib = gdspy.GdsLibrary(unit=unit, precision=precision)
        lib.write_gds(gdspath, cells=[cell], timestamp=timestamp)
        logger.info(f"Write GDS to {gdspath}")

        if cell.get_dependencies():
            write_cells_recursively(
                cell=cell,
                unit=unit,
                precision=precision,
                timestamp=timestamp,
                dirpath=dirpath,
            )


def write_cells(
    gdspath: Optional[PathType] = None,
    cell: Optional[gdspy.Cell] = None,
    dirpath: Optional[PathType] = None,
    unit: float = 1e-6,
    precision: float = 1e-9,
    timestamp: Optional[datetime.datetime] = _timestamp2019,
    recursively: bool = True,
) -> None:
    """Writes cells into separate GDS files.

    Args:
        gdspath: GDS file. You need to define either gdspath or cell.
        cell: gdspy cell. You need to define either gdspath or cell.
        unit: unit size for objects in library. 1um by default.
        precision: for object dimensions in the library (m). 1nm by default.
        timestamp: Defaults to 2019-10-25. If None uses current time.
        dirpath: directory for the GDS file. Defaults to current working directory.
        recursively: writes all cells recursively. If False writes only top cells.
    """
    if cell is None and gdspath is None:
        raise ValueError("You need to specify component or gdspath")

    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(gdspath)
    top_level_cells = gdsii_lib.top_level()

    dirpath = dirpath or pathlib.Path.cwd()
    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(exist_ok=True, parents=True)

    for cell in top_level_cells:
        gdspath = f"{pathlib.Path(dirpath)/cell.name}.gds"
        lib = gdspy.GdsLibrary(unit=unit, precision=precision)
        lib.write_gds(gdspath, cells=[cell], timestamp=timestamp)
        logger.info(f"Write GDS to {gdspath}")

        if recursively:
            write_cells_recursively(
                cell=cell,
                unit=unit,
                precision=precision,
                timestamp=timestamp,
                dirpath=dirpath,
            )


if __name__ == "__main__":
    import gdsfactory as gf

    gdspath = CONFIG["gdsdir"] / "mzi2x2.gds"
    gf.show(gdspath)

    write_cells(gdspath=gdspath, dirpath="extra/gds")

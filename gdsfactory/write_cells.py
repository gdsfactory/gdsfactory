"""Generate the code from a GDS file based PDK."""

import datetime
import pathlib
from pathlib import Path
from typing import Dict, Optional

import gdspy

from gdsfactory.component import _timestamp2019
from gdsfactory.config import CONFIG, logger
from gdsfactory.name import clean_name
from gdsfactory.types import PathType

script_prefix = """
from pathlib import PosixPath
from functools import partial
import gdsfactory as gf

add_ports_optical = gf.partial(
    gf.add_ports.add_ports_from_markers_inside, pin_layer=(1, 0), port_layer=(2, 0)
)
add_ports_electrical = gf.partial(
    gf.add_ports.add_ports_from_markers_inside, pin_layer=(41, 0), port_layer=(1, 0)
)
add_ports = gf.compose(add_ports_optical, add_ports_electrical)

"""


def get_script(gdspath: PathType, module: Optional[str] = None) -> str:
    """Returns script for importing a fixed cell.

    Args:
        gdspath: fixed cell gdspath.
        module: if any includes plot directive.

    """
    cell = clean_name(gdspath.stem)
    gdspath = gdspath.stem + gdspath.suffix

    package = module.split(".")[0] if module and "." in module else module
    if module:
        return f"""

@gf.cell
def {cell}()->gf.Component:
    '''Returns {cell} fixed cell.

    .. plot::
      :include-source:

      import {package}

      c = {module}.{cell}()
      c.plot()
    '''
    return import_gds({str(gdspath)!r})

"""

    else:
        return f"""

@gf.cell
def {cell}()->gf.Component:
    '''Returns {cell} fixed cell.
    '''
    return import_gds({str(gdspath)!r})

"""


def get_import_gds_script(dirpath: PathType, module: Optional[str] = None) -> str:
    """Returns import_gds script from a directory with all the GDS files.

    Args:
        dirpath: fixed cell directory path.
        module: if any includes plot directive.

    """
    dirpath = pathlib.Path(dirpath)
    script = [script_prefix]
    script += [f"gdsdir = {dirpath.absolute()!r}\n"]
    script += [
        "import_gds = partial(gf.import_gds, gdsdir=gdsdir, decorator=add_ports)\n"
    ]

    cells = [get_script(gdspath, module=module) for gdspath in dirpath.glob("*.gds")]
    script += sorted(cells)
    return "\n".join(script)


def write_cells_recursively(
    cell: gdspy.Cell,
    unit: float = 1e-6,
    precision: float = 1e-9,
    timestamp: Optional[datetime.datetime] = _timestamp2019,
    dirpath: Optional[pathlib.Path] = None,
) -> Dict[str, Path]:
    """Write gdspy cells recursively.

    Args:
        cell: gdspy cell.
        unit: unit size for objects in library. 1um by default.
        precision: for library dimensions (m). 1nm by default.
        timestamp: Defaults to 2019-10-25. If None uses current time.
        dirpath: directory for the GDS file to write to.

    Returns:
        gdspaths: dict of cell name to gdspath.

    """
    dirpath = dirpath or pathlib.Path.cwd()
    gdspaths = {}

    for c in cell.get_dependencies():
        gdspath = f"{pathlib.Path(dirpath)/c.name}.gds"
        lib = gdspy.GdsLibrary(unit=unit, precision=precision)
        lib.write_gds(gdspath, cells=[c], timestamp=timestamp)
        logger.info(f"Write GDS to {gdspath}")

        gdspaths[c.name] = gdspath

        if c.get_dependencies():
            gdspaths2 = write_cells_recursively(
                cell=c,
                unit=unit,
                precision=precision,
                timestamp=timestamp,
                dirpath=dirpath,
            )
            gdspaths.update(gdspaths2)

    return gdspaths


def write_cells(
    gdspath: Optional[PathType] = None,
    dirpath: Optional[PathType] = None,
    unit: float = 1e-6,
    precision: float = 1e-9,
    timestamp: Optional[datetime.datetime] = _timestamp2019,
    recursively: bool = True,
    flatten: bool = False,
) -> Dict[str, Path]:
    """Writes cells into separate GDS files.

    Args:
        gdspath: GDS file to write cells.
        dirpath: directory path to write GDS files to.
            Defaults to current working directory.
        unit: unit size for objects in library. 1um by default.
        precision: for object dimensions in the library (m). 1nm by default.
        timestamp: Defaults to 2019-10-25. If None uses current time.
        recursively: writes all cells recursively. If False writes only top cells.
        flatten: flatten cell.

    Returns:
        gdspaths: dict of cell name to gdspath.

    """
    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(gdspath)
    top_level_cells = gdsii_lib.top_level()

    dirpath = dirpath or pathlib.Path.cwd()
    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(exist_ok=True, parents=True)

    gdspaths = {}

    for cell in top_level_cells:
        if flatten:
            cell = cell.flatten()
        gdspath = f"{pathlib.Path(dirpath)/cell.name}.gds"
        lib = gdspy.GdsLibrary(unit=unit, precision=precision)
        gdspy.library.use_current_library = False
        lib.write_gds(gdspath, cells=[cell], timestamp=timestamp)
        logger.info(f"Write {cell.name} to {gdspath}")
        gdspaths[cell.name] = gdspath

        if recursively:
            gdspaths2 = write_cells_recursively(
                cell=cell,
                unit=unit,
                precision=precision,
                timestamp=timestamp,
                dirpath=dirpath,
            )
            gdspaths.update(gdspaths2)
        return gdspaths


def test_write_cells():
    gdspath = CONFIG["gdsdir"] / "mzi2x2.gds"
    gdspaths = write_cells(gdspath=gdspath, dirpath="extra/gds")
    assert len(gdspaths) == 10, len(gdspaths)


if __name__ == "__main__":
    test_write_cells()
    import gdsfactory as gf

    # gdspath = CONFIG["gdsdir"] / "mzi2x2.gds"
    # gf.show(gdspath)
    # gdspaths = write_cells(gdspath=gdspath, dirpath="extra/gds")
    # print(len(gdspaths))

    sample_pdk_cells = gf.grid(
        [
            gf.components.straight,
            gf.components.bend_euler,
            gf.components.grating_coupler_elliptical,
        ]
    )
    sample_pdk_cells.write_gds("extra/pdk.gds")
    gf.write_cells.write_cells(gdspath="extra/pdk.gds", dirpath="extra/gds")

    print(gf.write_cells.get_import_gds_script("extra/gds", module="sky130.components"))

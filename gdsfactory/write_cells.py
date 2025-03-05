"""Generate the code from a GDS file based PDK."""

from __future__ import annotations

import pathlib
from pathlib import Path

import gdsfactory as gf
from gdsfactory import logger
from gdsfactory.name import clean_name
from gdsfactory.typings import PathType

script_prefix = """
from pathlib import Path
from functools import partial
import gdsfactory as gf

add_ports_optical = partial(
    gf.add_ports.add_ports_from_markers_inside, pin_layer=(1, 0), port_layer=(2, 0)
)
add_ports_electrical = partial(
    gf.add_ports.add_ports_from_markers_inside, pin_layer=(41, 0), port_layer=(1, 0)
)
add_ports = gf.compose(add_ports_optical, add_ports_electrical)

"""


def get_script(gdspath: PathType, module: str | None = None) -> str:
    """Returns script for importing a fixed cell.

    Args:
        gdspath: fixed cell gdspath.
        module: if any includes plot directive.

    """
    gdspath = pathlib.Path(gdspath)
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
    return import_gds(gdsdir/{str(gdspath)!r})

"""

    else:
        return f"""

@gf.cell
def {cell}()->gf.Component:
    '''Returns {cell} fixed cell.'''
    return import_gds(gdsdir/{str(gdspath)!r})

"""


def get_import_gds_script(dirpath: PathType, module: str | None = None) -> str:
    """Returns import_gds script from a directory with all the GDS files.

    Args:
        dirpath: fixed cell directory path.
        module: Optional plot directive to plot imported component.

    """
    dirpath = pathlib.Path(dirpath)
    if not dirpath.exists():
        raise ValueError(f"{str(dirpath.absolute())!r} does not exist.")

    gdspaths = list(dirpath.glob("*.gds")) + list(dirpath.glob("*.GDS"))

    if not gdspaths:
        raise ValueError(f"No GDS files found at {dirpath.absolute()!r}.")

    logger.info(f"Writing {len(gdspaths)} cells from {dirpath.absolute()!r}")

    script = [script_prefix]
    script += [f"gdsdir = Path({str(dirpath.absolute())!r})\n"]
    script += ["import_gds = partial(gf.import_gds, post_process=add_ports)\n"]

    cells = [get_script(gdspath, module=module) for gdspath in gdspaths]
    script += sorted(cells)
    return "\n".join(script)


def write_cells_recursively(
    gdspath: PathType,
    dirpath: PathType | None = None,
) -> dict[str, Path]:
    """Write gdstk cells recursively.

    Args:
        gdspath: gds file to write cells from.
        dirpath: directory for the GDS file to write to.

    Returns:
        gdspaths: dict of cell name to gdspath.
    """
    gf.kcl.read(gdspath)
    dirpath = dirpath or pathlib.Path.cwd()
    dirpath = pathlib.Path(dirpath).absolute()
    dirpath.mkdir(exist_ok=True, parents=True)

    gdspaths: dict[str, Path] = {}

    for cell_index in gf.kcl.each_cell_bottom_up():
        component = gf.kcl[cell_index]
        gdspath = dirpath / f"{component.name}.gds"
        component.write(gdspath)
        gdspaths[component.name] = gdspath
    return gdspaths


def write_cells(
    gdspath: PathType,
    dirpath: PathType | None = None,
) -> dict[str, Path]:
    """Writes cells into separate GDS files.

    Args:
        gdspath: GDS file to write cells.
        dirpath: directory path to write GDS files to.
            Defaults to current working directory.

    Returns:
        gdspaths: dict of cell name to gdspath.
    """
    gf.kcl.read(gdspath)
    components = [gf.kcl[top_cell.cell_index()] for top_cell in gf.kcl.top_cells()]

    dirpath = dirpath or pathlib.Path.cwd()
    dirpath = pathlib.Path(dirpath).absolute()
    dirpath.mkdir(exist_ok=True, parents=True)

    gdspaths: dict[str, Path] = {}

    for component in components:
        gdspath = dirpath / f"{component.name}.gds"
        component.write(gdspath)
        gdspaths[component.name] = gdspath
    return gdspaths


if __name__ == "__main__":
    c = gf.c.mzi()
    gdspath = c.write_gds()
    gf.clear_cache()
    write_cells_recursively(gdspath)

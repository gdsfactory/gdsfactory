import pathlib
from typing import Optional

import gdspy
from tqdm import tqdm

from gdsfactory.config import CONFIG, logger
from gdsfactory.name import clean_name
from gdsfactory.read.import_gds import import_gds
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


def write_cells(
    gdspath: PathType,
    dirpath: Optional[PathType] = None,
    unit: float = 1e-6,
    precision: float = 1e-9,
    recursively: bool = True,
    flatten: bool = False,
) -> None:
    """Writes cells into separate GDS files.

    Args:
        gdspath: GDS file. You need to define either gdspath.
        dirpath: output directory for GDS files. Defaults to current working directory.
        unit: unit size for objects in library. 1um by default.
        precision: for object dimensions in the library (m). 1nm by default.
        flatten: flatten cell.
    """
    dirpath = dirpath or pathlib.Path.cwd()
    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(exist_ok=True, parents=True)

    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(str(gdspath))
    top_level_cells = gdsii_lib.top_level()
    cellnames = [c.name for c in top_level_cells]

    for cellname in tqdm(cellnames):
        component = import_gds(gdspath=gdspath, cellname=cellname)
        gdspath_new = pathlib.Path(dirpath) / f"{component.name}.gds"
        logger.info(f"Write {cellname!r} to {gdspath_new}")
        if flatten:
            component.flatten()
        component.write_gds(gdspath=gdspath_new)


if __name__ == "__main__":
    import gdsfactory as gf

    gdspath = CONFIG["gdsdir"] / "mzi2x2.gds"
    gf.show(gdspath)

    write_cells(gdspath=gdspath, dirpath="extra/gds")

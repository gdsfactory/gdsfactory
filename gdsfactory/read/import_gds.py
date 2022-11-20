from pathlib import Path
from typing import Optional, Union, cast

import gdstk
from omegaconf import OmegaConf

from gdsfactory.cell import Settings, cell
from gdsfactory.component import Component
from gdsfactory.component_reference import ComponentReference
from gdsfactory.config import logger
from gdsfactory.name import get_name_short


@cell
def import_gds(
    gdspath: Union[str, Path],
    cellname: Optional[str] = None,
    gdsdir: Optional[Union[str, Path]] = None,
    read_metadata: bool = True,
    hashed_name: bool = True,
    **kwargs,
) -> Component:
    """Returns a Componenent from a GDS file.

    based on phidl/geometry.py

    if any cell names are found on the component CACHE we append a $ with a
    number to the name

    Args:
        gdspath: path of GDS file.
        cellname: cell of the name to import (None) imports top cell.
        gdsdir: optional GDS directory.
        read_metadata: loads metadata if it exists.
        hashed_name: appends a hash to a shortened component name.
        kwargs: extra to add to component.info (polarization, wavelength ...).
    """
    gdspath = Path(gdsdir) / Path(gdspath) if gdsdir else Path(gdspath)
    if not gdspath.exists():
        raise FileNotFoundError(f"No file {gdspath!r} found")

    metadata_filepath = gdspath.with_suffix(".yml")

    if gdspath.suffix.lower() == ".gds":
        gdsii_lib = gdstk.read_gds(str(gdspath))
    elif gdspath.suffix.lower() == ".oas":
        gdsii_lib = gdstk.read_oas(str(gdspath))
    else:
        raise ValueError(f"gdspath.suffix {gdspath.suffix!r} not .gds or .oas")

    top_level_cells = gdsii_lib.top_level()
    cellnames = [c.name for c in top_level_cells]
    cells_by_name = {c.name: c for c in top_level_cells}

    if not cellnames:
        raise ValueError(f"no cells found in {str(gdspath)!r}")

    if cellname is not None:
        if cellname not in cells_by_name:
            raise ValueError(
                f"cell {cellname!r} is not in file {gdspath} with cells {cellnames}"
            )
        topcell = cells_by_name[cellname]
    elif len(top_level_cells) == 1:
        topcell = top_level_cells[0]
    elif len(top_level_cells) > 1:
        raise ValueError(
            f"import_gds() There are multiple top-level cells in {gdspath!r}, "
            f"you must specify `cellname` to select of one of them among {cellnames}"
        )

    D_list = []
    cell_to_device = {}
    for c in gdsii_lib.cells:
        D = Component(name=c.name)
        D._cell = c
        D.name = c.name

        if hashed_name:
            D.name = get_name_short(D.name)

        cell_to_device[c] = D
        D_list += [D]

    for c, D in cell_to_device.items():
        for e in c.references:
            ref_device = cell_to_device[e.cell]
            ref = ComponentReference(component=ref_device)

            D._register_reference(ref)
            D._references.append(ref)
            ref._reference = e

    component = cell_to_device[topcell]
    cast(Component, component)

    if read_metadata and metadata_filepath.exists():
        logger.info(f"Read YAML metadata from {metadata_filepath}")
        metadata = OmegaConf.load(metadata_filepath)

        if "settings" in metadata:
            component.settings = Settings(**OmegaConf.to_container(metadata.settings))

        if "ports" in metadata:
            for port_name, port in metadata.ports.items():
                if port_name not in component.ports:
                    component.add_port(
                        name=port_name,
                        center=port.center,
                        width=port.width,
                        orientation=port.orientation,
                        layer=tuple(port.layer),
                        port_type=port.port_type,
                    )

    component.info.update(**kwargs)
    component.imported_gds = True
    return component


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.mzi()
    gdspath = c.write_oas()

    c2 = import_gds(gdspath)

    # gdspath = CONFIG["gdsdir"] / "mzi2x2.gds"
    # c = import_gds(gdspath, flatten=True, name="TOP")
    # c.settings = {}
    # print(clean_value_name(c))
    # c = import_gds(gdspath, flatten=False, polarization="te")
    # c = import_gds("/home/jmatres/gdsfactory/gdsfactory/gdsdiff/gds_diff_git.py")
    # print(c.hash_geometry())
    c.show(show_ports=True)

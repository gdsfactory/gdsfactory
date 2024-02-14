from __future__ import annotations

from pathlib import Path

import gdstk
import numpy as np
import orjson
from omegaconf import OmegaConf

from gdsfactory.cell import cell_import_gds
from gdsfactory.component import CellSettings, Component, Info
from gdsfactory.component_reference import ComponentReference
from gdsfactory.config import logger
from gdsfactory.typings import Callable


@cell_import_gds
def import_gds(
    gdspath: str | Path,
    cellname: str | None = None,
    gdsdir: str | Path | None = None,
    read_metadata: bool = False,
    read_metadata_json: bool = False,
    keep_name_short: bool = False,
    unique_names: bool = True,
    max_name_length: int = 250,
    post_process: Callable | None = None,
    **kwargs,
) -> Component:
    """Returns a Component from a GDS file.

    appends $ with a number to the name if the cell name is on CACHE

    Args:
        gdspath: path of GDS file.
        cellname: cell of the name to import. None imports top cell.
        gdsdir: optional GDS directory.
        read_metadata: loads metadata (ports, settings) if it exists in YAML format.
        read_metadata_json: loads metadata (ports, settings) if it exists in JSON format.
        keep_name_short: appends a hash to a shortened component name.
        unique_names: appends $ with a number to the name if the cell name is on CACHE. \
                This avoids name collisions when importing multiple times the same cell name.
        max_name_length: maximum length of the name.
        post_process: function to post process the component after importing.
        kwargs: extra to add to component.info (polarization, wavelength ...).
    """
    gdspath = Path(gdsdir) / Path(gdspath) if gdsdir else Path(gdspath)
    if not gdspath.exists():
        raise FileNotFoundError(f"No file {str(gdspath)!r} found")

    metadata_filepath = gdspath.with_suffix(".yml")
    metadata_json_filepath = gdspath.with_suffix(".json")

    if gdspath.suffix.lower() == ".gds":
        gdsii_lib = gdstk.read_gds(str(gdspath))
    elif gdspath.suffix.lower() == ".oas":
        gdsii_lib = gdstk.read_oas(str(gdspath))
    else:
        raise ValueError(f"gdspath.suffix {gdspath.suffix!r} not .gds or .oas")

    top_level_cells = gdsii_lib.top_level()
    top_cellnames = [c.name for c in top_level_cells]

    if not top_cellnames:
        raise ValueError(f"no top cells found in {str(gdspath)!r}")

    D_list = []
    cell_name_to_component = {}
    cell_to_component = {}

    # create a new Component for each gdstk Cell
    for c in gdsii_lib.cells:
        D = Component()
        D._cell = c
        if not keep_name_short:
            max_name_length = 10000000000000
        D.rename(c.name, cache=unique_names, max_name_length=max_name_length)

        cell_name_to_component[c.name] = D
        cell_to_component[c] = D
        D_list += [D]

    cellnames = list(cell_name_to_component.keys())
    if cellname:
        if cellname not in cell_name_to_component:
            raise ValueError(
                f"cell {cellname!r} is not in file {gdspath} with cells {cellnames}"
            )
    elif len(top_level_cells) == 1:
        cellname = top_level_cells[0].name
    elif len(top_level_cells) > 1:
        raise ValueError(
            f"import_gds() There are multiple top-level cells in {gdspath!r}, "
            f"you must specify `cellname` to select of one of them among {cellnames}"
        )

    # create a new ComponentReference for each gdstk CellReference
    for c, D in cell_to_component.items():
        for e in c.references:
            ref_device = cell_to_component[e.cell]
            ref = ComponentReference(
                component=ref_device,
                origin=e.origin,
                rotation=e.rotation,
                magnification=e.magnification,
                x_reflection=e.x_reflection,
                columns=e.repetition.columns or 1,
                rows=e.repetition.rows or 1,
                spacing=e.repetition.spacing,
                v1=e.repetition.v1,
                v2=e.repetition.v2,
            )
            D._register_reference(ref)
            D._references.append(ref)
            ref._reference = e

    component = cell_name_to_component[cellname]

    if read_metadata and metadata_filepath.exists():
        logger.info(f"Read YAML metadata from {metadata_filepath}")
        metadata = OmegaConf.load(metadata_filepath)

        if "settings" in metadata:
            settings = OmegaConf.to_container(metadata.settings)
            if settings:
                component.settings = CellSettings(**settings)
        if "info" in metadata:
            info = OmegaConf.to_container(metadata.info)
            if info:
                component.info = Info(**info)
        if "function" in metadata:
            component.function_name = metadata.function
        if "module" in metadata:
            component.module = metadata.module

        if "ports" in metadata:
            for port_name, port in metadata.ports.items():
                if port_name not in component.ports:
                    component.add_port(
                        name=port_name,
                        center=np.array(port.center, dtype="float64"),
                        width=port.width,
                        orientation=port.orientation,
                        layer=tuple(port.layer),
                        port_type=port.port_type,
                    )

    if read_metadata_json and metadata_json_filepath.exists():
        logger.info(f"Read JSON metadata from {metadata_json_filepath}")
        metadata = orjson.loads(open(metadata_json_filepath, "rb").read())

        if "settings" in metadata:
            if metadata.get("settings", {}):
                component.settings = CellSettings(**metadata["settings"])

        if "info" in metadata:
            if metadata["info"]:
                component.info = Info(**metadata["info"])
        if "function" in metadata:
            component.function_name = metadata["function"]
        if "module" in metadata:
            component.module = metadata["module"]

        if "ports" in metadata:
            for port_name, port in metadata["ports"].items():
                if port_name not in component.ports:
                    component.add_port(
                        name=port_name,
                        center=np.array(port["center"], dtype="float64"),
                        width=port["width"],
                        orientation=port["orientation"],
                        layer=tuple(port["layer"]),
                        port_type=port["port_type"],
                    )

    for k, v in kwargs.items():
        component.info[k] = v
    component.imported_gds = True
    if post_process:
        post_process(component)
    return component


def import_gds_raw(gdspath, top_cellname: str | None = None):
    if not top_cellname:
        if gdspath.suffix.lower() == ".gds":
            gdsii_lib = gdstk.read_gds(str(gdspath))
        elif gdspath.suffix.lower() == ".oas":
            gdsii_lib = gdstk.read_oas(str(gdspath))
        top_level_cells = gdsii_lib.top_level()
        top_cellnames = [c.name for c in top_level_cells]
        top_cellname = top_cellnames[0]

    cells = gdstk.read_rawcells(gdspath)
    c = Component(name=top_cellname)
    c._cell = cells.pop(top_cellname)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.Component(name="a" * 250)
    # _ = c << gf.components.mzi()
    c = gf.c.straight()
    gdspath = c.write_gds("a.gds", with_metadata=True)

    # c = import_gds(gdspath)
    c = import_gds("a.gds", read_metadata=True)
    c.show(show_ports=False)

    # gdspath = PATH.gdsdir / "mzi2x2.gds"
    # c = import_gds(gdspath, flatten=True, name="TOP")
    # c.settings = {}
    # print(clean_value_name(c))
    # c = import_gds(gdspath, flatten=False, polarization="te")
    # c = import_gds("/home/jmatres/gdsfactory/gdsfactory/gdsdiff/gds_diff_git.py")
    # print(c.hash_geometry())

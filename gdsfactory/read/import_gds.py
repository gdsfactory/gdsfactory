from pathlib import Path
from typing import Optional, Union, cast

import gdspy
import numpy as np
from omegaconf import OmegaConf
from phidl.device_layout import CellArray

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.component_reference import ComponentReference
from gdsfactory.config import CONFIG, logger
from gdsfactory.name import get_name_short
from gdsfactory.snap import snap_to_grid


@cell
def import_gds(
    gdspath: Union[str, Path],
    cellname: Optional[str] = None,
    snap_to_grid_nm: Optional[int] = None,
    gdsdir: Optional[Union[str, Path]] = None,
    read_metadata: bool = True,
    hashed_name: bool = True,
    **kwargs,
) -> Component:
    """Returns a Componenent from a GDS file.

    Adapted from phidl/geometry.py

    if any cell names are found on the component CACHE we append a $ with a
    number to the name

    Args:
        gdspath: path of GDS file.
        cellname: cell of the name to import (None) imports top cell.
        snap_to_grid_nm: snap to different nm grid (does not snap if False)
        gdsdir: optional GDS directory.
        read_metadata: loads metadata if it exists.
        hashed_name: appends a hash to a shortened component name
        kwargs: extra info for the imported component (polarization, wavelength ...).
    """
    gdspath = Path(gdsdir) / Path(gdspath) if gdsdir else Path(gdspath)
    if not gdspath.exists():
        raise FileNotFoundError(f"No file {gdspath!r} found")

    metadata_filepath = gdspath.with_suffix(".yml")

    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(str(gdspath))
    top_level_cells = gdsii_lib.top_level()
    cellnames = [c.name for c in top_level_cells]

    if cellname is not None:
        if cellname not in gdsii_lib.cells:
            raise ValueError(
                f"cell {cellname!r} is not in file {gdspath} with cells {cellnames}"
            )
        topcell = gdsii_lib.cells[cellname]
    elif len(top_level_cells) == 1:
        topcell = top_level_cells[0]
    elif len(top_level_cells) > 1:
        raise ValueError(
            f"import_gds() There are multiple top-level cells in {gdspath!r}, "
            f"you must specify `cellname` to select of one of them among {cellnames}"
        )

    D_list = []
    cell_to_device = {}
    for c in gdsii_lib.cells.values():
        D = Component(name=c.name)
        D.paths = c.paths
        D.polygons = c.polygons
        D.references = c.references
        D.name = c.name
        for label in c.labels:
            rotation = label.rotation
            if rotation is None:
                rotation = 0
            label_ref = D.add_label(
                text=label.text,
                position=np.asfarray(label.position),
                magnification=label.magnification,
                rotation=rotation * 180 / np.pi,
                layer=(label.layer, label.texttype),
            )
            label_ref.anchor = label.anchor

        if hashed_name:
            D.name = get_name_short(D.name)

        cell_to_device[c] = D
        D_list += [D]

    for D in D_list:
        # First convert each reference so it points to the right Device
        converted_references = []
        for e in D.references:
            ref_device = cell_to_device[e.ref_cell]
            if isinstance(e, gdspy.CellReference):
                dr = ComponentReference(
                    component=ref_device,
                    origin=e.origin,
                    rotation=e.rotation,
                    magnification=e.magnification,
                    x_reflection=e.x_reflection,
                )
                dr.owner = D
                converted_references.append(dr)
            elif isinstance(e, gdspy.CellArray):
                dr = CellArray(
                    device=ref_device,
                    columns=e.columns,
                    rows=e.rows,
                    spacing=e.spacing,
                    origin=e.origin,
                    rotation=e.rotation,
                    magnification=e.magnification,
                    x_reflection=e.x_reflection,
                )
                dr.owner = D
                converted_references.append(dr)
        D.references = converted_references

        # Next convert each Polygon
        temp_polygons = list(D.polygons)
        D.polygons = []
        for p in temp_polygons:
            if snap_to_grid_nm:
                points_on_grid = snap_to_grid(p.polygons[0], nm=snap_to_grid_nm)
                p = gdspy.Polygon(
                    points_on_grid, layer=p.layers[0], datatype=p.datatypes[0]
                )
            D.add_polygon(p)
    component = cell_to_device[topcell]
    cast(Component, component)

    if read_metadata and metadata_filepath.exists():
        logger.info(f"Read YAML metadata from {metadata_filepath}")
        metadata = OmegaConf.load(metadata_filepath)

        if "settings" in metadata:
            component.settings = OmegaConf.to_container(metadata.settings)

        if "ports" in metadata:
            for port_name, port in metadata.ports.items():
                if port_name not in component.ports:
                    component.add_port(
                        name=port_name,
                        midpoint=port.midpoint,
                        width=port.width,
                        orientation=port.orientation,
                        layer=tuple(port.layer),
                        port_type=port.port_type,
                    )

    component.info.update(**kwargs)
    component.imported_gds = True
    return component


if __name__ == "__main__":

    gdspath = CONFIG["gdsdir"] / "mzi2x2.gds"
    # c = import_gds(gdspath, snap_to_grid_nm=5, flatten=True, name="TOP")
    # c.settings = {}
    # print(clean_value_name(c))
    c = import_gds(gdspath, snap_to_grid_nm=5, flatten=False, polarization="te")
    c.show()

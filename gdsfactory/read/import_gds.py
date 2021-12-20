from pathlib import Path
from typing import Callable, Optional, Union, cast

import gdspy
import numpy as np
from omegaconf import OmegaConf
from phidl.device_layout import CellArray, DeviceReference

from gdsfactory.cell import cell_without_validator
from gdsfactory.component import Component
from gdsfactory.config import CONFIG, logger
from gdsfactory.snap import snap_to_grid


def import_gds(
    gdspath: Union[str, Path],
    cellname: Optional[str] = None,
    flatten: bool = False,
    snap_to_grid_nm: Optional[int] = None,
    name: Optional[str] = None,
    decorator: Optional[Callable] = None,
    max_name_length: int = 32,
    **kwargs,
) -> Component:
    """Returns a Componenent from a GDS file.

    Adapted from phidl/geometry.py

    Args:
        gdspath: path of GDS file
        cellname: cell of the name to import (None) imports top cell
        flatten: if True returns flattened (no hierarchy)
        snap_to_grid_nm: snap to different nm grid (does not snap if False)
        name: Optional name
        decorator: function to apply over the imported gds
        max_name_length: can truncate the name of the cell before importing it
        kwargs: component.info
    """
    gdspath = Path(gdspath)
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
                f"cell {cellname} is not in file {gdspath} with cells {cellnames}"
            )
        topcell = gdsii_lib.cells[cellname]
    elif cellname is None and len(top_level_cells) == 1:
        topcell = top_level_cells[0]
    elif cellname is None and len(top_level_cells) > 1:
        raise ValueError(
            f"import_gds() There are multiple top-level cells in {gdspath}, "
            f"you must specify `cellname` to select of one of them among {cellnames}"
        )
    if flatten:
        component = Component(name=name or cellname or cellnames[0])
        polygons = topcell.get_polygons(by_spec=True)

        for layer_in_gds, polys in polygons.items():
            component.add_polygon(polys, layer=layer_in_gds)

    else:
        D_list = []
        c2dmap = {}
        for c in gdsii_lib.cells.values():
            D = Component(name=c.name)
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
            c2dmap.update({c: D})
            D_list += [D]

        for D in D_list:
            # First convert each reference so it points to the right Device
            converted_references = []
            for e in D.references:
                ref_device = c2dmap[e.ref_cell]
                if isinstance(e, gdspy.CellReference):
                    dr = DeviceReference(
                        device=ref_device,
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
            # temp_polygons = list(D.polygons)
            # D.polygons = []
            # for p in temp_polygons:
            #     D.add_polygon(p)

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
        component = c2dmap[topcell]
        cast(Component, component)

    if decorator:
        component_new = decorator(component)
        component = component_new or component
    if flatten:
        component.flatten()

    component.info.update(**kwargs)
    component.name = name or component.name

    component = cell_without_validator(lambda: component)(
        name=component.name, max_name_length=max_name_length, autoname=False
    )

    if metadata_filepath.exists():
        logger.info(f"Read YAML metadata from {metadata_filepath}")
        metadata = OmegaConf.load(metadata_filepath)

        for port_name, port in metadata.ports.items():
            if port_name not in component.ports:
                component.add_port(
                    name=port_name,
                    midpoint=port.midpoint,
                    width=port.width,
                    orientation=port.orientation,
                    layer=port.layer,
                    port_type=port.port_type,
                )

        component.info = metadata.info
    return component


if __name__ == "__main__":
    gdspath = CONFIG["gdsdir"] / "mzi2x2.gds"
    # c = import_gds(gdspath, snap_to_grid_nm=5, flatten=True, name="TOP")
    c = import_gds(gdspath, snap_to_grid_nm=5, flatten=True)
    print(c)
    c.show()

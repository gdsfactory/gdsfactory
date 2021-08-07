import json
import pathlib
from pathlib import Path
from typing import Iterable, Optional, Union, cast

import gdspy
import numpy as np
from phidl.device_layout import CellArray, DeviceReference

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import auto_rename_ports, read_port_markers
from gdsfactory.types import Layer, PathType


def add_ports_from_markers_inside(*args, **kwargs) -> None:
    """markers inside the device"""
    return add_ports_from_markers_center(inside=True, *args, **kwargs)


def add_ports_from_markers_square(
    component: Component,
    layer: Layer = gf.LAYER.PORTE,
    port_layer: Optional[Layer] = None,
    port_type: "str" = "dc",
    orientation: Optional[int] = 90,
    min_pin_area_um2: float = 0,
    max_pin_area_um2: float = 150 * 150,
    pin_extra_width: float = 0.0,
    port_names: Optional[Iterable[str]] = None,
) -> None:
    """add ports from markers in port_layer

    adds ports at the marker center

    squared

    Args:
        component: to read polygons from and to write ports to
        layer: for port markers
        port_layer: for the new created port
        port_type: electrical, dc, optical
        orientation: orientation in degrees
            90: north, 0: east, 180: west, 270: south
        min_pin_area_um2: ignores pins with area smaller than min_pin_area_um2
        max_pin_area_um2: ignore pins for area above certain size
        pin_extra_width: 2*offset from pin to straight
        port_names: names of the ports (defaults to f"{port_type}_{i}")

    """
    port_markers = read_port_markers(component, [layer])
    port_names = None or [f"{port_type}_{i}" for i in range(len(port_markers.polygons))]
    port_layer = port_layer or layer

    for port_name, p in zip(port_names, port_markers.polygons):
        dy = gf.snap.snap_to_grid(p.ymax - p.ymin)
        dx = gf.snap.snap_to_grid(p.xmax - p.xmin)
        x = p.x
        y = p.y
        if dx == dy and max_pin_area_um2 > dx * dy > min_pin_area_um2:
            component.add_port(
                port_name,
                midpoint=(x, y),
                width=dx - pin_extra_width,
                orientation=orientation,
                port_type=port_type,
                layer=port_layer,
            )


def add_ports_from_markers_center(
    component: Component,
    layer: Layer = gf.LAYER.PORT,
    port_layer: Optional[Layer] = None,
    port_type: "str" = "optical",
    inside: bool = False,
    tol: float = 0.1,
    pin_extra_width: float = 0.0,
    min_pin_area_um2: Optional[float] = None,
    max_pin_area_um2: float = 150.0 * 150.0,
    skip_square_ports: bool = True,
    xcenter: Optional[float] = None,
    ycenter: Optional[float] = None,
) -> None:
    """add ports from polygons in certain layers

    markers at port center, so half of the marker goes inside and half ouside the port. Works only for rectangular pins.

    guess orientation of the port by looking at the xcenter and ycenter (which default to the component center)

    Args:
        component: to read polygons from and to write ports to
        layer: GDS layer for maker [int, int]
        port_layer: for the new created port
        port_type: optical, dc, rf
        inside: True-> markers  inside. False-> markers at center
        tol: tolerance for comparing how rectangular is the pin
        pin_extra_width: 2*offset from pin to straight
        min_pin_area_um2: ignores pins with area smaller than min_pin_area_um2
        max_pin_area_um2: ignore pins for area above certain size
        skip_square_ports: skips square ports (hard to guess orientation)
        xcenter: for guessing orientation of rectangular ports
        ycenter: for guessing orientation of rectangular ports

    For the default center case (inside=False)

    .. code::
           _______________
          |               |
          |               |
         |||             |||____  | pin_extra_width/2 > 0
         |||             |||
         |||             |||____
         |||             |||
          |      __       |
          |_______________|
                 __


    For the inside case (inside=True)

    .. code::
           _______________
          |               |
          |               |
          |               |
          | |             |
          | |             |
          |      __       |
          |               |
          |_______________|



    dx < dy: port is east or west
        x > xc: east
        x < xc: west

    dx > dy: port is north or south
        y > yc: north
        y < yc: south

    dx = dy
        x > xc: east
        x < xc: west

    """
    xc = xcenter or component.x
    yc = ycenter or component.y
    xmax = component.xmax
    xmin = component.xmin
    ymax = component.ymax
    ymin = component.ymin

    port_markers = read_port_markers(component, layers=(layer,))
    port_layer = port_layer or layer

    for i, p in enumerate(port_markers.polygons):
        port_name = f"{port_type}_{i}"
        dy = p.ymax - p.ymin
        dx = p.xmax - p.xmin
        x = p.x
        y = p.y
        if min_pin_area_um2 and dx * dy <= min_pin_area_um2:
            continue

        if max_pin_area_um2 and dx * dy > max_pin_area_um2:
            continue

        # skip square ports as they have no clear orientation
        if skip_square_ports and gf.snap.snap_to_grid(dx) == gf.snap.snap_to_grid(dy):
            continue
        pxmax = p.xmax
        pxmin = p.xmin
        pymax = p.ymax
        pymin = p.ymin

        if dx < dy and x > xc:  # east
            orientation = 0
            width = dy
            if inside:
                x = p.xmax
        elif dx < dy and x < xc:  # west
            orientation = 180
            width = dy
            if inside:
                x = p.xmin
        elif dx > dy and y > yc:  # north
            orientation = 90
            width = dx
            if inside:
                y = p.ymax
        elif dx > dy and y < yc:  # south
            orientation = 270
            width = dx
            if inside:
                y = p.ymin

        # port markers have same width and height
        # check which edge (E, W, N, S) they are closer to
        elif pxmax > xmax - tol:  # east
            orientation = 0
            width = dy
            x = p.xmax
        elif pxmin < xmin + tol:  # west
            orientation = 180
            width = dy
            x = p.xmin
        elif pymax > ymax - tol:  # north
            orientation = 90
            width = dx
            y = p.ymax
        elif pymin < ymin + tol:  # south
            orientation = 270
            width = dx
            y = p.ymin

        component.add_port(
            name=port_name,
            midpoint=(x, y),
            width=width - pin_extra_width,
            orientation=orientation,
            port_type=port_type,
            layer=port_layer,
        )


# pytype: disable=bad-return-type
def import_gds(
    gdspath: Union[str, Path],
    cellname: Optional[str] = None,
    flatten: bool = False,
    snap_to_grid_nm: Optional[int] = None,
) -> Component:
    """Returns a Componenent from a GDS file.

    Adapted from phidl/geometry.py

    Args:
        gdspath: path of GDS file
        cellname: cell of the name to import (None) imports top cell
        flatten: if True returns flattened (no hierarchy)
        snap_to_grid_nm: snap to different nm grid (does not snap if False)

    """
    gdspath = Path(gdspath)
    if not gdspath.exists():
        raise FileNotFoundError(f"No file {gdspath} found")
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

    if not flatten:
        D_list = []
        c2dmap = {}
        for cell in gdsii_lib.cells.values():
            D = Component(name=cell.name)
            D.polygons = cell.polygons
            D.references = cell.references
            D.name = cell.name
            for label in cell.labels:
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
            c2dmap.update({cell: D})
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
                    points_on_grid = gf.snap.snap_to_grid(
                        p.polygons[0], nm=snap_to_grid_nm
                    )
                    p = gdspy.Polygon(
                        points_on_grid, layer=p.layers[0], datatype=p.datatypes[0]
                    )
                D.add_polygon(p)
        component = c2dmap[topcell]
        cast(Component, component)
        return component
    if flatten:
        component = Component()
        polygons = topcell.get_polygons(by_spec=True)

        for layer_in_gds, polys in polygons.items():
            component.add_polygon(polys, layer=layer_in_gds)
        return component


def write_top_cells(gdspath: Union[str, Path], **kwargs) -> None:
    """Writes each top level cells into separate GDS file."""
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
    gdspath: Union[str, Path],
    dirpath: Optional[Union[str, Path]] = None,
) -> None:
    """Writes each top level cells into separate GDS file.

    Args:
        gdspath: to read cells from
        dirpath: directory path to store gds cells
    """
    gdspath = dirpath or pathlib.Path(gdspath)
    dirpath = dirpath or gdspath.parent / "gds"
    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(gdspath)
    top_level_cells = gdsii_lib.top_level()
    cellnames = [c.name for c in top_level_cells]

    for cellname in cellnames:
        component = import_gds(gdspath, cellname=cellname)
        component.write_gds(f"{dirpath/cellname}.gds")
        write_cells_from_component(component=component, dirpath=dirpath)


def write_cells_from_component(
    component: Component, dirpath: Optional[PathType] = None
) -> None:
    """Writes all Component cells recursively.

    Args:
        component:
        dirpath: directory path to write GDS (defaults to CWD)
    """
    dirpath = dirpath or pathlib.Path(__file__).parent.absolute()
    if component.references:
        for ref in component.references:
            component = ref.parent
            component.write_gds(f"{dirpath/component.name}.gds")
            write_cells_from_component(component=component, dirpath=dirpath)
    else:
        component.write_gds(f"{dirpath/component.name}.gds")


def add_settings_from_label(component: Component) -> None:
    """Adds settings from label."""
    for label in component.labels:
        if label.text.startswith("settings="):
            d = json.loads(label.text[9:])
            component.settings = d.pop("settings", {})
            for k, v in d.items():
                setattr(component, k, v)


def _demo_optical() -> None:
    """Demo. See equivalent test in tests/import_gds_markers.py"""
    # c  =  gf.components.mmi1x2()
    # for p in c.ports.values():
    #     print(p)
    # c.show()

    gdspath = gf.CONFIG["gdsdir"] / "mmi1x2.gds"
    c = import_gds(gdspath)
    add_ports_from_markers_center(c)
    auto_rename_ports(c)

    # for p in c.ports.values():
    #     print(p)


def _demo_electrical() -> None:
    """Demo. See equivalent test in tests/import_gds_markers.py"""

    gdspath = gf.CONFIG["gdsdir"] / "mzi2x2.gds"
    c = import_gds(gdspath)
    add_ports_from_markers_center(c)
    auto_rename_ports(c)
    c.show()

    for p in c.ports.values():
        print(p)


def _demo_import_gds_markers() -> None:
    import gdsfactory as gf

    name = "mmi1x2"
    gdspath = gf.CONFIG["gdsdir"] / f"{name}.gds"
    c = import_gds(gdspath)
    add_ports_from_markers_center(c)
    assert len(c.ports) == 3
    return c


if __name__ == "__main__":
    c = _demo_import_gds_markers()

    gdspath = gf.CONFIG["gdslib"] / "gds" / "mzi2x2.gds"
    c = import_gds(gdspath, snap_to_grid_nm=5)
    print(c)
    c.show()

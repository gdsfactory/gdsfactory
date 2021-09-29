import json
import pathlib
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, cast

import gdspy
import numpy as np
from phidl.device_layout import CellArray, DeviceReference

from gdsfactory.component import Component
from gdsfactory.config import CONFIG
from gdsfactory.port import (
    Port,
    auto_rename_ports,
    read_port_markers,
    sort_ports_clockwise,
)
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import Layer, PathType


def add_ports_from_markers_square(
    component: Component,
    pin_layer: Layer = (69, 0),
    port_layer: Optional[Layer] = None,
    orientation: Optional[int] = 90,
    min_pin_area_um2: float = 0,
    max_pin_area_um2: float = 150 * 150,
    pin_extra_width: float = 0.0,
    port_names: Optional[Tuple[str, ...]] = None,
    port_name_prefix: str = "o",
) -> Component:
    """add ports from markers center in port_layer

    squared

    Args:
        component: to read polygons from and to write ports to
        pin_layer: for port markers
        port_layer: for the new created port
        orientation: in degrees 90: north, 0: east, 180: west, 270: south
        min_pin_area_um2: ignores pins with area smaller than min_pin_area_um2
        max_pin_area_um2: ignore pins for area above certain size
        pin_extra_width: 2*offset from pin to straight
        port_names: names of the ports (defaults to {i})

    """
    port_markers = read_port_markers(component, [pin_layer])
    port_names = port_names or [
        f"{port_name_prefix}{i+1}" for i in range(len(port_markers.polygons))
    ]
    layer = port_layer or pin_layer

    for port_name, p in zip(port_names, port_markers.polygons):
        dy = snap_to_grid(p.ymax - p.ymin)
        dx = snap_to_grid(p.xmax - p.xmin)
        x = p.x
        y = p.y
        if dx == dy and max_pin_area_um2 > dx * dy > min_pin_area_um2:
            component.add_port(
                port_name,
                midpoint=(x, y),
                width=dx - pin_extra_width,
                orientation=orientation,
                layer=layer,
            )
    return component


def add_ports_from_markers_center(
    component: Component,
    pin_layer: Layer = (1, 10),
    port_layer: Optional[Layer] = None,
    inside: bool = False,
    tol: float = 0.1,
    pin_extra_width: float = 0.0,
    min_pin_area_um2: Optional[float] = None,
    max_pin_area_um2: float = 150.0 * 150.0,
    skip_square_ports: bool = True,
    xcenter: Optional[float] = None,
    ycenter: Optional[float] = None,
    port_name_prefix: str = "",
    port_type: str = "optical",
) -> Component:
    """Add ports from rectangular pin markers.

    markers at port center, so half of the marker goes inside and half ouside the port.

    guess port orientation from the component center

    Args:
        component: to read polygons from and to write ports to
        pin_layer: GDS layer for maker [int, int]
        port_layer: for the new created port
        inside: True-> markers  inside. False-> markers at center
        tol: tolerance for comparing how rectangular is the pin
        pin_extra_width: 2*offset from pin to straight
        min_pin_area_um2: ignores pins with area smaller than min_pin_area_um2
        max_pin_area_um2: ignore pins for area above certain size
        skip_square_ports: skips square ports (hard to guess orientation)
        xcenter: for guessing orientation of rectangular ports
        ycenter: for guessing orientation of rectangular ports
        port_name_prefix: o for optical ports (o1, o2, o3)

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

    port_markers = read_port_markers(component, layers=(pin_layer,))
    layer = port_layer or pin_layer
    port_locations = []

    ports = {}

    for i, p in enumerate(port_markers.polygons):
        port_name = f"{port_name_prefix}{i+1}" if port_name_prefix else i
        dy = p.ymax - p.ymin
        dx = p.xmax - p.xmin
        x = p.x
        y = p.y
        if min_pin_area_um2 and dx * dy <= min_pin_area_um2:
            continue

        if max_pin_area_um2 and dx * dy > max_pin_area_um2:
            continue

        # skip square ports as they have no clear orientation
        if skip_square_ports and snap_to_grid(dx) == snap_to_grid(dy):
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

        x = snap_to_grid(x)
        y = snap_to_grid(y)
        width = np.round(width - pin_extra_width, 3)

        if (x, y) not in port_locations:
            port_locations.append((x, y))
            ports[port_name] = Port(
                name=port_name,
                midpoint=(x, y),
                width=width,
                orientation=orientation,
                layer=layer,
                port_type=port_type,
            )

    ports = sort_ports_clockwise(ports)

    for port_name, port in ports.items():
        component.add_port(name=port_name, port=port)
    return component


add_ports_from_markers_inside = partial(add_ports_from_markers_center, inside=True)


def add_ports_from_labels(
    component: Component,
    port_width: float,
    port_layer: Layer,
    xcenter: Optional[float] = None,
    port_name_prefix: str = "o",
    port_type: str = "optical",
) -> Component:
    """Add ports from labels.
    Assumes that all ports have a label at the port center.
    """
    xc = xcenter or component.x
    yc = component.y
    for i, label in enumerate(component.labels):
        x, y = label.position
        port_name = f"{port_name_prefix}{i+1}" if port_name_prefix else i
        if x > xc:  # east
            orientation = 0
        elif x < xc:  # west
            orientation = 180
        elif y > yc:  # north
            orientation = 90
        elif y < yc:  # south
            orientation = 270

        component.add_port(
            name=port_name,
            midpoint=(x, y),
            width=port_width,
            orientation=orientation,
            port_type=port_type,
            layer=port_layer,
        )
    return component


# pytype: disable=bad-return-type
@lru_cache(maxsize=None)
def import_gds(
    gdspath: Union[str, Path],
    cellname: Optional[str] = None,
    flatten: bool = False,
    snap_to_grid_nm: Optional[int] = None,
    decorator: Optional[Callable] = None,
    **kwargs,
) -> Component:
    """Returns a Componenent from a GDS file.

    Adapted from phidl/geometry.py

    Args:
        gdspath: path of GDS file
        cellname: cell of the name to import (None) imports top cell
        flatten: if True returns flattened (no hierarchy)
        snap_to_grid_nm: snap to different nm grid (does not snap if False)
        **kwargs
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
    if flatten:
        component = Component()
        polygons = topcell.get_polygons(by_spec=True)

        for layer_in_gds, polys in polygons.items():
            component.add_polygon(polys, layer=layer_in_gds)

    else:
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
                    points_on_grid = snap_to_grid(p.polygons[0], nm=snap_to_grid_nm)
                    p = gdspy.Polygon(
                        points_on_grid, layer=p.layers[0], datatype=p.datatypes[0]
                    )
                D.add_polygon(p)
        component = c2dmap[topcell]
        cast(Component, component)
    for key, value in kwargs.items():
        setattr(component, key, value)
    if decorator:
        decorator(component)
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

    gdspath = CONFIG["gdsdir"] / "mmi1x2.gds"
    c = import_gds(gdspath)
    add_ports_from_markers_center(c)
    auto_rename_ports(c)

    # for p in c.ports.values():
    #     print(p)


def _demo_electrical() -> None:
    """Demo. See equivalent test in tests/import_gds_markers.py"""

    gdspath = CONFIG["gdsdir"] / "mzi2x2.gds"
    c = import_gds(gdspath)
    add_ports_from_markers_center(c)
    auto_rename_ports(c)
    c.show()

    for p in c.ports.values():
        print(p)


def _demo_import_gds_markers() -> None:

    name = "mmi1x2"
    # name = "mzi2x2"
    gdspath = CONFIG["gdsdir"] / f"{name}.gds"
    c = import_gds(gdspath)
    add_ports_from_markers_center(c)
    assert len(c.ports) == 3
    return c


if __name__ == "__main__":
    c = _demo_import_gds_markers()

    gdspath = CONFIG["gds"] / "mzi2x2.gds"
    c = import_gds(gdspath, snap_to_grid_nm=5)
    print(c)
    c.show()

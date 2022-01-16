import warnings
from functools import partial
from typing import Optional, Tuple

import numpy as np

from gdsfactory.component import Component
from gdsfactory.port import Port, read_port_markers, sort_ports_clockwise
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import Layer


def add_ports_from_markers_square(
    component: Component,
    pin_layer: Layer = (69, 0),
    port_layer: Optional[Layer] = None,
    orientation: Optional[int] = 90,
    min_pin_area_um2: float = 0,
    max_pin_area_um2: float = 150 * 150,
    pin_extra_width: float = 0.0,
    port_names: Optional[Tuple[str, ...]] = None,
    port_name_prefix: Optional[str] = None,
    port_type: str = "optical",
) -> Component:
    """Add ports from square markers at the port center in port_layer

    Args:
        component: to read polygons from and to write ports to.
        pin_layer: for port markers.
        port_layer: for the new created port.
        orientation: in degrees 90: north, 0: east, 180: west, 270: south
        min_pin_area_um2: ignores pins with area smaller than min_pin_area_um2
        max_pin_area_um2: ignore pins for area above certain size
        pin_extra_width: 2*offset from pin to straight
        port_names: names of the ports (defaults to {i})
        port_name_prefix: defaults to 'o' for optical and 'e' for electrical
        port_type: optical, electrical

    """
    port_name_prefix_default = "o" if port_type == "optical" else "e"
    port_name_prefix = port_name_prefix or port_name_prefix_default
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
    skip_square_ports: bool = False,
    xcenter: Optional[float] = None,
    ycenter: Optional[float] = None,
    port_name_prefix: Optional[str] = None,
    port_type: str = "optical",
    auto_rename_ports: bool = True,
) -> Component:
    """Add ports from rectangular pin markers.

    markers at port center, so half of the marker goes inside and half ouside the port.

    guess port orientation from the component center (xcenter)

    Args:
        component: to read polygons from and to write ports to.
        pin_layer: GDS layer for maker [int, int].
        port_layer: for the new created port
        inside: True-> markers  inside. False-> markers at center
        tol: tolerance for comparing how rectangular is the pin
        pin_extra_width: 2*offset from pin to straight
        min_pin_area_um2: ignores pins with area smaller than min_pin_area_um2
        max_pin_area_um2: ignore pins for area above certain size
        skip_square_ports: skips square ports (hard to guess orientation)
        xcenter: for guessing orientation of rectangular ports
        ycenter: for guessing orientation of rectangular ports
        port_name_prefix: defaults to 'o' for optical and 'e' for electrical ports.
        port_type: type of port (optical, electrical ...)
        auto_rename_ports:

    For inside=False the port location is at the middle of the PIN

    .. code::
           _______________
          |               |
          |               |
         |||             |||____  | pin_extra_width/2 > 0
         |||             |||
         |||             |||____
         |||             |||
          |      __       |
          |_____|__|______|
                |__|


    For inside=True all the pin is inside the port

    .. code::
           _______________
          |               |
          |               |
          |_              |
          | |             |
          |_|             |
          |               |
          |      __       |
          |_____|__|______|



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

    port_name_prefix_default = "o" if port_type == "optical" else "e"
    port_name_prefix = port_name_prefix or port_name_prefix_default

    for i, p in enumerate(port_markers.polygons):
        port_name = f"{port_name_prefix}{i+1}" if port_name_prefix else i
        dy = p.ymax - p.ymin
        dx = p.xmax - p.xmin
        x = p.x
        y = p.y

        if min_pin_area_um2 and dx * dy < min_pin_area_um2:
            warnings.warn(f"skipping port with min_pin_area_um2 {dx * dy}")
            continue

        if max_pin_area_um2 and dx * dy > max_pin_area_um2:
            continue

        if skip_square_ports and snap_to_grid(dx) == snap_to_grid(dy):
            warnings.warn("skipping square port with no clear orientation")
            continue

        pxmax = p.xmax
        pxmin = p.xmin
        pymax = p.ymax
        pymin = p.ymin

        if dx < dy and x > xc:  # east
            orientation = 0
            width = dy
            x = p.xmax if inside else p.x
        elif dx < dy and x < xc:  # west
            orientation = 180
            width = dy
            x = p.xmin if inside else p.x
        elif dx > dy and y > yc:  # north
            orientation = 90
            width = dx
            y = p.ymax if inside else p.y
        elif dx > dy and y < yc:  # south
            orientation = 270
            width = dx
            y = p.ymin if inside else p.y

        # square port markers have same width and height
        # check which edge (E, W, N, S) they are closer to

        elif pxmax > xmax - tol:  # east
            orientation = 0
            width = dy
            x = p.xmax if inside else p.x
        elif pxmin < xmin + tol:  # west
            orientation = 180
            width = dy
            x = p.xmin if inside else p.x
        elif pymax > ymax - tol:  # north
            orientation = 90
            width = dx
            y = p.ymax if inside else p.y
        elif pymin < ymin + tol:  # south
            orientation = 270
            width = dx
            y = p.ymin if inside else p.y

        elif pxmax > xc:
            orientation = 0
            width = dy
            x = p.xmax if inside else p.x

        elif pxmax < xc:
            orientation = 180
            width = dy
            x = p.xmin if inside else p.x

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
        if port_name not in component.ports:
            component.add_port(name=port_name, port=port)
    if auto_rename_ports:
        component.auto_rename_ports()
    return component


add_ports_from_markers_inside = partial(add_ports_from_markers_center, inside=True)


def add_ports_from_labels(
    component: Component,
    port_width: float,
    port_layer: Layer,
    xcenter: Optional[float] = None,
    port_name_prefix: Optional[str] = None,
    port_type: str = "optical",
) -> Component:
    """Add ports from labels.
    Assumes that all ports have a label at the port center.
    because labels do not have width, you have to manually specify the ports width

    Args:
        component: to read polygons from and to write ports to.
        port_width: for ports.
        port_layer: for the new created port.
        xcenter: center of the component, for guessing port orientation.
        port_name_prefix: defaults to 'o' for optical and 'e' for electrical
        port_type: optical, electrical

    """
    port_name_prefix_default = "o" if port_type == "optical" else "e"
    port_name_prefix = port_name_prefix or port_name_prefix_default
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


if __name__ == "__main__":
    pass

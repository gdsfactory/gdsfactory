"""Add ports from pin markers or labels."""
from functools import partial
from typing import Optional, Tuple

import numpy as np
from numpy import arctan2, degrees, isclose

from gdsfactory.component import Component
from gdsfactory.port import Port, read_port_markers, sort_ports_clockwise
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import LayerSpec


def add_ports_from_markers_square(
    component: Component,
    pin_layer: LayerSpec = "DEVREC",
    port_layer: Optional[LayerSpec] = None,
    orientation: Optional[int] = 90,
    min_pin_area_um2: float = 0,
    max_pin_area_um2: float = 150 * 150,
    pin_extra_width: float = 0.0,
    port_names: Optional[Tuple[str, ...]] = None,
    port_name_prefix: Optional[str] = None,
    port_type: str = "optical",
) -> Component:
    """Add ports from square markers at the port center in port_layer.

    Args:
        component: to read polygons from and to write ports to.
        pin_layer: for port markers.
        port_layer: for the new created port.
        orientation: in degrees 90 north, 0 east, 180 west, 270 south.
        min_pin_area_um2: ignores pins with area smaller than min_pin_area_um2.
        max_pin_area_um2: ignore pins for area above certain size.
        pin_extra_width: 2*offset from pin to straight.
        port_names: names of the ports (defaults to {i}).
        port_name_prefix: defaults to 'o' for optical and 'e' for electrical.
        port_type: optical, electrical.
    """
    port_name_prefix_default = "o" if port_type == "optical" else "e"
    port_name_prefix = port_name_prefix or port_name_prefix_default
    port_markers = read_port_markers(component, (pin_layer,))
    port_names = port_names or [
        f"{port_name_prefix}{i+1}" for i in range(len(port_markers.polygons))
    ]
    layer = port_layer or pin_layer

    for port_name, p in zip(port_names, port_markers.polygons):
        dy = snap_to_grid(p.ymax - p.ymin)
        dx = snap_to_grid(p.xmax - p.xmin)
        if dx == dy and max_pin_area_um2 > dx * dy > min_pin_area_um2:
            x = p.x
            y = p.y
            component.add_port(
                port_name,
                center=(x, y),
                width=dx - pin_extra_width,
                orientation=orientation,
                layer=layer,
            )
    return component


def add_ports_from_markers_center(
    component: Component,
    pin_layer: LayerSpec = "PORT",
    port_layer: Optional[LayerSpec] = None,
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
    short_ports: bool = False,
    auto_rename_ports: bool = True,
    debug: bool = False,
) -> Component:
    """Add ports from rectangular pin markers.

    markers at port center, so half of the marker goes inside and half outside the port.

    guess port orientation from the component center (xcenter)

    Args:
        component: to read polygons from and to write ports to.
        pin_layer: GDS layer for maker [int, int].
        port_layer: for the new created port.
        inside: True-> markers  inside. False-> markers at center.
        tol: tolerance for comparing how rectangular is the pin.
        pin_extra_width: 2*offset from pin to straight.
        min_pin_area_um2: ignores pins with area smaller than min_pin_area_um2.
        max_pin_area_um2: ignore pins for area above certain size.
        skip_square_ports: skips square ports (hard to guess orientation).
        xcenter: for guessing orientation of rectangular ports.
        ycenter: for guessing orientation of rectangular ports.
        port_name_prefix: defaults to 'o' for optical and 'e' for electrical ports.
        port_type: type of port (optical, electrical ...).
        short_ports: if the port is on the short side rather than the long side
        auto_rename_ports:
        debug: if True prints ports that are skipped.

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
        port_name = f"{port_name_prefix}{i+1}" if port_name_prefix else str(i)
        dy = p.ymax - p.ymin
        dx = p.xmax - p.xmin
        x = p.x
        y = p.y

        if min_pin_area_um2 and dx * dy < min_pin_area_um2:
            if debug:
                print(f"skipping port at ({x}, {y}) with min_pin_area_um2 {dx * dy}")
            continue

        if max_pin_area_um2 and dx * dy > max_pin_area_um2:
            continue

        if skip_square_ports and snap_to_grid(dx) == snap_to_grid(dy):
            if debug:
                print(f"skipping square port at ({x}, {y})")
            continue

        pxmax = p.xmax
        pxmin = p.xmin
        pymax = p.ymax
        pymin = p.ymin

        orientation = 0

        if dy < dx if short_ports else dx < dy:
            if x > xc:  # east
                orientation = 0
                width = dy
                x = p.xmax if inside else p.x
            elif x < xc:  # west
                orientation = 180
                width = dy
                x = p.xmin if inside else p.x
        elif dy > dx if short_ports else dx > dy:
            if y > yc:  # north
                orientation = 90
                width = dx
                y = p.ymax if inside else p.y
            elif y < yc:  # south
                orientation = 270
                width = dx
                y = p.ymin if inside else p.y

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
                center=(x, y),
                width=width,
                orientation=orientation,
                layer=layer,
                port_type=port_type,
            )

    ports = sort_ports_clockwise(ports)

    for port_name, port in ports.items():
        if port_name in component.ports:
            component_ports = list(component.ports.keys())
            raise ValueError(
                f"port {port_name!r} already in {component_ports}. "
                "You can pass a port_name_prefix to add it with a different name."
            )

        else:
            component.add_port(name=port_name, port=port)
    if auto_rename_ports:
        component.auto_rename_ports()
    return component


add_ports_from_markers_inside = partial(add_ports_from_markers_center, inside=True)


def add_ports_from_labels(
    component: Component,
    port_width: float,
    port_layer: LayerSpec,
    xcenter: Optional[float] = None,
    port_name_prefix: Optional[str] = None,
    port_type: str = "optical",
    get_name_from_label: bool = False,
    layer_label: Optional[LayerSpec] = None,
    fail_on_duplicates: bool = False,
    port_orientation: Optional[float] = None,
    guess_port_orientation: bool = True,
) -> Component:
    """Add ports from labels.

    Assumes that all ports have a label at the port center.
    because labels do not have width, you have to manually specify the ports width

    Args:
        component: to read polygons from and to write ports to.
        port_width: for ports.
        port_layer: for the new created port.
        xcenter: center of the component, for guessing port orientation.
        port_name_prefix: defaults to 'o' for optical and 'e' for electrical.
        port_type: optical, electrical.
        layer_label:
        fail_on_duplicates: raises ValueError for duplicated port names.
            if False adds incremental suffix (1, 2 ...) to port name.
        port_orientation: None for electrical ports.
        guess_port_orientation: assumes right: 0, left: 180, top: 90, bot: 270.
    """
    port_name_prefix_default = "o" if port_type == "optical" else "e"
    port_name_prefix = port_name_prefix or port_name_prefix_default
    xc = xcenter or component.x
    yc = component.y

    port_name_to_index = {}

    for i, label in enumerate(component.labels):
        x, y = label.position

        if layer_label and not (
            layer_label[0] == label.layer and layer_label[1] == label.texttype
        ):
            continue

        if get_name_from_label:
            port_name = label.text
        else:
            port_name = f"{port_name_prefix}{i+1}" if port_name_prefix else i

        orientation = port_orientation

        if guess_port_orientation:
            if x > xc:  # east
                orientation = 0
            elif x < xc:  # west
                orientation = 180
            elif y > yc:  # north
                orientation = 90
            elif y < yc:  # south
                orientation = 270

        if fail_on_duplicates and port_name in component.ports:
            component_ports = list(component.ports.keys())
            raise ValueError(
                f"port {port_name!r} already in {component_ports}. "
                "You can pass a port_name_prefix to add it with a different name."
            )
        if get_name_from_label and port_name in component.ports:
            port_name_to_index[label.text] = (
                port_name_to_index[label.text] + 1
                if label.text in port_name_to_index
                else 1
            )
            port_name = f"{label.text}{port_name_to_index[label.text]}"

        component.add_port(
            name=port_name,
            center=(x, y),
            width=port_width,
            orientation=orientation,
            port_type=port_type,
            layer=port_layer,
        )
    return component


def add_ports_from_siepic_pins(
    component: Component,
    pin_layer_optical: LayerSpec = "PORT",
    port_layer_optical: Optional[LayerSpec] = None,
    pin_layer_electrical: LayerSpec = "PORTE",
    port_layer_electrical: Optional[LayerSpec] = None,
) -> Component:
    """Add ports from SiEPIC-type cells.

    Looks for label, path pairs.

    Args:
        component: component.
        pin_layer_optical: layer for optical pins.
        port_layer_optical: layer for optical ports.
        pin_layer_electrical: layer for electrical pins.
        port_layer_electrical: layer for electrical ports.
    """
    pin_layers = {"optical": pin_layer_optical, "electrical": pin_layer_electrical}

    c = component
    labels = c.get_labels()

    for path in c.paths:
        p1, p2 = path.points

        # Find the center of the path
        center = (p1 + p2) / 2

        # Find the label closest to the pin
        label = None
        for i, l in enumerate(labels):
            if (
                all(isclose(l.position, center))
                or all(isclose(l.position, p1))
                or all(isclose(l.position, p2))
            ):
                label = l
                labels.pop(i)
        if label is None:
            print(f"Warning: label not found for path: ({p1}, {p2})")
            continue
        if pin_layer_optical[0] in path.layers:
            port_type = "optical"
            port_layer = port_layer_optical or None
        elif pin_layer_electrical[0] in path.layers:
            port_type = "electrical"
            port_layer = port_layer_electrical or None
        else:
            continue

        port_name = str(label.text)

        # If the port name is already used, add a number to it
        i = 1
        while port_name in c.ports:
            port_name += f"_{i}"

        angle = round(degrees(arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 360)

        port = Port(
            name=port_name,
            center=center,
            width=path.widths[0][0],
            orientation=angle,
            layer=port_layer or pin_layers[port_type],
            port_type=port_type,
        )

        c.add_port(port)
    return c

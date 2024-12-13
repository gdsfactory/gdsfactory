"""Add ports from pin markers or labels."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import Port, read_port_markers, sort_ports_clockwise
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import AngleInDegrees, LayerSpec


def add_ports_from_markers_square(
    component: Component,
    pin_layer: LayerSpec = "DEVREC",
    port_layer: LayerSpec | None = None,
    orientation: AngleInDegrees | None = 90,
    min_pin_area_um2: float = 0,
    max_pin_area_um2: float | None = 150 * 150,
    pin_extra_width: float = 0.0,
    port_names: Sequence[str] | None = None,
    port_name_prefix: str | None = None,
    port_type: str = "optical",
) -> Component:
    """Add ports from square markers at the port dcenter in port_layer.

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
    port_names = list(
        port_names
        or [f"{port_name_prefix}{i + 1}" for i in range(len(port_markers.polygons))]
    )
    layer = port_layer or pin_layer

    for port_name, p in zip(port_names, port_markers.polygons):
        (xmin, ymin), (xmax, ymax) = p.bounding_box()
        x, y = np.sum(p.bounding_box(), 0) / 2

        dy = snap_to_grid(ymax - ymin)
        dx = snap_to_grid(xmax - xmin)
        width = dx - pin_extra_width

        # Snap to the nearest 2 nm (0.002 µm)
        width = np.round((width - pin_extra_width) / 0.002) * 0.002

        if dx == dy and max_pin_area_um2 > dx * dy > min_pin_area_um2:
            x = x
            y = y
            component.add_port(
                port_name,
                center=(x, y),
                width=width,
                orientation=orientation,
                layer=layer,
            )
    return component


def add_ports_from_markers_center(
    component: Component,
    pin_layer: LayerSpec,
    port_layer: LayerSpec | None = None,
    inside: bool = False,
    tol: float = 0.1,
    pin_extra_width: float = 0.0,
    min_pin_area_um2: float | None = None,
    max_pin_area_um2: float | None = None,
    skip_square_ports: bool = False,
    xcenter: float | None = None,
    ycenter: float | None = None,
    port_name_prefix: str | None = None,
    port_type: str = "optical",
    ports_on_short_side: bool = False,
    auto_rename_ports: bool = True,
    debug: bool = False,
) -> Component:
    """Add ports from pins guessing port orientation from component boundary.

    Args:
        component: to read polygons from and to write ports to.
        pin_layer: layer for pin maker.
        port_layer: for the new created port. Defaults to pin_layer.
        inside: True-> markers  inside. False-> markers at dcenter.
        tol: tolerance area to search ports at component boundaries dxmin, dymin, dxmax, dxmax.
        pin_extra_width: 2*offset from pin to straight.
        min_pin_area_um2: ignores pins with area smaller than min_pin_area_um2.
        max_pin_area_um2: ignore pins for area above certain size.
        skip_square_ports: skips square ports (hard to guess orientation).
        xcenter: for guessing orientation of rectangular ports.
        ycenter: for guessing orientation of rectangular ports.
        port_name_prefix: defaults to 'o' for optical and 'e' for electrical ports.
        port_type: type of port (optical, electrical ...).
        ports_on_short_side: if the port is on the short side rather than the long side.
        auto_rename_ports: if True auto rename ports to avoid duplicates.
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
        dx > xc: east
        dx < xc: west

    dx > dy: port is north or south
        dy > yc: north
        dy < yc: south

    dx = dy
        dx > xc: east
        dx < xc: west
    """
    xc = xcenter or component.dx
    yc = ycenter or component.dy
    dxmax = component.dxmax
    dxmin = component.dxmin
    dymax = component.dymax
    dymin = component.dymin
    dbu = float(component.kcl.dbu)

    layer = port_layer or pin_layer
    port_locations: list[tuple[float, float]] = []

    port_name_prefix_default = "o" if port_type == "optical" else "e"
    port_name_prefix = port_name_prefix or port_name_prefix_default

    pin_layer = gf.get_layer(pin_layer)

    polygons = component.get_polygons(by="index")
    if pin_layer not in polygons:
        warnings.warn(f"no pin layer {pin_layer} found in {component.layers}")
        return component

    port_markers = polygons[pin_layer]
    ports: list[Port] = []

    for i, p in enumerate(port_markers):
        port_name = f"{port_name_prefix}{i + 1}" if port_name_prefix else str(i)
        bbox = p.bbox()
        pxmin, pymin, pxmax, pymax = map(
            float,
            (
                (bbox.left),
                (bbox.bottom),
                (bbox.right),
                (bbox.top),
            ),
        )

        x = (pxmax + pxmin) / 2
        y = (pymin + pymax) / 2
        dy = abs(pymax - pymin)
        dx = abs(pxmax - pxmin)
        dx *= dbu
        dy *= dbu
        x = x * dbu
        y = y * dbu
        pxmax *= dbu
        pymax *= dbu
        pxmin *= dbu
        pymin *= dbu

        if min_pin_area_um2 and dx * dy < min_pin_area_um2:
            if debug:
                print(f"skipping port at ({dx}, {dy}) with min_pin_area_um2 {dx * dy}")
            continue

        if max_pin_area_um2 and dx * dy > max_pin_area_um2:
            continue

        if skip_square_ports and snap_to_grid(dx) == snap_to_grid(dy):
            if debug:
                print(f"skipping square port at ({dx}, {dy})")
            continue

        orientation = -1

        # rectangular ports orientation is easier to detect
        if dy < dx if ports_on_short_side else dx < dy:
            width = dy
            if x > xc:  # east
                orientation = 0
                x = pxmax if inside else x
            else:
                orientation = 180
                x = pxmin if inside else x
        elif dy > dx if ports_on_short_side else dx > dy:
            width = dx
            if y > yc:  # north
                orientation = 90
                y = pymax if inside else y
            else:
                orientation = 270
                y = pymin if inside else y

        elif pxmax > dxmax - tol:  # east
            orientation = 0
            width = dy
            x = pxmax if inside else x
        elif pxmin < dxmin + tol:  # west
            orientation = 180
            width = dy
            x = pxmin if inside else x
        elif pymax > dymax - tol:  # north
            orientation = 90
            width = dx
            y = pymax if inside else y
        elif pymin < dymin + tol:  # south
            orientation = 270
            width = dx
            y = pymin if inside else y

        elif pxmax > xc:
            orientation = 0
            width = dy
            x = pxmax if inside else x

        else:
            orientation = 180
            width = dy
            x = pxmin if inside else x

        if orientation == -1:
            raise ValueError(f"Unable to detect port at ({dx}, {dy})")

        width = width - pin_extra_width

        # Snap to the nearest 2 nm (0.002 µm)
        width = np.round((width - pin_extra_width) / 0.002) * 0.002

        if (x, y) not in port_locations:
            port_locations.append((x, y))
            port = Port(
                name=port_name,
                center=(x, y),
                width=width,
                orientation=orientation,
                layer=layer,
                port_type=port_type,
            )
            ports.append(port)

    ports = sort_ports_clockwise(ports)

    for port in ports:
        _port_name_or_none = port.name
        if port in component.ports:
            component_ports = [p.name for p in component.ports]
            raise ValueError(
                f"port {_port_name_or_none!r} already in {component_ports}. "
                "You can pass a port_name_prefix to add it with a different name."
            )

        else:
            component.add_port(name=port_name, port=port)
    if auto_rename_ports:
        component.auto_rename_ports()
    return component


def add_ports_from_boxes(
    component: Component,
    pin_layer: LayerSpec,
    port_layer: LayerSpec | None = None,
    inside: bool = False,
    tol: float = 0.1,
    pin_extra_width: float = 0.0,
    min_pin_area_um2: float | None = None,
    max_pin_area_um2: float | None = 150.0 * 150.0,
    skip_square_ports: bool = False,
    xcenter: float | None = None,
    ycenter: float | None = None,
    port_name_prefix: str | None = None,
    port_type: str = "optical",
    ports_on_short_side: bool = False,
    auto_rename_ports: bool = True,
    debug: bool = False,
) -> Component:
    """Add ports from pins guessing port orientation from component boundary.

    Args:
        component: to read polygons from and to write ports to.
        pin_layer: layer for pin maker.
        port_layer: for the new created port. Defaults to pin_layer.
        inside: True-> markers  inside. False-> markers at dcenter.
        tol: tolerance area to search ports at component boundaries dxmin, dymin, dxmax, dxmax.
        pin_extra_width: 2*offset from pin to straight.
        min_pin_area_um2: ignores pins with area smaller than min_pin_area_um2.
        max_pin_area_um2: ignore pins for area above certain size.
        skip_square_ports: skips square ports (hard to guess orientation).
        xcenter: for guessing orientation of rectangular ports.
        ycenter: for guessing orientation of rectangular ports.
        port_name_prefix: defaults to 'o' for optical and 'e' for electrical ports.
        port_type: type of port (optical, electrical ...).
        ports_on_short_side: if the port is on the short side rather than the long side.
        auto_rename_ports: if True auto rename ports to avoid duplicates.
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
        dx > xc: east
        dx < xc: west

    dx > dy: port is north or south
        dy > yc: north
        dy < yc: south

    dx = dy
        dx > xc: east
        dx < xc: west
    """
    xc = xcenter or component.dx
    yc = ycenter or component.dy
    dxmax = component.dxmax
    dxmin = component.dxmin
    dymax = component.dymax
    dymin = component.dymin

    layer = port_layer or pin_layer
    port_locations: list[tuple[float, float]] = []

    ports: list[Port] = []
    port_name_prefix_default = "o" if port_type == "optical" else "e"
    port_name_prefix = port_name_prefix or port_name_prefix_default

    pin_layer = gf.get_layer(pin_layer)
    layer = gf.get_layer(layer)

    port_markers = component.get_boxes(layer=pin_layer)
    for i, p in enumerate(port_markers):
        port_name = f"{port_name_prefix}{i + 1}" if port_name_prefix else str(i)
        bbox = p.bbox()
        pxmin, pymin, pxmax, pymax = bbox.left, bbox.bottom, bbox.right, bbox.top

        x = (pxmax + pxmin) / 2
        y = (pymin + pymax) / 2
        dy = abs(pymax - pymin)
        dx = abs(pxmax - pxmin)

        if min_pin_area_um2 and dx * dy < min_pin_area_um2:
            if debug:
                print(f"skipping port at ({dx}, {dy}) with min_pin_area_um2 {dx * dy}")
            continue

        if max_pin_area_um2 and dx * dy > max_pin_area_um2:
            continue

        if skip_square_ports and snap_to_grid(dx) == snap_to_grid(dy):
            if debug:
                print(f"skipping square port at ({dx}, {dy})")
            continue

        orientation = -1

        # rectangular ports orientation is easier to detect
        if dy < dx if ports_on_short_side else dx < dy:
            width = dy
            if x > xc:  # east
                orientation = 0
                x = pxmax if inside else x
            else:
                orientation = 180
                x = pxmin if inside else x
        elif dy > dx if ports_on_short_side else dx > dy:
            width = dx
            if y > yc:  # north
                orientation = 90
                y = pymax if inside else y
            else:
                orientation = 270
                y = pymin if inside else y

        elif pxmax > dxmax - tol:  # east
            orientation = 0
            width = dy
            x = pxmax if inside else x
        elif pxmin < dxmin + tol:  # west
            orientation = 180
            width = dy
            x = pxmin if inside else x
        elif pymax > dymax - tol:  # north
            orientation = 90
            width = dx
            y = pymax if inside else y
        elif pymin < dymin + tol:  # south
            orientation = 270
            width = dx
            y = pymin if inside else y

        elif pxmax > xc:
            orientation = 0
            width = dy
            x = pxmax if inside else x

        else:
            orientation = 180
            width = dy
            x = pxmin if inside else x

        if orientation == -1:
            raise ValueError(
                f"Unable to detect port at ({dx=}, {dy=}, {x=}, {y=}, {xc=}, {yc=}"
            )

        # Snap to the nearest 2 nm (0.002 µm)
        width = np.round((width - pin_extra_width) / 0.002) * 0.002

        if (x, y) not in port_locations:
            port_locations.append((x, y))
            port = Port(
                name=port_name,
                center=(x, y),
                width=width,
                orientation=orientation,
                layer=layer,
                port_type=port_type,
            )
            ports.append(port)

    ports = sort_ports_clockwise(ports)

    for port in ports:
        _port_name_or_none = port.name
        if _port_name_or_none is None:
            continue
        if _port_name_or_none in component.ports:
            component_ports = [p.name for p in component.ports]
            raise ValueError(
                f"port {_port_name_or_none!r} already in {component_ports}. "
                "You can pass a port_name_prefix to add it with a different name."
            )

        else:
            component.add_port(name=_port_name_or_none, port=port)
    if auto_rename_ports:
        component.auto_rename_ports()
    return component


add_ports_from_markers_inside = partial(add_ports_from_markers_center, inside=True)


def add_ports_from_labels(
    component: Component,
    port_width: float,
    port_layer: LayerSpec,
    xcenter: float | None = None,
    port_name_prefix: str | None = None,
    port_type: str = "optical",
    get_name_from_label: bool = False,
    layer_label: LayerSpec | None = None,
    fail_on_duplicates: bool = False,
    port_orientation: AngleInDegrees | None = None,
    guess_port_orientation: bool = True,
    port_filter_prefix: str | None = None,
) -> Component:
    """Add ports from labels.

    Assumes that all ports have a label at the port dcenter.
    because labels do not have width, you have to manually specify the ports width

    Args:
        component: to read polygons from and to write ports to.
        port_width: for ports.
        port_layer: for the new created port.
        xcenter: dcenter of the component, for guessing port orientation.
        port_name_prefix: defaults to 'o' for optical and 'e' for electrical.
        port_type: optical, electrical.
        get_name_from_label: uses the label text as port name.
        layer_label: layer for the label.
        fail_on_duplicates: raises ValueError for duplicated port names.
            if False adds incremental suffix (1, 2 ...) to port name.
        port_orientation: None for electrical ports.
        guess_port_orientation: assumes right: 0, left: 180, top: 90, bot: 270.
        port_filter_prefix: prefix for the port name.
    """
    port_name_prefix_default = "o" if port_type == "optical" else "e"
    port_name_prefix = port_name_prefix or port_name_prefix_default
    yc = component.dy

    port_name_to_index: dict[str, int] = {}
    layer_label = layer_label or port_layer

    xc = xcenter or component.dx
    for i, label in enumerate(component.get_labels(layer=layer_label)):
        dx = label.x
        dy = label.y

        if port_filter_prefix and not label.string.startswith(port_filter_prefix):
            continue

        if get_name_from_label:
            port_name = label.string
        else:
            port_name = f"{port_name_prefix}{i + 1}" if port_name_prefix else str(i)

        orientation = port_orientation

        if guess_port_orientation:
            if dx > xc:  # east
                orientation = 0
            elif dx < xc:  # west
                orientation = 180
            elif dy > yc:  # north
                orientation = 90
            elif dy < yc:  # south
                orientation = 270

        if fail_on_duplicates and port_name in component.ports:
            component_ports = [port.name for port in component.ports]
            raise ValueError(
                f"port {port_name!r} already in {component_ports}. "
                "You can pass a port_name_prefix to add it with a different name."
            )
        if get_name_from_label and port_name in component.ports:
            port_name_to_index[port_name] = (
                port_name_to_index[port_name] + 1
                if port_name in port_name_to_index
                else 1
            )
            port_name = f"{port_name}{port_name_to_index[port_name]}"

        component.add_port(
            name=port_name,
            center=(dx, dy),
            width=port_width,
            orientation=orientation,
            port_type=port_type,
            layer=port_layer,
        )
    return component


def add_ports_from_siepic_pins(
    component: Component,
    pin_layer: LayerSpec = "PORT",
    port_layer: LayerSpec | None = None,
    port_type: str = "optical",
) -> Component:
    """Add ports from SiEPIC-type cells, where the pins are defined as paths.

    Looks for label, path pairs.

    Args:
        component: component.
        pin_layer: layer for optical pins.
        port_layer: layer for optical ports.
        port_type: optical, electrical.
    """
    import gdsfactory as gf

    port_layer = port_layer or pin_layer

    pin_layer = gf.get_layer(pin_layer)
    port_layer = gf.get_layer(port_layer)

    c = component
    paths = c.get_paths(pin_layer)
    port_prefix = "o" if port_type == "optical" else "e"

    for i, path in enumerate(paths):
        p1, p2 = list(path.each_point())
        v = p2 - p1
        if v.x < 0:
            orientation = 2
        elif v.x > 0:
            orientation = 0
        elif v.y > 0:
            orientation = 1
        else:
            orientation = 3

        c.create_port(
            name=f"{port_prefix}{i + 1}",
            dwidth=round(path.width / c.kcl.dbu) * c.kcl.dbu,
            dcplx_trans=gf.kdb.DCplxTrans(
                1, orientation, False, path.bbox().center().to_v()
            ),
            layer=port_layer,
            port_type=port_type,
        )

    return c

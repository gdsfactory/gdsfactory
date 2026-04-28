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


def _should_skip_marker(
    dx: float,
    dy: float,
    min_pin_area_um2: float | None,
    max_pin_area_um2: float | None,
    skip_square_ports: bool,
    debug: bool = False,
) -> bool:
    """Return True if a pin marker should be skipped based on area or shape.

    Args:
        dx: marker width.
        dy: marker height.
        min_pin_area_um2: minimum pin area in um^2.
        max_pin_area_um2: maximum pin area in um^2.
        skip_square_ports: whether to skip square markers.
        debug: if True, print skip reasons.
    """
    area = dx * dy
    if min_pin_area_um2 and area < min_pin_area_um2:
        if debug:
            print(f"skipping port at ({dx}, {dy}) with min_pin_area_um2 {area}")
        return True
    if max_pin_area_um2 and area > max_pin_area_um2:
        return True
    if skip_square_ports and snap_to_grid(dx) == snap_to_grid(dy):
        if debug:
            print(f"skipping square port at ({dx}, {dy})")
        return True
    return False


def _infer_port_direction(
    x: float,
    y: float,
    dx: float,
    dy: float,
    pxmin: float,
    pymin: float,
    pxmax: float,
    pymax: float,
    xc: float,
    yc: float,
    dxmin: float,
    dymin: float,
    dxmax: float,
    dymax: float,
    tol: float,
    ports_on_short_side: bool = False,
) -> tuple[float, float, float, float]:
    """Infer port orientation, width, and center position from marker geometry.

    Returns:
        Tuple of (orientation, width, x, y) where orientation is in degrees
        (0=east, 90=north, 180=west, 270=south) and x, y are the center
        coordinates (unchanged from input for non-inside mode).

    Args:
        x: marker center x.
        y: marker center y.
        dx: marker width.
        dy: marker height.
        pxmin: marker bounding box left.
        pymin: marker bounding box bottom.
        pxmax: marker bounding box right.
        pymax: marker bounding box top.
        xc: component center x.
        yc: component center y.
        dxmin: component bounding box left.
        dymin: component bounding box bottom.
        dxmax: component bounding box right.
        dymax: component bounding box top.
        tol: tolerance for boundary detection.
        ports_on_short_side: if True, port is on the short side.
    """
    # Rectangular ports: orientation is easier to detect
    if dy < dx if ports_on_short_side else dx < dy:
        width = dy
        orientation = 0.0 if x > xc else 180.0
    elif dy > dx if ports_on_short_side else dx > dy:
        width = dx
        orientation = 90.0 if y > yc else 270.0
    # Square ports: use boundary proximity
    elif pxmax > dxmax - tol:
        orientation, width = 0.0, dy
    elif pxmin < dxmin + tol:
        orientation, width = 180.0, dy
    elif pymax > dymax - tol:
        orientation, width = 90.0, dx
    elif pymin < dymin + tol:
        orientation, width = 270.0, dx
    # Fallback: use center comparison
    elif pxmax > xc:
        orientation, width = 0.0, dy
    else:
        orientation, width = 180.0, dy

    return orientation, width, x, y


def _apply_inside_position(
    orientation: float,
    x: float,
    y: float,
    pxmin: float,
    pymin: float,
    pxmax: float,
    pymax: float,
    inside: bool,
    use_opposite_side: bool = False,
) -> tuple[float, float]:
    """Adjust port position for inside or opposite-side placement.

    Args:
        orientation: port orientation in degrees.
        x: marker center x.
        y: marker center y.
        pxmin: marker bounding box left.
        pymin: marker bounding box bottom.
        pxmax: marker bounding box right.
        pymax: marker bounding box top.
        inside: whether markers are inside the component.
        use_opposite_side: if True and inside=True, place at opposite edge.
    """
    if not inside:
        return x, y

    if orientation in (0.0, 180.0):
        # East/West: adjust x
        if orientation == 0.0:
            x = pxmin if use_opposite_side else pxmax
        else:
            x = pxmax if use_opposite_side else pxmin
    else:
        # North/South: adjust y
        if orientation == 90.0:
            y = pymin if use_opposite_side else pymax
        else:
            y = pymax if use_opposite_side else pymin

    return x, y


def _snap_port_width(width: float, pin_extra_width: float) -> float:
    """Calculate and snap port width to the nearest 2 nm (0.002 µm).

    Args:
        width: raw port width.
        pin_extra_width: extra width offset to subtract.
    """
    return float(np.round((width - pin_extra_width) / 0.002) * 0.002)


def _auto_detect_port_layer(
    component: Component,
    x: float,
    y: float,
    width: float,
    dbu: float,
    default_layer_idx: int,
    pin_layer: int,
) -> int:
    """Detect the actual port layer from adjacent component geometry.

    When the specified port_layer has no geometry at the port position,
    find the layer with a matching-width edge at the port position.

    Args:
        component: the component to search.
        x: port center x in um.
        y: port center y in um.
        width: port width in um.
        dbu: database unit.
        default_layer_idx: default layer index to return.
        pin_layer: pin layer index to exclude from search.

    Returns:
        The detected layer index, or default_layer_idx if no match found.
    """
    _tol = 0.001
    _marker_box = gf.kdb.DBox(x - _tol, y - _tol, x + _tol, y + _tol)
    _marker_region = gf.kdb.Region(gf.kdb.DPolygon(_marker_box).to_itype(dbu))

    _layer_indexes = list(component.kcl.layer_indexes())

    # Check if default layer already has geometry at this position
    if default_layer_idx in _layer_indexes:
        _region = gf.kdb.Region(component.begin_shapes_rec(default_layer_idx))
        if not _region.edges().interacting(_marker_region).is_empty():
            return default_layer_idx

    # Search other layers for a matching-width edge
    _best_idx = None
    _best_area = float("inf")
    for _li in _layer_indexes:
        if _li in (default_layer_idx, pin_layer):
            continue
        _region = gf.kdb.Region(component.begin_shapes_rec(_li))
        _edges = _region.edges().interacting(_marker_region)
        if _edges.is_empty():
            continue
        _width_match = any(
            np.isclose(e.length() * dbu, width, atol=0.1) for e in _edges.each()
        )
        if not _width_match:
            continue
        # Prefer the layer with the smallest polygon at port position
        _polys = _region.interacting(_marker_region)
        for _p in _polys.each():
            _area = abs(_p.area()) * dbu * dbu
            if _area < _best_area:
                _best_area = _area
                _best_idx = _li
            break

    return _best_idx if _best_idx is not None else default_layer_idx


def _register_ports(
    component: Component,
    ports: list[Port],
    auto_rename_ports: bool,
    allow_none_names: bool = False,
) -> Component:
    """Sort, deduplicate, and add ports to a component.

    Args:
        component: target component.
        ports: list of ports to add.
        auto_rename_ports: if True, auto-rename ports after adding.
        allow_none_names: if True, skip ports with None names instead of raising.
    """
    ports = sort_ports_clockwise(ports)

    for port in ports:
        _port_name = port.name
        if allow_none_names and _port_name is None:
            continue
        if _port_name in component.ports or port in component.ports:
            component_ports = [p.name for p in component.ports]
            raise ValueError(
                f"port {_port_name!r} already in {component_ports}. "
                "You can pass a port_name_prefix to add it with a different name."
            )
        component.add_port(name=_port_name, port=port)

    if auto_rename_ports:
        component.auto_rename_ports()
    return component


def add_ports_from_markers_square(
    component: Component,
    pin_layer: LayerSpec = "DEVREC",
    port_layer: LayerSpec | None = None,
    orientation: AngleInDegrees = 90,
    min_pin_area_um2: float = 0,
    max_pin_area_um2: float | None = 150 * 150,
    pin_extra_width: float = 0.0,
    port_names: Sequence[str] | None = None,
    port_name_prefix: str | None = None,
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
    port_names = list(
        port_names
        or [f"{port_name_prefix}{i + 1}" for i in range(len(port_markers.polygons))]
    )
    layer = port_layer or pin_layer

    for port_name, p in zip(port_names, port_markers.polygons, strict=False):
        (xmin, ymin), (xmax, ymax) = p.bounding_box()
        x, y = np.sum(p.bounding_box(), 0) / 2

        dy = snap_to_grid(ymax - ymin)
        dx = snap_to_grid(xmax - xmin)
        width = dx - pin_extra_width

        # Snap to the nearest 2 nm (0.002 µm)
        width = np.round((width - pin_extra_width) / 0.002) * 0.002

        if dx == dy and max_pin_area_um2 > dx * dy > min_pin_area_um2:
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
    auto_detect_port_layer: bool = False,
    debug: bool = False,
) -> Component:
    """Add ports from pins guessing port orientation from component boundary.

    Args:
        component: to read polygons from and to write ports to.
        pin_layer: layer for pin maker.
        port_layer: for the new created port. Defaults to pin_layer.
        inside: True-> markers  inside. False-> markers at center.
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
        auto_detect_port_layer: if True, detect the actual port layer from adjacent
            component geometry when port_layer has no geometry at the port position.
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
    from gdsfactory.pdk import get_layer

    xc = xcenter or component.x
    yc = ycenter or component.y
    dxmax = component.xmax
    dxmin = component.xmin
    dymax = component.ymax
    dymin = component.ymin
    dbu = float(component.kcl.dbu)

    layer = port_layer or pin_layer
    port_locations: list[tuple[float, float]] = []

    port_name_prefix_default = "o" if port_type == "optical" else "e"
    port_name_prefix = port_name_prefix or port_name_prefix_default

    pin_layer = gf.get_layer(pin_layer)

    polygons = component.get_polygons(by="index")
    if pin_layer not in polygons:
        warnings.warn(
            f"no pin layer {pin_layer} found in {component.layers}", stacklevel=3
        )
        return component

    port_markers = polygons[pin_layer]
    ports: list[Port] = []

    for i, p in enumerate(port_markers):
        port_name = f"{port_name_prefix}{i + 1}" if port_name_prefix else str(i)
        bbox = p.bbox()
        pxmin, pymin, pxmax, pymax = map(
            float, (bbox.left, bbox.bottom, bbox.right, bbox.top)
        )

        x = (pxmax + pxmin) / 2
        y = (pymin + pymax) / 2
        dy = abs(pymax - pymin)
        dx = abs(pxmax - pxmin)
        dx *= dbu
        dy *= dbu
        x *= dbu
        y *= dbu
        pxmax *= dbu
        pymax *= dbu
        pxmin *= dbu
        pymin *= dbu

        if _should_skip_marker(dx, dy, min_pin_area_um2, max_pin_area_um2, skip_square_ports, debug):
            continue

        orientation, width, x, y = _infer_port_direction(
            x, y, dx, dy, pxmin, pymin, pxmax, pymax,
            xc, yc, dxmin, dymin, dxmax, dymax, tol, ports_on_short_side,
        )
        x, y = _apply_inside_position(
            orientation, x, y, pxmin, pymin, pxmax, pymax, inside,
        )
        width = _snap_port_width(width - pin_extra_width, pin_extra_width)

        _port_layer_idx = get_layer(layer)
        if auto_detect_port_layer:
            _port_layer_idx = _auto_detect_port_layer(
                component, x, y, width, dbu, _port_layer_idx, pin_layer,
            )

        if (x, y) not in port_locations:
            port_locations.append((x, y))
            ports.append(Port(
                name=port_name,
                center=(x, y),
                width=width,
                orientation=orientation,
                layer=_port_layer_idx,
                port_type=port_type,
            ))

    return _register_ports(component, ports, auto_rename_ports)


def add_ports_from_boxes(
    component: Component,
    pin_layer: LayerSpec,
    port_layer: LayerSpec | None = None,
    inside: bool = False,
    use_opposite_side: bool = False,
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
        inside: True-> markers  inside. False-> markers at center.
        use_opposite_side: if True and inside=True, place port at opposite edge of pin.
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
    xc = xcenter or component.x
    yc = ycenter or component.y
    dxmax = component.xmax
    dxmin = component.xmin
    dymax = component.ymax
    dymin = component.ymin

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

        if _should_skip_marker(dx, dy, min_pin_area_um2, max_pin_area_um2, skip_square_ports, debug):
            continue

        orientation, width, x, y = _infer_port_direction(
            x, y, dx, dy, pxmin, pymin, pxmax, pymax,
            xc, yc, dxmin, dymin, dxmax, dymax, tol, ports_on_short_side,
        )
        x, y = _apply_inside_position(
            orientation, x, y, pxmin, pymin, pxmax, pymax, inside, use_opposite_side,
        )
        width = _snap_port_width(width, pin_extra_width)

        if (x, y) not in port_locations:
            port_locations.append((x, y))
            ports.append(Port(
                name=port_name,
                center=(x, y),
                width=width,
                orientation=orientation,
                layer=layer,
                port_type=port_type,
            ))

    return _register_ports(component, ports, auto_rename_ports, allow_none_names=True)


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
    port_orientation: AngleInDegrees = 0,
    guess_port_orientation: bool = True,
    port_filter_prefix: str | None = None,
    skip_duplicates: bool = False,
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
        get_name_from_label: uses the label text as port name.
        layer_label: layer for the label.
        fail_on_duplicates: raises ValueError for duplicated port names.
            if False adds incremental suffix (1, 2 ...) to port name.
        port_orientation: None for electrical ports.
        guess_port_orientation: assumes right: 0, left: 180, top: 90, bot: 270.
        port_filter_prefix: prefix for the port name.
        skip_duplicates: if True skips ports with the same name.
    """
    port_name_prefix_default = "o" if port_type == "optical" else "e"
    port_name_prefix = port_name_prefix or port_name_prefix_default
    yc = component.y

    port_name_to_index: dict[str, int] = {}
    layer_label = layer_label or port_layer

    label_names = set()

    xc = xcenter or component.x
    for i, label in enumerate(component.get_labels(layer=layer_label)):
        dx = label.x
        dy = label.y

        if skip_duplicates and label.string in label_names:
            print("Skipping duplicate label:", label.string)
            continue

        label_names.add(label.string)

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
            width=round(path.width / c.kcl.dbu) * c.kcl.dbu,
            dcplx_trans=gf.kdb.DCplxTrans(
                1, orientation, False, path.bbox().center().to_v()
            ),
            layer=port_layer,
            port_type=port_type,
        )

    return c

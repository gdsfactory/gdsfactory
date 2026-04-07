from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.typings import CrossSectionSpec, LayerSpec
from gdsfactory.utils import spline_points


@gf.cell_with_module_name
def taper_spline(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    widths: tuple[float, ...] | None = None,
    positions: tuple[float, ...] | None = None,
    npoints: int = 100,
    degree: int = 3,
    bc_type: str | None = None,
    monotonic: bool = True,
    layer: LayerSpec | None = None,
    port: Port | None = None,
    with_two_ports: bool = True,
    cross_section: CrossSectionSpec = "strip",
    port_names: tuple[str, str] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
    with_bbox: bool = True,
) -> Component:
    """Returns a taper where the width profile is a spline or PCHIP.

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
        widths: optional intermediate width values. If provided, width1 and width2 are used as start/end.
        positions: relative positions (0 to 1) for the widths. Defaults to uniform.
        npoints: number of points to generate for the polygon.
        degree: spline degree (only if monotonic=False).
        bc_type: boundary conditions for the spline (only if monotonic=False).
        monotonic: if True, uses PCHIP interpolation to ensure monotonic width changes.
        layer: layer spec.
        port: can taper from a port instead of defining width1.
        with_two_ports: includes a second port.
        cross_section: cross_section specification.
        port_names: input and output port names.
        port_types: input and output port types.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.

    .. code::

        import gdsfactory as gf
        c = gf.components.taper_spline(length=20, width1=0.5, width2=4.0, widths=(2.0, 1.0))
        c.show()
    """
    if len(port_types) != 2:
        raise ValueError("port_types should have two elements")

    x1 = gf.get_cross_section(cross_section, width=width1)
    if width2:
        width2 = gf.snap.snap_to_grid2x(width2)
        x2 = gf.get_cross_section(cross_section, width=width2)
    else:
        x2 = x1

    width1 = x1.width
    width2 = x2.width

    if isinstance(port, gf.Port):
        width1 = port.width

    width2 = width2 or width1

    # Combine widths: width1, widths..., width2
    all_widths = [width1]
    if widths:
        all_widths.extend(widths)
    all_widths.append(width2)

    all_positions = positions
    if all_positions is None:
        all_positions = np.linspace(0, 1, len(all_widths))

    if len(all_positions) != len(all_widths):
        raise ValueError("len(positions) must equal len(widths) + 2")

    # Create points (positions * length, width/2) for the upper boundary
    control_points = np.stack(
        [np.asarray(all_positions) * length, np.asarray(all_widths) / 2.0], axis=1
    )

    # Interpolate
    method = "pchip" if monotonic else "bspline"
    smooth_points = spline_points(
        control_points,
        degree=degree,
        npoints=npoints,
        bc_type=bc_type,
        method=method,
    )

    x = smooth_points[:, 0]
    y = smooth_points[:, 1]

    # Create the full polygon (top curve and then bottom curve flipped)
    points = np.concatenate(
        [smooth_points, np.stack([np.flip(x), np.flip(-y)], axis=1)]
    )

    c = gf.Component()
    width_max = max(all_widths)
    if layer:
        xs = gf.get_cross_section(cross_section, width=width_max, layer=layer)
    else:
        xs = gf.get_cross_section(cross_section, width=width_max)
    layer = layer or xs.layer
    assert layer is not None

    c.add_polygon(points, layer=layer)

    c.add_port(
        name=port_names[0],
        center=(0, 0),
        width=float(width1),
        orientation=180,
        layer=layer,
        cross_section=x1,
        port_type=port_types[0],
    )
    if with_two_ports:
        c.add_port(
            name=port_names[1],
            center=(length, 0),
            width=float(width2),
            orientation=0,
            layer=layer,
            cross_section=x2,
            port_type=port_types[1],
        )

    if with_bbox:
        xs.add_bbox(c)

    c.info["length"] = length
    c.info["width1"] = float(width1)
    c.info["width2"] = float(width2)
    c.info["widths"] = [float(w) for w in all_widths]
    c.info["positions"] = [float(p) for p in all_positions]
    return c

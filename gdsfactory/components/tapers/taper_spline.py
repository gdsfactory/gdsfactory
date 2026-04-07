from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec
from gdsfactory.utils import spline_points


@gf.cell_with_module_name
def taper_spline(
    length: float = 10.0,
    widths: tuple[float, ...] = (0.5, 2.0, 1.0, 4.0),
    positions: tuple[float, ...] | None = None,
    npoints: int = 100,
    degree: int = 3,
    bc_type: str | None = None,
    layer: LayerSpec | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns a taper where the width profile is a spline.

    Args:
        length: taper length.
        widths: width values at specified positions.
        positions: relative positions (0 to 1) for the widths. Defaults to uniform.
        npoints: number of points to generate for the polygon.
        degree: spline degree.
        bc_type: boundary conditions for the spline (e.g., 'clamped', 'natural').
        layer: layer spec.
        cross_section: cross_section specification.

    .. code::

        import gdsfactory as gf
        c = gf.components.taper_spline(length=20, widths=(0.5, 2.0, 1.0, 4.0))
        c.show()
    """
    if positions is None:
        positions = np.linspace(0, 1, len(widths))

    if len(positions) != len(widths):
        raise ValueError("len(positions) must equal len(widths)")

    # Create points (positions * length, width/2) for the upper boundary
    control_points = np.stack(
        [np.asarray(positions) * length, np.asarray(widths) / 2.0], axis=1
    )

    # Interpolate using spline
    smooth_points = spline_points(
        control_points, degree=degree, npoints=npoints, bc_type=bc_type
    )

    x = smooth_points[:, 0]
    y = smooth_points[:, 1]

    # Create the full polygon (top curve and then bottom curve flipped)
    points = np.concatenate(
        [smooth_points, np.stack([np.flip(x), np.flip(-y)], axis=1)]
    )

    c = gf.Component()
    # Handle layer from cross-section if not specified
    xs = gf.get_cross_section(cross_section)
    layer = layer or xs.layer

    c.add_polygon(points, layer=layer)

    c.add_port(
        name="o1",
        center=(0, 0),
        width=float(widths[0]),
        orientation=180,
        layer=layer,
        cross_section=xs,
    )
    c.add_port(
        name="o2",
        center=(length, 0),
        width=float(widths[-1]),
        orientation=0,
        layer=layer,
        cross_section=xs,
    )

    c.info["length"] = length
    c.info["widths"] = list(widths)
    c.info["positions"] = list(positions)
    return c

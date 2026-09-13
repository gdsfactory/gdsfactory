"""Inductor components PDK."""

import math

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs

from .._schematic import inductor_schematic
from ._geometry import (
    Poly,
    _add_via_array,
    _make_aspect_shift_y,
    _map_y,
    _mirror_x,
    _pgs,
    _routing_geometric_45,
    _via_component_info,
    _zip,
)

__all__ = ["inductor", "spiral_inductor", "symmetric_inductor"]


def inductor_min_diameter(width: float, space: float, turns: int, grid: float) -> float:
    """Calculate minimum diameter for inductor.

    Args:
        width: Width of the inductor trace in micrometers.
        space: Space between turns in micrometers.
        turns: Number of turns.
        grid: Grid resolution.

    Returns:
        Minimum diameter in micrometers.
    """
    min_d = 2 * turns * (width + space) + 4 * width
    return round(min_d / grid) * grid


@gf.cell_with_module_name(schematic_function=inductor_schematic, tags=["analog"])
def inductor(
    width: float = 2.0,
    space: float = 2.1,
    diameter: float = 25.35,
    resistance: float = 0.5777,
    inductance: float = 33.303e-12,
    turns: int = 1,
    layer_metal: LayerSpec = "M3",
    layer_inductor: LayerSpec = "M1",
    layer_metal_pin: LayerSpec = "WG_PIN",
    layers_no_fill: LayerSpecs = ("DEVREC", "NO_TILE_SI"),
) -> Component:
    """Create a 2-turn inductor.

    Args:
        width: Width of the inductor trace in micrometers.
        space: Space between turns in micrometers.
        diameter: Inner diameter in micrometers.
        resistance: Resistance in ohms.
        inductance: Inductance in henries.
        turns: Number of turns (default 1 for inductor2).
        layer_metal: Layer for the metal trace.
        layer_inductor: Layer for the inductor region.
        layer_metal_pin: Layer for the metal pins.
        layers_no_fill: Layers to exclude from fill.

    Returns:
        Component with inductor layout.
    """
    c = Component()

    # Grid fixing for manufacturing constraints
    grid = 0.01
    w = round(width / (2 * grid)) * 2 * grid
    s = round(space / grid) * grid
    d = round(diameter / (2 * grid)) * 2 * grid

    # Calculate geometry parameters
    r = d / 2 + s
    octagon_center_y = 3 * r
    pi_over_4 = math.radians(45)

    path_points = []
    path_points.append((+space / 2, octagon_center_y - r * math.cos(pi_over_4 / 2)))

    for i in range(-2, 6):
        angle = i * pi_over_4 + pi_over_4 / 2
        r = d / 2 + s
        x = r * math.cos(angle)
        y = r * math.sin(angle) + octagon_center_y

        if -2 <= i < 2:
            path_points.append((x, y))
        else:
            path_points.append((x, y))

    path_points.append((-space / 2, octagon_center_y - r * math.cos(pi_over_4 / 2)))

    # Create the path
    path = gf.Path(path_points)
    c = gf.path.extrude(path, layer=layer_metal, width=w)

    # Adding ports
    length = 2 * r + s

    port1_trace = c << gf.components.rectangle(size=(s, length), layer=layer_metal)
    port1_trace.move((-s - s / 2, 0))
    c.add_port(name="P1", center=(-s, s), width=s, orientation=270, layer=layer_metal)

    port2_trace = c << gf.components.rectangle(size=(s, length), layer=layer_metal)
    port2_trace.move((s - s / 2, 0))
    c.add_port(name="P2", center=(+s, s), width=s, orientation=270, layer=layer_metal)

    # Add IND layer
    outer_polygon_pts = []
    for i in range(8):
        r_outer = (d / 2 + length) / (math.cos(pi_over_4 / 2))
        angle = i * pi_over_4 + pi_over_4 / 2
        x = r_outer * math.cos(angle)
        y = r_outer * math.sin(angle) + octagon_center_y
        outer_polygon_pts.append((x, y))

    c.add_polygon(points=outer_polygon_pts, layer=layer_inductor)

    # Add No fill layers
    for layer in layers_no_fill:
        c.add_polygon(points=outer_polygon_pts, layer=layer)

    # Adding pins
    pin_1_trace = c << gf.components.rectangle(size=(s, s), layer=layer_metal_pin)
    pin_1_trace.move((s / 2, 0))

    pin_2_trace = c << gf.components.rectangle(size=(s, s), layer=layer_metal_pin)
    pin_2_trace.move((-s - s / 2, 0))

    # Add metadata
    c.info["resistance"] = resistance
    c.info["inductance"] = inductance
    c.info["model"] = "inductor2"
    c.info["turns"] = turns
    c.info["width"] = width
    c.info["space"] = space
    c.info["diameter"] = diameter
    return c


@gf.cell_with_module_name(tags=["analog"])
def spiral_inductor(
    d_out: float = 130.0,
    N: int = 3,
    sides: int = 8,
    width: float = 10.0,
    spacing: float = 4.0,
    aspect_ratio: float = 1.0,
    port_side: str = "same",
    add_pgs: bool = False,
    pgs_diameter: float = 150.0,
    pgs_width: float = 2.0,
    pgs_spacing: float = 1.0,
    via: ComponentSpec = "via2",
    resistance: float = 0.5777,
    inductance: float = 33.303e-12,
    layer_winding: LayerSpec = "M3",
    layer_underpass: LayerSpec = "M2",
    layers_pgs: LayerSpecs = ("M1",),
) -> Component:
    """Polygonal spiral inductor.

    Args:
        d_out: Outer diameter of the spiral in micrometers.
        N: Number of complete turns.
        sides: Number of polygon sides per full turn (8 = octagonal).
        width: Metal trace width in micrometers.
        spacing: Gap between adjacent turns in micrometers.
        aspect_ratio: Y-axis scale factor for non-square spirals (1.0 = symmetric).
        port_side: ``"same"`` keeps both ports on the same side;
            ``"opposite"`` places them on opposite sides.
        add_pgs: When True, add a patterned ground shield on layers_pgs.
        pgs_diameter: Bounding size D of the ground shield square, in micrometers.
        pgs_width: Strip width w of each ground shield finger, in micrometers.
        pgs_spacing: Gap s between adjacent ground shield fingers, in micrometers.
        via: via ComponentSpec connecting winding <-> underpass.
        resistance: Series resistance in ohms, stored as metadata only.
        inductance: Inductance in henries, stored as metadata only.
        layer_winding: Metal layer for the main spiral winding.
        layer_underpass: Metal layer for the inner-terminal underpass bridge (one layer below layer_winding).
        layers_pgs: Layers on which the patterned ground shield is drawn,
            kept separate from both layer_winding and layer_underpass
            since the underpass carries a live signal and shouldn't s
            hare a layer with a grounded shield.

    Returns:
        Component with 2 RF ports:
          P1  ->  entry terminal  (layer_winding)
          P2  ->  exit terminal   (layer_underpass)
    """
    c = Component()
    PI = math.pi
    opposite = port_side == "opposite"

    # Derived parameters (identical to build_spiral_inductor)
    s = (spacing + width) / math.cos(PI / sides)
    v = width / math.cos(PI / sides)
    R1 = d_out / 2 / math.cos(PI / sides)
    R2 = R1 - v

    n_pts = sides // 2
    angles = [
        PI * (1 / (2 * n_pts) + i * (1 - 1 / n_pts) / (n_pts - 1)) for i in range(n_pts)
    ]

    x_shift = -s / 2 * math.cos(PI / sides)
    y_shift = -s / 2 * math.sin(PI / sides)

    n_sections = 2 * N - 1 if opposite else 2 * N

    x_out: list[float] = []
    y_out: list[float] = []
    x_in: list[float] = []
    y_in: list[float] = []
    r1, r2 = R1, R2

    for section in range(n_sections):
        if section % 2 == 0:
            for phi in angles:
                x_out.append(r1 * math.cos(phi))
                x_in.append(r2 * math.cos(phi))
                y_out.append(r1 * math.sin(phi))
                y_in.append(r2 * math.sin(phi))
        else:
            for phi in angles:
                x_out.append(-r1 * math.cos(phi) + x_shift)
                x_in.append(-r2 * math.cos(phi) + x_shift)
                y_out.append(-r1 * math.sin(phi) + y_shift)
                y_in.append(-r2 * math.sin(phi) + y_shift)
        r1 -= s / 2
        r2 -= s / 2

    entry_yc = 0.0 if opposite else (width + spacing) / 2
    exit_yc = 0.0 if opposite else -(width + spacing) / 2

    x_out_start = [d_out / 2 + width, x_out[0]]
    x_in_start = [d_out / 2 + width, x_in[0]]
    y_out_start = [entry_yc + width / 2, entry_yc + width / 2]
    y_in_start = [entry_yc - width / 2, entry_yc - width / 2]

    x_out_end = [x_out[-1]]
    x_in_end = [x_in[-1]]
    y_end = [-width / 2 if opposite else -spacing / 2]

    x_poly = (
        x_out_start
        + x_out
        + x_out_end
        + list(reversed(x_in_end))
        + list(reversed(x_in))
        + list(reversed(x_in_start))
    )
    y_poly = (
        y_out_start
        + y_out
        + y_end
        + list(reversed(y_end))
        + list(reversed(y_in))
        + list(reversed(y_in_start))
    )
    winding_polygon: Poly = list(zip(x_poly, y_poly, strict=True))

    last_x_in = x_in[-1]
    last_x_out = x_out[-1]
    underpass_end_x = -(d_out / 2 + width) if opposite else d_out / 2 + width

    underpass_polygon: Poly = [
        (last_x_in, exit_yc - width / 2),
        (underpass_end_x, exit_yc - width / 2),
        (underpass_end_x, exit_yc + width / 2),
        (last_x_in, exit_yc + width / 2),
    ]

    shift_y = _make_aspect_shift_y(d_out, aspect_ratio)

    # Winding + underpass polygons
    c.add_polygon(_map_y(winding_polygon, shift_y), layer=layer_winding)
    c.add_polygon(_map_y(underpass_polygon, shift_y), layer=layer_underpass)

    # Via array: fill the overlap region between winding and underpass
    via_cx = last_x_out + (last_x_in - last_x_out) / 2
    via_cy = exit_yc

    via_component = gf.get_component(via)
    _w, h, enclosure, _pitch_x, pitch_y = _via_component_info(via_component)

    via_spacing_y = pitch_y - h
    extend = 2 * (h + enclosure) + via_spacing_y

    avail_x = width - 2 * enclosure
    if extend > width:
        avail_y = extend - 2 * enclosure
        via_center_y = via_cy + (extend - width) / 2
    else:
        avail_y = width - 2 * enclosure
        via_center_y = via_cy

    _add_via_array(c, via_component, via_cx, shift_y(via_center_y), avail_x, avail_y)

    if add_pgs:
        for layer in layers_pgs:
            for strip in _pgs(pgs_diameter, pgs_width, pgs_spacing):
                c.add_polygon(_map_y(strip, shift_y), layer=layer)

    # Ports
    c.add_port(
        "P1",
        center=(d_out / 2 + width, shift_y(entry_yc)),
        width=width,
        orientation=0.0,
        layer=layer_winding,
    )
    c.add_port(
        "P2",
        center=(underpass_end_x, shift_y(exit_yc)),
        width=width,
        orientation=180.0,
        layer=layer_underpass,
    )

    # Metadata
    c.info["resistance"] = resistance
    c.info["inductance"] = inductance
    c.info["model"] = "spiral_inductor"
    c.info["turns"] = N
    c.info["width"] = width
    c.info["spacing"] = spacing
    c.info["diameter"] = d_out

    return c


# Symmetric (differential) inductor


@gf.cell_with_module_name(tags=["analog"])
def symmetric_inductor(
    d_out: float = 150.0,
    N: int = 3,
    sides: int = 8,
    width: float = 10.0,
    spacing: float = 2.0,
    center_tap: bool = False,
    via_extent: float | None = None,
    port_spacing: float | None = None,
    aspect_ratio: float = 1.0,
    via: ComponentSpec = "via2",
    resistance: float = 0.5777,
    inductance: float = 33.303e-12,
    add_pgs: bool = False,
    pgs_diameter: float = 180.0,
    pgs_width: float = 4.0,
    pgs_spacing: float = 2.0,
    layer_winding: LayerSpec = "M3",
    layer_underpass: LayerSpec = "M2",
    layers_pgs: LayerSpecs = ("M1",),
) -> Component:
    """Symmetric (differential) spiral inductor.

    Args:
        d_out: Outer diameter of the two-lobe structure, in micrometers.
        N: Number of complete windings per side.
        sides: Number of polygon sides per full turn (8 = octagonal).
        width: Metal trace width in micrometers.
        spacing: Gap between adjacent turns in micrometers.
        center_tap: When True, add a center-tap bridge (and CT port)
            routed through layer_underpass with its own via connection
            to the winding.
        via_extent: Length the crossing route extends past the crossing
            box on layer_underpass, and the box size used to size the
            crossing/centertap via arrays. If None, it's derived from the
            chosen via's own geometry.
        port_spacing: Horizontal spacing of the differential ports P1/P2.
        aspect_ratio: Y-axis scale factor for non-square windings
        via: via ComponentSpec connecting winding <-> crossing/centertap.
        resistance: Series resistance in ohms, stored as metadata only
        inductance: Inductance in henries, stored as metadata only
        add_pgs: When True, add a patterned ground shield on layers_pgs.
        pgs_diameter: Bounding size D of the ground shield square, in micrometers.
        pgs_width: Strip width w of each ground shield finger, in micrometers.
        pgs_spacing: Gap s between adjacent ground shield fingers, in micrometers.
        layer_winding: Metal layer for the main winding (top metal).
        layer_underpass: Metal layer for crossings and the center-tap bridge (one layer below layer_winding).
        layers_pgs: Layers on which the patterned ground shield is drawn.

    Returns:
        Component with 2 or 3 ports:
          P1  ->  left differential terminal   (layer_winding)
          P2  ->  right differential terminal  (layer_winding)
          CT  ->  center tap (only if center_tap=True), on layer_winding
                  if N <= 2, else on layer_underpass.
    """
    c = Component()
    PI = math.pi
    SQRT2 = math.sqrt(2)

    via_component = gf.get_component(via)
    _via_w, via_h, via_enclosure, _via_pitch_x, via_pitch_y = _via_component_info(
        via_component
    )

    if via_extent is None:
        via_spacing_y = via_pitch_y - via_h
        extend = 2 * (via_h + via_enclosure) + via_spacing_y
    else:
        extend = via_extent

    ps = spacing if port_spacing is None else port_spacing

    v = width / math.cos(PI / sides)
    s = (spacing + width) / math.cos(PI / sides)
    R1 = d_out / 2 / math.cos(PI / sides)
    R2 = R1 - v

    n_half = sides // 2
    left_angles = [PI * (0.5 + (i + 0.5) * 2 / sides) for i in range(n_half)]
    right_angles = [PI * (-0.5 + (i + 0.5) * 2 / sides) for i in range(n_half)]
    sep_total = width + spacing + (SQRT2 - 1) * (2 * spacing + width)

    shift_y = _make_aspect_shift_y(d_out, aspect_ratio)

    def add_winding(poly: Poly) -> None:
        c.add_polygon(_map_y(poly, shift_y), layer=layer_winding)

    def add_crossing(poly: Poly) -> None:
        c.add_polygon(_map_y(poly, shift_y), layer=layer_underpass)

    for winding in range(N):
        # Left section
        x_out = [R1 * math.cos(p) for p in left_angles]
        y_out = [R1 * math.sin(p) for p in left_angles]
        x_in = [R2 * math.cos(p) for p in left_angles]
        y_in = [R2 * math.sin(p) for p in left_angles]
        if winding == N - 1:
            if N % 2 == 0:
                x_out = [-sep_total / 2, *x_out, 0]
                x_in = [-sep_total / 2, *x_in, 0]
            else:
                x_out = [0, *x_out, -sep_total / 2]
                x_in = [0, *x_in, -sep_total / 2]
        else:
            x_out = [-sep_total / 2, *x_out, -sep_total / 2]
            x_in = [-sep_total / 2, *x_in, -sep_total / 2]
        y_out = [y_out[0], *y_out, y_out[-1]]
        y_in = [y_in[0], *y_in, y_in[-1]]
        add_winding(_zip([*x_out, *reversed(x_in)], [*y_out, *reversed(y_in)]))

        # Right section
        x_out = [R1 * math.cos(p) for p in right_angles]
        y_out = [R1 * math.sin(p) for p in right_angles]
        x_in = [R2 * math.cos(p) for p in right_angles]
        y_in = [R2 * math.sin(p) for p in right_angles]
        if winding == N - 1:
            if N % 2 == 0:
                x_out = [0, *x_out, sep_total / 2]
                x_in = [0, *x_in, sep_total / 2]
            else:
                x_out = [sep_total / 2, *x_out, 0]
                x_in = [sep_total / 2, *x_in, 0]
        else:
            x_out = [sep_total / 2, *x_out, sep_total / 2]
            x_in = [sep_total / 2, *x_in, sep_total / 2]
        y_out = [y_out[0], *y_out, y_out[-1]]
        y_in = [y_in[0], *y_in, y_in[-1]]
        add_winding(_zip([*x_out, *reversed(x_in)], [*y_out, *reversed(y_in)]))

        # Crossings (skip on the innermost winding — nothing left to cross)
        if winding != N - 1:
            if winding % 2 == 0:
                h = R1 * math.sin(PI * (0.5 - 1 / sides))
            else:
                h = (-R2 + s) * math.sin(PI * (0.5 - 1 / sides))

            add_crossing(
                _routing_geometric_45(
                    width, spacing, 0, h - width - spacing / 2, extend
                )
            )
            cross_top = _routing_geometric_45(
                width, spacing, 0, h - width - spacing / 2, 0
            )
            add_winding(_mirror_x(cross_top))

            for cx, cy in [
                (-sep_total / 2 - width / 2, h - 3 * width / 2 - spacing),
                (sep_total / 2 + width / 2, h - width / 2),
            ]:
                dx = math.copysign(1, cx) * (extend - width) / 2
                _add_via_array(
                    c,
                    via_component,
                    cx + dx,
                    shift_y(cy),
                    extend - 2 * via_enclosure,
                    width - 2 * via_enclosure,
                )

        R1 -= s
        R2 -= s

    # Center tap
    ct_port_layer = layer_winding
    if center_tap:
        x_ct = [-width / 2, -width / 2, width / 2, width / 2]
        if N % 2 != 0:
            if N <= 2:
                y_ct = [
                    -d_out / 2,
                    d_out / 2 - spacing * (N - 1) - width * (N - 1),
                    d_out / 2 - spacing * (N - 1) - width * (N - 1),
                    -d_out / 2,
                ]
            else:
                y_ct = [
                    -d_out / 2 + width - extend,
                    d_out / 2 - spacing * (N - 1) - width * (N - 1) - extend,
                    d_out / 2 - spacing * (N - 1) - width * (N - 1) - extend,
                    -d_out / 2 + width - extend,
                ]
        else:
            if N <= 2:
                y_ct = [
                    -d_out / 2,
                    -d_out / 2 + spacing * (N - 1) + width * (N - 1),
                    -d_out / 2 + spacing * (N - 1) + width * (N - 1),
                    -d_out / 2,
                ]
            else:
                y_ct = [
                    -d_out / 2 + width - extend,
                    -d_out / 2 + spacing * (N - 1) + width * (N - 1),
                    -d_out / 2 + spacing * (N - 1) + width * (N - 1),
                    -d_out / 2 + width - extend,
                ]

        if N <= 2:
            add_winding(_zip(x_ct, y_ct))
        else:
            ct_port_layer = layer_underpass
            add_crossing(_zip(x_ct, y_ct))

            if N % 2 != 0:
                x_ct1, y_ct1 = (
                    0,
                    d_out / 2 - spacing * (N - 1) - width * (N - 1) - extend / 2,
                )
                x_ct2, y_ct2 = 0, -d_out / 2 + width / 2 + (width - extend) / 2
            else:
                x_ct1, y_ct1 = (
                    0,
                    -d_out / 2 + spacing * (N - 1) + width * N - width + extend / 2,
                )
                x_ct2, y_ct2 = 0, -d_out / 2 + width - extend / 2

            xvp1 = [
                x_ct1 - width / 2,
                x_ct1 - width / 2,
                x_ct1 + width / 2,
                x_ct1 + width / 2,
            ]
            yvp1 = [
                y_ct1 - extend / 2,
                y_ct1 + extend / 2,
                y_ct1 + extend / 2,
                y_ct1 - extend / 2,
            ]
            xvp2 = [
                x_ct2 - width / 2,
                x_ct2 - width / 2,
                x_ct2 + width / 2,
                x_ct2 + width / 2,
            ]
            yvp2 = [
                y_ct2 - extend / 2,
                y_ct2 + extend / 2,
                y_ct2 + extend / 2,
                y_ct2 - extend / 2,
            ]

            add_winding(_zip(xvp1, yvp1))
            add_crossing(_zip(xvp1, yvp1))
            add_crossing(_zip(xvp2, yvp2))

            for cx, cy in [(x_ct1, y_ct1), (x_ct2, y_ct2)]:
                _add_via_array(
                    c,
                    via_component,
                    cx,
                    shift_y(cy),
                    width - 2 * via_enclosure,
                    extend - 2 * via_enclosure,
                )

    # Ports
    pxo = ps + width if center_tap else (ps + width) / 2
    x_port = [
        -sep_total / 2,
        -pxo + width / 2,
        -pxo + width / 2,
        -pxo - width / 2,
        -pxo - width / 2,
        -sep_total / 2,
    ]
    y_port = [
        -d_out / 2 + width,
        -d_out / 2 + width,
        -d_out / 2 - width,
        -d_out / 2 - width,
        -d_out / 2,
        -d_out / 2,
    ]
    if center_tap:
        add_winding(
            _zip(
                [-width / 2, -width / 2, width / 2, width / 2],
                [
                    -d_out / 2 - width,
                    -d_out / 2 + width,
                    -d_out / 2 + width,
                    -d_out / 2 - width,
                ],
            )
        )
    add_winding(_zip(x_port, y_port))
    add_winding(_zip([-x for x in x_port], y_port))

    port_y = -d_out / 2 - width
    port_x = ps + width if center_tap else (ps + width) / 2

    c.add_port(
        "P1",
        center=(-port_x, shift_y(port_y)),
        width=width,
        orientation=270.0,
        layer=layer_winding,
    )
    c.add_port(
        "P2",
        center=(port_x, shift_y(port_y)),
        width=width,
        orientation=270.0,
        layer=layer_winding,
    )
    if center_tap:
        c.add_port(
            "CT",
            center=(0, shift_y(port_y)),
            width=width,
            orientation=270.0,
            layer=ct_port_layer,
        )

    if add_pgs:
        for layer in layers_pgs:
            for strip in _pgs(pgs_diameter, pgs_width, pgs_spacing):
                c.add_polygon(_map_y(strip, shift_y), layer=layer)

    # Metadata
    c.info["resistance"] = resistance
    c.info["inductance"] = inductance
    c.info["model"] = "symmetric_inductor"
    c.info["turns"] = N
    c.info["width"] = width
    c.info["spacing"] = spacing
    c.info["diameter"] = d_out
    c.info["center_tap"] = center_tap

    return c


if __name__ == "__main__":
    c = inductor()
    c.show()

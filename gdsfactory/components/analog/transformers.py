"""Transformer components PDK."""

import math

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs

from ._geometry import (
    Poly,
    _add_via_array,
    _map_y,
    _mirror_x,
    _pgs,
    _routing_geometric_45,
    _via_array_at,
    _via_component_info,
    _zip,
)
from .inductors import inductor

__all__ = [
    "get_extended_layer_stack",
    "stacked_transformer",
    "symmetric_transformer",
    "via3",
]


# Symmetric transformer


@gf.cell_with_module_name(tags=["analog"])
def symmetric_transformer(
    d_out: float = 150.0,
    N1: int = 2,
    N2: int = 3,
    sides: int = 8,
    width: float = 7.0,
    spacing: float = 2.0,
    center_tap_primary: bool = False,
    center_tap_secondary: bool = False,
    via_extent: float | None = None,
    port_spacing: float | None = None,
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
    """Symmetric (interleaved) transformer.

    Two interleaved windings (primary: N1 turns, secondary: N2 turns) share
    the same octagonal spiral, alternating turn-by-turn. Crossings are
    routed on layer_underpass wherever a turn boundary needs to jump
    between quadrants; bridges wire together the top/bottom/left/right
    quadrant segments directly on layer_winding.

    Args:
        d_out: Outer diameter of the winding structure, in micrometers.
        N1: Number of turns in the primary winding.
        N2: Number of turns in the secondary winding.
        sides: Number of polygon sides per full turn (8 = octagonal).
            Must be a multiple of 4 (quadrant angle lists use sides // 4).
        width: Metal trace width in micrometers.
        spacing: Gap between adjacent turns in micrometers.
        center_tap_primary: When True, add a center-tap bridge (and CT1 or
            CT2 port, depending on parity) for the primary winding.
        center_tap_secondary: When True, add a center-tap bridge (and CT1
            or CT2 port, depending on parity) for the secondary winding.
        via_extent: Length crossing/bridge routes extend past their
            crossing box on layer_underpass, and the box size used to size
            the crossing/centertap via arrays. If None, it's derived from
            the chosen via's own geometry the same way spiral_inductor and
            symmetric_inductor derive their "extend" value.
        port_spacing: Horizontal spacing of the differential port pairs.
        via: via ComponentSpec connecting winding <-> crossing/centertap.
        resistance: Series resistance in ohms, stored as metadata only.
        inductance: Inductance in henries, stored as metadata only.
        add_pgs: When True, add a patterned ground shield on layers_pgs.
        pgs_diameter: Bounding size D of the ground shield square, in micrometers.
        pgs_width: Strip width w of each ground shield finger, in micrometers.
        pgs_spacing: Gap s between adjacent ground shield fingers, in micrometers.
        layer_winding: Metal layer for the main winding (top metal).
        layer_underpass: Metal layer for crossings and center-tap bridges (one layer below layer_winding).
        layers_pgs: Layers on which the patterned ground shield is drawn,
            kept separate from both layer_winding and layer_underpass

    Returns:
        Component with 4-6 ports:
          P1+ / P1-  ->  primary differential terminals   (bottom, layer_winding)
          P2+ / P2-  ->  secondary differential terminals (top, layer_winding)
          CT1        ->  present if a center tap lands on the bottom side
          CT2        ->  present if a center tap lands on the top side
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

    N = N1 + N2
    Nmin = min(N1, N2)
    N1_end = N - 1 if N1 > N2 else N - abs(N1 - N2) - 1
    N2_end = N - 1 if N1 < N2 else N - abs(N1 - N2) - 1
    v = width / math.cos(PI / sides)
    s = (spacing + width) / math.cos(PI / sides)
    R1_init = d_out / 2 / math.cos(PI / sides)

    ul: list[float] = []
    ur: list[float] = []
    ll: list[float] = []
    lr: list[float] = []
    for i in range(sides // 4):
        t = (i + 0.5) * 2 / sides
        ul.append(PI * (0.5 + t))
        ur.append(PI * (0 + t))
        ll.append(PI * (1 + t))
        lr.append(PI * (1.5 + t))

    sep_total = width + spacing + (SQRT2 - 1) * (2 * spacing + width)

    def rng(a: int, b: int | None = None) -> list[int]:
        return list(range(a)) if b is None else list(range(a, b))

    top_bridge: list[int] = []
    bot_bridge: list[int] = []
    top_crossing: list[int] = []
    bot_crossing: list[int] = []
    if N2 % 2 == 0:
        top_bridge.append(N2_end)
        if N1 % 2 == 0:
            bot_bridge.append(N1_end)
            if N1 >= N2:
                top_crossing += [
                    w for w in rng(N) if w % 2 != 0 and 0 < w < Nmin * 2 - 1
                ]
                top_crossing += [
                    w for w in rng(N) if w % 2 == 0 and N > w > Nmin * 2 - 1
                ]
                bot_crossing += [w for w in rng(N) if w % 2 != 0 and w < N - 1]
            else:
                bot_crossing += [
                    w for w in rng(N) if w % 2 != 0 and 0 < w < Nmin * 2 - 1
                ]
                bot_crossing += [
                    w for w in rng(N) if w % 2 == 0 and N > w > Nmin * 2 - 1
                ]
                top_crossing += [w for w in rng(N) if w % 2 != 0 and w < N - 1]
        else:
            top_bridge.append(N1_end)
            top_crossing += [w for w in rng(N) if w % 2 != 0 and 0 < w < Nmin * 2 - 1]
            top_crossing += [
                w for w in rng(N) if w % 2 == 0 and N - 1 > w > Nmin * 2 - 1
            ]
            bot_crossing += [w for w in rng(N) if w % 2 != 0 and w < N]
    else:
        bot_bridge.append(N2_end)
        if N1 % 2 == 0:
            bot_bridge.append(N1_end)
            top_crossing += [w for w in rng(N) if w % 2 != 0 and 0 < w < N - 1]
            bot_crossing += [
                w for w in rng(N) if w % 2 == 0 and N - 1 > w > Nmin * 2 - 1
            ]
            bot_crossing += [w for w in rng(N) if w % 2 != 0 and w < Nmin * 2 - 1]
        else:
            top_bridge.append(N1_end)
            if N1 >= N2:
                top_crossing += [w for w in rng(N) if w % 2 != 0 and w < N - 1]
                bot_crossing += [w for w in rng(N) if w % 2 != 0 and w < Nmin * 2 - 1]
                bot_crossing += [
                    w for w in rng(N) if w % 2 == 0 and N - 1 > w > Nmin * 2 - 1
                ]
            else:
                top_crossing += [
                    w for w in rng(N) if w % 2 == 0 and N - 1 > w > Nmin * 2 - 1
                ]
                top_crossing += [w for w in rng(N) if w % 2 != 0 and w < Nmin * 2 - 1]
                bot_crossing += [w for w in rng(N) if w % 2 != 0 and w < Nmin * 2]
                bot_crossing += [
                    w for w in rng(N) if w % 2 != 0 and N - 1 > w > Nmin * 2 - 1
                ]
    lr_bridge = [w - 1 for w in rng(1, N + 1) if w > 2 * Nmin]
    lr_crossing = [w - 1 for w in rng(1, N + 1) if w % 2 != 0 and w < 2 * Nmin]

    def add_winding(poly: Poly) -> None:
        c.add_polygon(poly, layer=layer_winding)

    def add_crossing(poly: Poly) -> None:
        c.add_polygon(poly, layer=layer_underpass)

    via_centers_tct: list[tuple[float, float]] = []

    R1 = R1_init
    R2 = R1 - v
    for winding in range(N):
        all_angles = [ul, ll, ur, lr]
        for qi in range(4):
            angs = all_angles[qi]
            x_out = [R1 * math.cos(p) for p in angs]
            y_out = [R1 * math.sin(p) for p in angs]
            x_in = [R2 * math.cos(p) for p in angs]
            y_in = [R2 * math.sin(p) for p in angs]
            if qi == 0:
                y_out = [y_out[0], *y_out, sep_total / 2]
                y_in = [y_in[0], *y_in, sep_total / 2]
                x_out = [-sep_total / 2, *x_out, x_out[-1]]
                x_in = [-sep_total / 2, *x_in, x_in[-1]]
            elif qi == 1:
                y_out = [-sep_total / 2, *y_out, y_out[-1]]
                y_in = [-sep_total / 2, *y_in, y_in[-1]]
                x_out = [x_out[0], *x_out, -sep_total / 2]
                x_in = [x_in[0], *x_in, -sep_total / 2]
            elif qi == 2:
                y_out = [sep_total / 2, *y_out, y_out[-1]]
                y_in = [sep_total / 2, *y_in, y_in[-1]]
                x_out = [x_out[0], *x_out, sep_total / 2]
                x_in = [x_in[0], *x_in, sep_total / 2]
            else:
                y_out = [y_out[0], *y_out, -sep_total / 2]
                y_in = [y_in[0], *y_in, -sep_total / 2]
                x_out = [sep_total / 2, *x_out, x_out[-1]]
                x_in = [sep_total / 2, *x_in, x_in[-1]]
            add_winding(_zip([*x_out, *reversed(x_in)], [*y_out, *reversed(y_in)]))

        if winding in bot_bridge:
            h = -R2 * math.sin(PI * (0.5 - 1 / sides))
            add_winding(
                _zip(
                    [-sep_total / 2, sep_total / 2, sep_total / 2, -sep_total / 2],
                    [h, h, h - width, h - width],
                )
            )
        if winding in top_bridge:
            h = (R2 + v) * math.sin(PI * (0.5 - 1 / sides))
            add_winding(
                _zip(
                    [-sep_total / 2, sep_total / 2, sep_total / 2, -sep_total / 2],
                    [h, h, h - width, h - width],
                )
            )
        if winding in lr_bridge:
            hR = (R2 + v) * math.sin(PI * (0.5 - 1 / sides))
            add_winding(
                _zip(
                    [hR, hR, hR - width, hR - width],
                    [-sep_total / 2, sep_total / 2, sep_total / 2, -sep_total / 2],
                )
            )
            hL = -R2 * math.sin(PI * (0.5 - 1 / sides))
            add_winding(
                _zip(
                    [hL, hL, hL - width, hL - width],
                    [-sep_total / 2, sep_total / 2, sep_total / 2, -sep_total / 2],
                )
            )

        if winding in top_crossing:
            h = R1 * math.sin(PI * (0.5 - 1 / sides))
            add_crossing(
                _routing_geometric_45(
                    width, spacing, 0, h - width - spacing / 2, extend
                )
            )
            ct = _routing_geometric_45(width, spacing, 0, h - width - spacing / 2, 0)
            add_winding(_mirror_x(ct))
        if winding in bot_crossing:
            h = (-R2 + s) * math.sin(PI * (0.5 - 1 / sides))
            add_crossing(
                _routing_geometric_45(
                    width, spacing, 0, h - width - spacing / 2, extend
                )
            )
            ct = _routing_geometric_45(width, spacing, 0, h - width - spacing / 2, 0)
            add_winding(_mirror_x(ct))
        if winding in lr_crossing:
            hR = R1 * math.sin(PI * (0.5 - 1 / sides))
            cr = _routing_geometric_45(
                width, spacing, 0, hR - width - spacing / 2, extend
            )
            add_crossing([(y, x) for (x, y) in cr])
            cr = _routing_geometric_45(width, spacing, 0, hR - width - spacing / 2, 0)
            add_winding([(-y, x) for (x, y) in cr])
            hL = (-R2 + s) * math.sin(PI * (0.5 - 1 / sides))
            cr = _routing_geometric_45(
                width, spacing, 0, hL - width - spacing / 2, extend
            )
            add_crossing([(y, x) for (x, y) in cr])
            cr = _routing_geometric_45(width, spacing, 0, hL - width - spacing / 2, 0)
            add_winding([(-y, x) for (x, y) in cr])

        # Crossing vias — recessed under the layer_underpass crossing strips above.
        h_top = R1 * math.sin(PI * (0.5 - 1 / sides))
        h_bot = (-R2 + s) * math.sin(PI * (0.5 - 1 / sides))
        if winding in top_crossing:
            _via_array_at(
                c,
                via_component,
                -sep_total / 2 - width / 2,
                h_top - 3 * width / 2 - spacing,
                extend,
                width,
                via_enclosure,
            )
            _via_array_at(
                c,
                via_component,
                sep_total / 2 + width / 2,
                h_top - width / 2,
                extend,
                width,
                via_enclosure,
            )
        if winding in bot_crossing:
            _via_array_at(
                c,
                via_component,
                -sep_total / 2 - width / 2,
                h_bot - 3 * width / 2 - spacing,
                extend,
                width,
                via_enclosure,
            )
            _via_array_at(
                c,
                via_component,
                sep_total / 2 + width / 2,
                h_bot - width / 2,
                extend,
                width,
                via_enclosure,
            )
        if winding in lr_crossing:
            _via_array_at(
                c,
                via_component,
                h_bot - 3 * width / 2 - spacing,
                -sep_total / 2 - width / 2,
                extend,
                width,
                via_enclosure,
            )
            _via_array_at(
                c,
                via_component,
                h_bot - width / 2,
                sep_total / 2 + width / 2,
                extend,
                width,
                via_enclosure,
            )
            _via_array_at(
                c,
                via_component,
                h_top - 3 * width / 2 - spacing,
                -sep_total / 2 - width / 2,
                extend,
                width,
                via_enclosure,
            )
            _via_array_at(
                c,
                via_component,
                h_top - width / 2,
                sep_total / 2 + width / 2,
                extend,
                width,
                via_enclosure,
            )

        R1 -= s
        R2 -= s

    # Center taps
    def add_ct(n_end: int, ends_bottom: bool) -> None:
        _ext = min(width, extend)
        if ends_bottom:
            x_ct = [-width / 2, -width / 2, width / 2, width / 2]
            y_ct = [
                -d_out / 2 + width - _ext,
                -d_out / 2 + (spacing + width) * n_end,
                -d_out / 2 + (spacing + width) * n_end,
                -d_out / 2 + width - _ext,
            ]
            x_ct1, y_ct1 = (
                0,
                -d_out / 2 + spacing * n_end + width * (n_end + 1) - width + _ext / 2,
            )
            x_ct2, y_ct2 = 0, -d_out / 2 + width / 2 + (width - _ext) / 2
        else:
            x_ct = [width / 2, width / 2, -width / 2, -width / 2]
            y_ct = [
                d_out / 2 - width + _ext,
                d_out / 2 - (spacing + width) * n_end,
                d_out / 2 - (spacing + width) * n_end,
                d_out / 2 - width + _ext,
            ]
            x_ct1, y_ct1 = (
                0,
                d_out / 2 - spacing * n_end - width * (n_end + 1) + width - _ext / 2,
            )
            x_ct2, y_ct2 = 0, d_out / 2 - width / 2 - (width - _ext) / 2

        if n_end > 1:
            via_centers_tct.append((x_ct1, y_ct1))
            via_centers_tct.append((x_ct2, y_ct2))
            xvp1 = [
                x_ct1 - width / 2,
                x_ct1 - width / 2,
                x_ct1 + width / 2,
                x_ct1 + width / 2,
            ]
            yvp1 = [
                y_ct1 - _ext / 2,
                y_ct1 + _ext / 2,
                y_ct1 + _ext / 2,
                y_ct1 - _ext / 2,
            ]
            xvp2 = [
                x_ct2 - width / 2,
                x_ct2 - width / 2,
                x_ct2 + width / 2,
                x_ct2 + width / 2,
            ]
            yvp2 = [
                y_ct2 - _ext / 2,
                y_ct2 + _ext / 2,
                y_ct2 + _ext / 2,
                y_ct2 - _ext / 2,
            ]
            add_winding(_zip(xvp1, yvp1))
            add_crossing(_zip(xvp1, yvp1))
            add_crossing(_zip(xvp2, yvp2))
            if n_end > 2:
                add_crossing(_zip(x_ct, y_ct))
                add_crossing(_zip(xvp1, yvp1))
                add_crossing(_zip(xvp2, yvp2))
            else:
                add_crossing(_zip(x_ct, y_ct))
        else:
            add_winding(_zip(x_ct, y_ct))

    if center_tap_primary:
        add_ct(N1_end, N1 % 2 == 0)
    if center_tap_secondary:
        add_ct(N2_end, N2 % 2 != 0)

    # Ports
    has_bottom_ct = (center_tap_primary and N1 % 2 == 0) or (
        center_tap_secondary and N2 % 2 != 0
    )
    has_top_ct = (center_tap_primary and N1 % 2 != 0) or (
        center_tap_secondary and N2 % 2 == 0
    )
    bpx = ps + width if has_bottom_ct else (ps + width) / 2
    tpx = ps + width if has_top_ct else (ps + width) / 2

    x_port_b = [
        -sep_total / 2,
        -bpx + width / 2,
        -bpx + width / 2,
        -bpx - width / 2,
        -bpx - width / 2,
        -sep_total / 2,
    ]
    y_port_b = [
        -d_out / 2 + width,
        -d_out / 2 + width,
        -d_out / 2 - width,
        -d_out / 2 - width,
        -d_out / 2,
        -d_out / 2,
    ]
    if has_bottom_ct:
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
    add_winding(_zip(x_port_b, y_port_b))
    add_winding(_zip([-x for x in x_port_b], y_port_b))

    x_port_t = [
        -sep_total / 2,
        -tpx + width / 2,
        -tpx + width / 2,
        -tpx - width / 2,
        -tpx - width / 2,
        -sep_total / 2,
    ]
    y_port_t = [
        -d_out / 2 + width,
        -d_out / 2 + width,
        -d_out / 2 - width,
        -d_out / 2 - width,
        -d_out / 2,
        -d_out / 2,
    ]
    if has_top_ct:
        add_winding(
            _zip(
                [-width / 2, -width / 2, width / 2, width / 2],
                [
                    d_out / 2 + width,
                    d_out / 2 - width,
                    d_out / 2 - width,
                    d_out / 2 + width,
                ],
            )
        )
    add_winding(_zip(x_port_t, [-y for y in y_port_t]))
    add_winding(_zip([-x for x in x_port_t], [-y for y in y_port_t]))

    # Center-tap vias
    ext_ct = min(width, extend)
    for cx, cy in via_centers_tct:
        _add_via_array(
            c,
            via_component,
            cx,
            cy,
            width - 2 * via_enclosure,
            ext_ct - 2 * via_enclosure,
        )

    bot_y = -d_out / 2 - width
    top_y = d_out / 2 + width

    c.add_port(
        "P1+", center=(-bpx, bot_y), width=width, orientation=270.0, layer=layer_winding
    )
    c.add_port(
        "P1-", center=(bpx, bot_y), width=width, orientation=270.0, layer=layer_winding
    )
    c.add_port(
        "P2+", center=(-tpx, top_y), width=width, orientation=90.0, layer=layer_winding
    )
    c.add_port(
        "P2-", center=(tpx, top_y), width=width, orientation=90.0, layer=layer_winding
    )
    if has_bottom_ct:
        c.add_port(
            "CT1",
            center=(0, bot_y),
            width=width,
            orientation=270.0,
            layer=layer_winding,
        )
    if has_top_ct:
        c.add_port(
            "CT2", center=(0, top_y), width=width, orientation=90.0, layer=layer_winding
        )

    if add_pgs:
        for layer in layers_pgs:
            for strip in _pgs(pgs_diameter, pgs_width, pgs_spacing):
                c.add_polygon(strip, layer=layer)

    # Metadata (mirrors inductor() / spiral_inductor() / symmetric_inductor())
    c.info["resistance"] = resistance
    c.info["inductance"] = inductance
    c.info["model"] = "symmetric_transformer"
    c.info["turns_primary"] = N1
    c.info["turns_secondary"] = N2
    c.info["width"] = width
    c.info["spacing"] = spacing
    c.info["diameter"] = d_out
    c.info["center_tap_primary"] = center_tap_primary
    c.info["center_tap_secondary"] = center_tap_secondary

    return c


# Stacked transformer
# ---------------------------------------------------------------------------
# 4th metal (M4) + VIA3, used by stacked_transformer's DEFAULT full-isolation config.
# The gdsfactory generic PDK only has 3 real metals (M1/M2/M3).
# M4_LAYER/VIA3_LAYER are synthetic: valid GDS layers (KLayout accepts any registered layer number)
# but NOT part of this PDK's real, fabricatable metal stack or its get_layer_stack() z-geometry.
#
# *** SIMULATION PIPELINE WARNING ***
# If you're feeding a stacked_transformer() default-config component into
# the gsim/Palace EM simulation pipeline, sim.set_stack(substrate_thickness=...)
# will NOT know M4 has any zmin/thickness/material, since get_layer_stack() has no metal4 entry.
# You MUST instead call:
#     sim.set_stack(stack=get_extended_layer_stack(), substrate_thickness=...)
# using get_extended_layer_stack() so the mesher has real z-geometry for M4/VIA3.
# This only matters for simulation, plain GDS layout/export works fine with the defaults as-is.
# ---------------------------------------------------------------------------

_nm = 1e-3


# Raw GDS layer registration. Picked layer/datatype numbers (53,0) and (48,0) as unused slots in this PDK's map;
# change these if they collide with something in the layer map.
def _m4_layer() -> int:
    return gf.kcl.layer(53, 0)


def _via3_layer() -> int:
    return gf.kcl.layer(48, 0)


@gf.cell
def via3(
    size: tuple[float, float] = (0.7, 0.7),
    enclosure: float = 1.0,
    pitch: float = 2.0,
) -> Component:
    """Via connecting M3 <-> M4, mirroring via1/via2/viac's shape (0.7x0.7um squares, 1um enclosure, 2um pitch) but on VIA3_LAYER.

    Only meaningful once M4 also has real z-geometry — see
    get_extended_layer_stack() for the matching LayerStack entry.
    """
    c = Component()
    w, h = size
    c.add_polygon(
        [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)],
        layer=_via3_layer(),
    )
    c.info["xsize"] = w
    c.info["ysize"] = h
    c.info["enclosure"] = enclosure
    c.info["column_pitch"] = pitch
    c.info["row_pitch"] = pitch
    return c


def get_extended_layer_stack(
    thickness_metal4: float = 700 * _nm,
    thickness_via3: float | None = None,
) -> "LayerStack":
    """Return the active generic PDK's LayerStack with metal4 + via3 appended, so Palace/gsim mesher knows M4's z-position, thickness, and material.

    Since M4 doesn't exist in the generic PDK's real stack, there's no measured
    metal3-to-metal4 dielectric gap to copy.
    thickness_via3 defaults to the PDK's actual via2 thickness (metal2<->metal3, ~0.2um)
    as the closest real analogue — via1 (metal1<->metal2, ~0.5um) is thicker,
    consistent with vias generally thinning higher up the stack, so via2's
    value is the more defensible guess for a via directly above it.
    This is an assumption, not a measured value — override thickness_via3
    if there is a better estimate for the actual process.

    metal4 sits thickness_via3 above metal3's top, with via3 filling that
    real dielectric gap (matching how via1/via2 physically work in this
    stack — metal layers don't touch directly; a via plug bridges a
    nonzero-thickness interlayer dielectric between them).

    Must be called with the generic PDK active (gf.gpdk.get_generic_pdk()
    .activate(), already done at import time in this module) so
    get_layer_stack() resolves the base metal1/2/3 stack correctly.

    Example:
        >>> sim.set_stack(stack=get_extended_layer_stack(), substrate_thickness=180.0)
    """
    from gdsfactory.technology import LayerLevel, LogicalLayer

    stack_mod = gf.gpdk.layer_stack
    params = stack_mod.LayerStackParameters
    base = stack_mod.get_layer_stack()

    metal3_top = params.zmin_metal3 + params.thickness_metal3

    if thickness_via3 is None:
        # via2's real thickness in this PDK: metal3.zmin - metal2_top
        metal2_top = params.zmin_metal2 + params.thickness_metal2
        thickness_via3 = params.zmin_metal3 - metal2_top

    zmin_metal4 = metal3_top + thickness_via3

    base.layers["via3"] = LayerLevel(
        layer=LogicalLayer(layer=_via3_layer()),
        thickness=thickness_via3,
        zmin=metal3_top,
        material="Aluminum",
        mesh_order=1,
    )
    base.layers["metal4"] = LayerLevel(
        layer=LogicalLayer(layer=_m4_layer()),
        thickness=thickness_metal4,
        zmin=zmin_metal4,
        material="Aluminum",
        mesh_order=2,
    )
    return base


def _stacked_half(
    c: Component,
    *,
    N: int,
    center_tap: bool,
    d_out: float,
    sides: int,
    width: float,
    spacing: float,
    R1_start: float,
    extend: float,
    ps: float,
    winding_layer: LayerSpec,
    crossing_layer: LayerSpec,
    via_component: Component,
    via_enclosure: float,
    sign: int,
    port_prefix: str,
) -> None:
    """Build one winding half (primary or secondary) of a stacked transformer directly onto Component c.

    sign=+1 places it at the bottom as-is;
    sign=-1 flips every y-coordinate, since every shape here is y-symmetric under
    negation, negating y per-point at draw time is equivalent to mirroring the
    assembled set afterwards.
    """
    PI = math.pi
    SQRT2 = math.sqrt(2)
    v = width / math.cos(PI / sides)
    s = (spacing + width) / math.cos(PI / sides)
    sep_total = width + spacing + (SQRT2 - 1) * (2 * spacing + width)

    n_half = sides // 2
    left_angles = [PI * (0.5 + (i + 0.5) * 2 / sides) for i in range(n_half)]
    right_angles = [PI * (-0.5 + (i + 0.5) * 2 / sides) for i in range(n_half)]

    def add_w(poly: Poly) -> None:
        c.add_polygon(_map_y(poly, lambda y: sign * y), layer=winding_layer)

    def add_c(poly: Poly) -> None:
        c.add_polygon(_map_y(poly, lambda y: sign * y), layer=crossing_layer)

    R1 = R1_start
    R2 = R1 - v
    for winding in range(N):
        for angles, left in ((left_angles, True), (right_angles, False)):
            x_out = [R1 * math.cos(p) for p in angles]
            y_out = [R1 * math.sin(p) for p in angles]
            x_in = [R2 * math.cos(p) for p in angles]
            y_in = [R2 * math.sin(p) for p in angles]
            if winding == N - 1:
                if left:
                    if N % 2 == 0:
                        x_out = [-sep_total / 2, *x_out, 0]
                        x_in = [-sep_total / 2, *x_in, 0]
                    else:
                        x_out = [0, *x_out, -sep_total / 2]
                        x_in = [0, *x_in, -sep_total / 2]
                else:
                    if N % 2 == 0:
                        x_out = [0, *x_out, sep_total / 2]
                        x_in = [0, *x_in, sep_total / 2]
                    else:
                        x_out = [sep_total / 2, *x_out, 0]
                        x_in = [sep_total / 2, *x_in, 0]
            else:
                sgn = -1 if left else 1
                x_out = [sgn * sep_total / 2, *x_out, sgn * sep_total / 2]
                x_in = [sgn * sep_total / 2, *x_in, sgn * sep_total / 2]
            y_out = [y_out[0], *y_out, y_out[-1]]
            y_in = [y_in[0], *y_in, y_in[-1]]
            add_w(_zip([*x_out, *reversed(x_in)], [*y_out, *reversed(y_in)]))

        if winding != N - 1:
            if winding % 2 == 0:
                h = R1 * math.sin(PI * (0.5 - 1 / sides))
            else:
                h = (-R2 + s) * math.sin(PI * (0.5 - 1 / sides))

            add_c(
                _routing_geometric_45(
                    width, spacing, 0, h - width - spacing / 2, extend
                )
            )
            ct = _routing_geometric_45(width, spacing, 0, h - width - spacing / 2, 0)
            add_w(_mirror_x(ct))

            for cx, cy in [
                (-sep_total / 2 - width / 2, h - 3 * width / 2 - spacing),
                (sep_total / 2 + width / 2, h - width / 2),
            ]:
                dx = math.copysign(1, cx) * (extend - width) / 2
                _add_via_array(
                    c,
                    via_component,
                    cx + dx,
                    sign * cy,
                    extend - 2 * via_enclosure,
                    width - 2 * via_enclosure,
                )

        R1 -= s
        R2 -= s

    ct_port_layer = winding_layer
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
            add_w(_zip(x_ct, y_ct))
        else:
            ct_port_layer = crossing_layer
            add_c(_zip(x_ct, y_ct))

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

            add_w(_zip(xvp1, yvp1))
            add_c(_zip(xvp1, yvp1))
            add_c(_zip(xvp2, yvp2))

            for cx, cy in [(x_ct1, y_ct1), (x_ct2, y_ct2)]:
                _add_via_array(
                    c,
                    via_component,
                    cx,
                    sign * cy,
                    width - 2 * via_enclosure,
                    extend - 2 * via_enclosure,
                )

    # Ports (always built in "bottom" orientation, then sign-flipped above)
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
        add_w(
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
    add_w(_zip(x_port, y_port))
    add_w(_zip([-x for x in x_port], y_port))

    port_marker_y = sign * (-d_out / 2 - width)
    orientation = 270.0 if sign > 0 else 90.0

    c.add_port(
        f"{port_prefix}+",
        center=(-pxo, port_marker_y),
        width=width,
        orientation=orientation,
        layer=winding_layer,
    )
    c.add_port(
        f"{port_prefix}-",
        center=(pxo, port_marker_y),
        width=width,
        orientation=orientation,
        layer=winding_layer,
    )
    if center_tap:
        c.add_port(
            f"CT_{port_prefix}",
            center=(0, port_marker_y),
            width=width,
            orientation=orientation,
            layer=ct_port_layer,
        )


@gf.cell_with_module_name(tags=["analog"])
def stacked_transformer(
    d_out: float = 150.0,
    N1: int = 3,
    N2: int = 3,
    sides: int = 8,
    width: float = 10.0,
    spacing: float = 2.0,
    center_tap_primary: bool = False,
    center_tap_secondary: bool = False,
    via_extent: float | None = None,
    port_spacing: float | None = None,
    via_primary: ComponentSpec = via3,
    via_secondary: ComponentSpec = "via1",
    resistance: float = 0.5777,
    inductance: float = 33.303e-12,
    add_pgs: bool = False,
    pgs_diameter: float = 180.0,
    pgs_width: float = 4.0,
    pgs_spacing: float = 2.0,
    # Defaults use the synthetic M4/VIA3 layers for full 4-metal isolation —
    # see the module-level comment above M4_LAYER for the SIMULATION
    # PIPELINE WARNING: use get_extended_layer_stack() with sim.set_stack(),
    # not the PDK default, when meshing a component built with these.
    layer_winding_primary: LayerSpec | None = None,
    layer_crossing_primary: LayerSpec = "M3",
    layer_winding_secondary: LayerSpec = "M2",
    layer_crossing_secondary: LayerSpec = "M1",
    layers_pgs: LayerSpecs = (),
) -> Component:
    """Stacked transformer.

    The primary winding sits on layer_winding_primary with
    its crossings on layer_crossing_primary (connected by via_primary);
    the secondary winding sits on layer_winding_secondary with its
    crossings on layer_crossing_secondary (connected by via_secondary),
    mirrored to sit on the opposite side of d_out so the two windings
    stack vertically over the same footprint.

    LAYER STACK: this gdsfactory generic PDK only has 3 real metals
    (M1/M2/M3). A fully isolated stacked transformer needs 4 independent
    metals (primary winding, primary crossing, secondary winding,
    secondary crossing), so the defaults here use an EXTRA, non-native
    4th metal registered via gf.kcl.layer() at import time (M4_LAYER,
    layer (53, 0)) with its own via (via3(), VIA3_LAYER, layer (48, 0))
    connecting it to M3. This gives full primary/secondary isolation by
    default:
        layer_winding_primary=M4_LAYER, layer_crossing_primary="M3", via_primary=via3
        layer_winding_secondary="M2",   layer_crossing_secondary="M1", via_secondary="via1"

    M4_LAYER/VIA3_LAYER are valid GDS layers but are NOT part of this PDK's real,
    fabricatable metal stack. If you need this component to be simulation-ready
    (Palace/gsim meshing) ortape-out-accurate, call get_extended_layer_stack()
    instead of the PDK's default stack, e.g. sim.set_stack(stack=get_extended_layer_stack()).

    If your ACTUAL target PDK has a genuine 4th metal, pass its real
    layer name/via in place of M4_LAYER/via3 instead of relying on this
    synthetic one.

    layers_pgs defaults to an empty tuple: even with M4 in play, all of
    M1/M2/M3/M4 are live signal layers in the default configuration, so
    there's still no obviously-safe spare metal for a patterned ground
    shield — pass an explicit LayerSpecs of your own (accepting whatever
    coupling that implies) if you need one anyway.

    Args:
        d_out: Outer diameter of each winding, in micrometers.
        N1: Number of turns in the primary winding.
        N2: Number of turns in the secondary winding.
        sides: Number of polygon sides per full turn (8 = octagonal).
        width: Metal trace width in micrometers.
        spacing: Gap between adjacent turns in micrometers.
        center_tap_primary: When True, add a center-tap bridge and CT_P
            port to the primary winding.
        center_tap_secondary: When True, add a center-tap bridge and CT_S
            port to the secondary winding.
        via_extent: Length crossing routes extend past their crossing box,
            and the box size used to size the crossing/centertap via
            arrays, for both halves. If None, it's derived independently
            for each half from that half's own via geometry (same
            derivation spiral_inductor/symmetric_inductor use).
        port_spacing: Horizontal spacing of both differential port pairs.
            Defaults to spacing.
        via_primary: via ComponentSpec connecting layer_winding_primary <-> layer_crossing_primary.
        via_secondary: via ComponentSpec connecting layer_winding_secondary <-> layer_crossing_secondary.
        resistance: Series resistance in ohms, stored as metadata only.
        inductance: Inductance in henries, stored as metadata only.
        add_pgs: When True, add a patterned ground shield on layers_pgs.
            See the layer-stack caveat above before enabling this.
        pgs_diameter: Bounding size D of the ground shield square, in micrometers.
        pgs_width: Strip width w of each ground shield finger, in micrometers.
        pgs_spacing: Gap s between adjacent ground shield fingers, in micrometers.
        layer_winding_primary: Metal layer for the primary winding.
        layer_crossing_primary: Metal layer for the primary crossings.
        layer_winding_secondary: Metal layer for the secondary winding.
        layer_crossing_secondary: Metal layer for the secondary crossings.
        layers_pgs: Layers on which the patterned ground shield is drawn.

    Returns:
        Component with 4-6 ports:
          P+ / P-    ->  primary differential terminals   (layer_winding_primary)
          S+ / S-    ->  secondary differential terminals (layer_winding_secondary)
          CT_P       ->  present if center_tap_primary=True
          CT_S       ->  present if center_tap_secondary=True
    """
    if layer_winding_primary is None:
        layer_winding_primary = _m4_layer()

    c = Component()
    PI = math.pi
    R1_init = d_out / 2 / math.cos(PI / sides)
    ps = spacing if port_spacing is None else port_spacing

    via_primary_component = gf.get_component(via_primary)
    _, via_p_h, via_p_enclosure, _, via_p_pitch_y = _via_component_info(
        via_primary_component
    )
    if via_extent is None:
        extend_primary = 2 * (via_p_h + via_p_enclosure) + (via_p_pitch_y - via_p_h)
    else:
        extend_primary = via_extent

    via_secondary_component = gf.get_component(via_secondary)
    _, via_s_h, via_s_enclosure, _, via_s_pitch_y = _via_component_info(
        via_secondary_component
    )
    if via_extent is None:
        extend_secondary = 2 * (via_s_h + via_s_enclosure) + (via_s_pitch_y - via_s_h)
    else:
        extend_secondary = via_extent

    _stacked_half(
        c,
        N=N1,
        center_tap=center_tap_primary,
        d_out=d_out,
        sides=sides,
        width=width,
        spacing=spacing,
        R1_start=R1_init,
        extend=extend_primary,
        ps=ps,
        winding_layer=layer_winding_primary,
        crossing_layer=layer_crossing_primary,
        via_component=via_primary_component,
        via_enclosure=via_p_enclosure,
        sign=1,
        port_prefix="P",
    )
    _stacked_half(
        c,
        N=N2,
        center_tap=center_tap_secondary,
        d_out=d_out,
        sides=sides,
        width=width,
        spacing=spacing,
        R1_start=R1_init,
        extend=extend_secondary,
        ps=ps,
        winding_layer=layer_winding_secondary,
        crossing_layer=layer_crossing_secondary,
        via_component=via_secondary_component,
        via_enclosure=via_s_enclosure,
        sign=-1,
        port_prefix="S",
    )

    if add_pgs:
        if not layers_pgs:
            raise ValueError(
                "add_pgs=True but layers_pgs is empty — this PDK's default "
                "layer stack has no spare metal for a shield in "
                "stacked_transformer (see docstring). Pass an explicit "
                "layers_pgs if you want one anyway."
            )
        for layer in layers_pgs:
            for strip in _pgs(pgs_diameter, pgs_width, pgs_spacing):
                c.add_polygon(strip, layer=layer)

    c.info["resistance"] = resistance
    c.info["inductance"] = inductance
    c.info["model"] = "stacked_transformer"
    c.info["turns_primary"] = N1
    c.info["turns_secondary"] = N2
    c.info["width"] = width
    c.info["spacing"] = spacing
    c.info["diameter"] = d_out
    c.info["center_tap_primary"] = center_tap_primary
    c.info["center_tap_secondary"] = center_tap_secondary

    return c


# Concentric (single-turn, coplanar) 1:1 transformer


@gf.cell
def _secondary_inductor(
    width: float = 3.0,
    space: float = 3.1,
    diameter: float = 50.0,
    layer_metal: LayerSpec = "M2",
    layer_jumper: LayerSpec = "M1",
    via: ComponentSpec = "via1",
    via_size: float = 3.0,
) -> Component:
    """Single-turn octagonal.

    The two leads jump to a lower metal (layer_jumper) right where
    they meet the coil body, so the leads can be routed straight out
    past an outer coil without shorting it — internal helper defining
    the secondary of transformer_concentric only, not a standalone
    component in its own right.

    Args:
        width: Metal trace width in micrometers.
        space: Space between the coil and the leads/gap, in micrometers.
        diameter: Coil diameter in micrometers.
        layer_metal: Layer for the coil body.
        layer_jumper: Layer for the two leads (must be via-connectable
            to layer_metal).
        via: via ComponentSpec connecting layer_jumper <-> layer_metal.
        via_size: Side length of the square via_stack junction pad, in
            micrometers. via_stack() requires this to be large enough to
            enclose at least one via square with its enclosure margin —
            too small raises a ValueError from via_stack() itself.

    Returns:
        Component with ports P1, P2 on layer_jumper.
    """
    w = width
    s = space
    d = diameter
    r = d / 2 + s
    octagon_center_y = 3 * r
    pi_over_4 = math.radians(45)

    path_points = [(+space / 2, octagon_center_y - r * math.cos(pi_over_4 / 2))]
    for i in range(-2, 6):
        angle = i * pi_over_4 + pi_over_4 / 2
        r = d / 2 + s
        x = r * math.cos(angle)
        y = r * math.sin(angle) + octagon_center_y
        path_points.append((x, y))
    path_points.append((-space / 2, octagon_center_y - r * math.cos(pi_over_4 / 2)))
    gap_y = octagon_center_y - r * math.cos(pi_over_4 / 2)

    path = gf.Path(path_points)
    c = gf.path.extrude(path, layer=layer_metal, width=w)

    length = 2 * r + s

    lead1 = c << gf.components.rectangle(size=(s, length), layer=layer_jumper)
    lead1.move((-s - s / 2, 0))
    c.add_port(name="P1", center=(-s, s), width=s, orientation=270, layer=layer_jumper)

    lead2 = c << gf.components.rectangle(size=(s, length), layer=layer_jumper)
    lead2.move((s - s / 2, 0))
    c.add_port(name="P2", center=(s, s), width=s, orientation=270, layer=layer_jumper)

    via_stack_component = gf.get_component(
        "via_stack",
        size=(via_size, via_size),
        layers=(layer_jumper, layer_metal),
        vias=(None, via),
    )
    junction1 = c.add_ref(via_stack_component)
    junction1.move(junction1.center, (-space, gap_y))
    junction2 = c.add_ref(via_stack_component)
    junction2.move(junction2.center, (space, gap_y))

    c.flatten()
    return c


@gf.cell_with_module_name(tags=["analog"])
def transformer_concentric(
    width_primary: float = 3.0,
    width_secondary: float = 3.0,
    space: float = 3.1,
    coupling_gap: float = 4.0,
    diameter_outer: float = 80.0,
    layer_primary: LayerSpec = "M3",
    layer_secondary: LayerSpec = "M2",
    layer_secondary_jumper: LayerSpec = "M1",
    via_secondary: ComponentSpec = "via1",
    via_size: float | None = None,
    layer_inductor: LayerSpec = "M1",
    layers_no_fill: LayerSpecs = ("DEVREC", "NO_TILE_SI"),
    add_pgs: bool = False,
    pgs_diameter: float = 120.0,
    pgs_width: float = 4.0,
    pgs_spacing: float = 2.0,
    layers_pgs: LayerSpecs = (),
) -> Component:
    """Concentric, coplanar 1:1 transformer (single-turn coils).

    Primary: standard inductor(), outer ring, on layer_primary.
    Secondary: an internal single-turn coil helper (not
    exposed as its own component), inner ring, whose leads are drawn on
    layer_secondary_jumper instead of layer_secondary, so they pass
    underneath the primary ring without colliding — a via_stack connects
    each lead to the coil body right where they meet.


    Args:
        width_primary: Primary coil trace width, in micrometers.
        width_secondary: Secondary coil trace width, in micrometers.
        space: Space between adjacent turns/leads, in micrometers
            (shared by both coils, matching the original).
        coupling_gap: Radial gap between the primary's inner edge and
            the secondary's outer edge, in micrometers.
        diameter_outer: Primary coil's outer diameter, in micrometers.
        layer_primary: Metal layer for the primary coil.
        layer_secondary: Metal layer for the secondary coil body.
        layer_secondary_jumper: Metal layer for the secondary's leads (via-connected to layer_secondary).
        via_secondary: via ComponentSpec connecting layer_secondary_jumper <-> layer_secondary.
        via_size: Side length of the secondary's via_stack junction pads,
            in micrometers. Defaults to width_secondary if None — bump
            this up if via_stack() raises an enclosure ValueError.
        layer_inductor: Marker layer for the outer IND-style polygon
            drawn around each coil (matches inductor()'s own
            layer_inductor role).
        layers_no_fill: Layers excluded from metal fill, drawn under
            the same outer marker polygon.
        add_pgs: When True, add a patterned ground shield on layers_pgs.
        pgs_diameter: Bounding size D of the ground shield square, in micrometers.
        pgs_width: Strip width w of each ground shield finger, in micrometers.
        pgs_spacing: Gap s between adjacent ground shield fingers, in micrometers.
        layers_pgs: Layers on which the patterned ground shield is drawn.
            Empty by default: layer_primary/layer_secondary/
            layer_secondary_jumper already consume all 3 of this PDK's
            real metals in the default configuration, so there's no
            obviously-safe spare layer — pass one explicitly (accepting
            whatever coupling that implies) if you need a shield anyway.

    Returns:
        Component with ports P1, P2 (primary, on layer_primary) and
        S1, S2 (secondary, on layer_secondary_jumper).
    """
    c = gf.Component()
    via_size = via_size or width_secondary

    # Primary coil (outer ring), standard inductor
    primary = inductor(
        width=width_primary,
        space=space,
        diameter=diameter_outer,
        turns=1,
        layer_metal=layer_primary,
        layer_inductor=layer_inductor,
        layer_metal_pin=layer_primary,
        layers_no_fill=layers_no_fill,
    )
    prim_ref = c.add_ref(primary)
    cx, cy = prim_ref.center
    prim_ref.move((-cx, -cy))

    # Inner diameter for secondary, leaving room for coupling_gap between
    # the primary's inner edge and the secondary's outer edge.
    diameter_secondary = diameter_outer - 2 * (width_primary + space) - 2 * coupling_gap

    secondary = _secondary_inductor(
        width=width_secondary,
        space=space,
        diameter=diameter_secondary,
        layer_metal=layer_secondary,
        layer_jumper=layer_secondary_jumper,
        via=via_secondary,
        via_size=via_size,
    )
    sec_ref = c.add_ref(secondary)
    sec_ref.rotate(180)
    cx, cy = sec_ref.center
    sec_ref.move((-cx, -cy))

    # Expose all 4 ports
    c.add_port(name="P1", port=prim_ref.ports["P1"])
    c.add_port(name="P2", port=prim_ref.ports["P2"])
    c.add_port(name="S1", port=sec_ref.ports["P1"])
    c.add_port(name="S2", port=sec_ref.ports["P2"])

    if add_pgs:
        if not layers_pgs:
            raise ValueError(
                "add_pgs=True but layers_pgs is empty — the default "
                "layer_primary/layer_secondary/layer_secondary_jumper "
                "already consume this PDK's 3 real metals, leaving no "
                "obviously-safe spare layer for a shield (see docstring). "
                "Pass an explicit layers_pgs if you want one anyway."
            )
        for layer in layers_pgs:
            for strip in _pgs(pgs_diameter, pgs_width, pgs_spacing):
                c.add_polygon(strip, layer=layer)

    c.flatten()
    return c


if __name__ == "__main__":
    c = symmetric_transformer()
    c.show()

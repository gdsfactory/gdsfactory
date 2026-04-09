"""Reusable schematic function factories for common photonic port patterns."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from kfactory.schematic import DSchematic

# Port pattern: 2-port horizontal (straight, bend, taper, etc.)
# o1 ────── o2
_LEFT_RIGHT = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
]

# Port pattern: 2-port bend (90-degree turn)
# o1 ──┐
#      o2
_LEFT_BOTTOM = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "bottom", "type": "photonic"},
]

# Port pattern: 1x2 splitter
#        ── o2
# o1 ──┤
#        ── o3
_1X2 = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
]

# Port pattern: 2x2 coupler
# o2 ──┐  ┌── o3
#      ├──┤
# o1 ──┘  └── o4
_2X2 = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "left", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
    {"name": "o4", "side": "right", "type": "photonic"},
]

# Port pattern: ring coupler (bus below, ring above)
_COUPLER_RING = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "top", "type": "photonic"},
    {"name": "o3", "side": "top", "type": "photonic"},
    {"name": "o4", "side": "right", "type": "photonic"},
]

# Port pattern: 4-port crossing
_CROSSING = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "bottom", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
    {"name": "o4", "side": "top", "type": "photonic"},
]

# Port pattern: 1-port terminator
_TERMINATOR = [
    {"name": "o1", "side": "left", "type": "photonic"},
]

# Port pattern: grating coupler (waveguide left, fiber top)
_GRATING = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "top", "type": "photonic"},
]

# Port pattern: 2-port with electrical contacts (modulators, heaters)
_MODULATOR = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
]

# Port pattern: photodiode (optical in, electrical out)
_PHOTODIODE = [
    {"name": "o1", "side": "left", "type": "photonic"},
]


def _make_schematic(
    symbol: str,
    tags: list[str],
    ports: list[dict[str, str]],
) -> DSchematic:
    s = DSchematic()
    s.info["tags"] = tags  # type: ignore[assignment]
    s.info["symbol"] = symbol
    s.info["ports"] = ports  # type: ignore[assignment]

    side_to_xy = {
        "left": (-1, 0, 180),
        "right": (1, 0, 0),
        "top": (0, 1, 90),
        "bottom": (0, -1, 270),
    }

    # Pre-count ports per side for centering
    side_counts: dict[str, int] = {}
    for port in ports:
        side_counts[port["side"]] = side_counts.get(port["side"], 0) + 1

    seen_sides: dict[str, int] = {}
    spacing = 0.5
    for port in ports:
        side = port["side"]
        idx = seen_sides.get(side, 0)
        seen_sides[side] = idx + 1
        total = side_counts[side]
        bx, by, orientation = side_to_xy[side]

        # Center multiple ports on the same side
        offset = (idx - (total - 1) / 2) * spacing
        if side in ("left", "right"):
            x, y = float(bx), by + offset
        else:
            x, y = bx + offset, float(by)

        xs = "metal1_routing" if port["type"] == "electric" else "strip"
        s.create_port(
            name=port["name"],
            cross_section=xs,
            x=x,
            y=y,
            orientation=cast(Literal[0, 90, 180, 270], orientation),
        )

    return s


def schematic(
    symbol: str,
    tags: list[str],
    ports: list[dict[str, str]],
) -> Callable[..., DSchematic]:
    """Returns a schematic function for use with @gf.cell(schematic_function=...)."""

    def _schematic_fn(**kwargs: Any) -> DSchematic:
        return _make_schematic(symbol, tags, ports)

    return _schematic_fn


# Pre-built schematic functions for common patterns
straight_schematic = schematic("straight", ["waveguide"], _LEFT_RIGHT)
bend_schematic = schematic("bend", ["bend"], _LEFT_BOTTOM)
sbend_schematic = schematic("sbend", ["bend", "s"], _LEFT_RIGHT)
taper_schematic = schematic("taper", ["taper"], _LEFT_RIGHT)
transition_schematic = schematic("transition", ["taper", "transition"], _LEFT_RIGHT)
terminator_schematic = schematic("terminator", ["terminator"], _TERMINATOR)
crossing_schematic = schematic("crossing", ["crossing"], _CROSSING)
coupler_schematic = schematic("coupler", ["coupler"], _2X2)
coupler_ring_schematic = schematic("coupler-ring", ["coupler", "ring"], _COUPLER_RING)
mmi_1x2_schematic = schematic("mmi-1x2", ["mmi", "1x2"], _1X2)
mmi_2x2_schematic = schematic("mmi-2x2", ["mmi", "2x2"], _2X2)
mzi_1x2_schematic = schematic("mzi-1x2", ["mzi", "1x2"], _LEFT_RIGHT)
mzi_2x2_schematic = schematic("mzi-2x2", ["mzi", "2x2"], _LEFT_RIGHT)
ring_single_schematic = schematic("ring-single", ["ring", "single"], _LEFT_RIGHT)
ring_double_schematic = schematic("ring-double", ["ring", "double"], _2X2)
spiral_schematic = schematic("spiral", ["spiral"], _LEFT_RIGHT)
grating_coupler_schematic = schematic("grating-coupler", ["grating-coupler"], _GRATING)
photodiode_schematic = schematic("photodiode", ["photodiode"], _PHOTODIODE)
modulator_schematic = schematic("modulator", ["modulator"], _MODULATOR)
inductor_schematic = schematic(
    "inductor",
    ["inductor"],
    [
        {"name": "P1", "side": "left", "type": "electric"},
        {"name": "P2", "side": "right", "type": "electric"},
    ],
)
capacitor_schematic = schematic(
    "capacitor",
    ["capacitor"],
    [
        {"name": "o1", "side": "left", "type": "electric"},
        {"name": "o2", "side": "right", "type": "electric"},
    ],
)
wire_schematic = schematic(
    "straight",
    ["wire"],
    [
        {"name": "e1", "side": "left", "type": "electric"},
        {"name": "e2", "side": "right", "type": "electric"},
    ],
)
pad_schematic = schematic(
    "pad",
    ["pad"],
    [
        {"name": "e1", "side": "left", "type": "electric"},
        {"name": "e2", "side": "top", "type": "electric"},
        {"name": "e3", "side": "right", "type": "electric"},
        {"name": "e4", "side": "bottom", "type": "electric"},
    ],
)
ckt_schematic = schematic("ckt", [], _LEFT_RIGHT)

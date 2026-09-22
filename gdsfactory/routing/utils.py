from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from gdsfactory.typings import ComponentSpec, Port

if TYPE_CHECKING:
    from gdsfactory.component import Component
    from gdsfactory.cross_section import CrossSection


class RouteWarning(UserWarning):
    pass


class BendPortTypeError(ValueError):
    """Raised when a bend has no two ports of the port type being routed."""


def get_default_bend(port_type: str, cross_section: CrossSection) -> ComponentSpec:
    """Returns the bend for a Manhattan route whose caller did not choose one.

    Keyed on the port type being routed rather than on the cross-section's port types,
    because the port type is what the placer selects the bend's own ports by, so the
    default is one that can be placed by construction.

    An electrical route gets a square corner. `wire_corner` draws the main section
    only, so a multi-section cross-section (GS, GSG) gets `wire_corner_sections`,
    which draws them all.

    Args:
        port_type: Port type the route is placed on, normally `ports1[0].port_type`
            or an explicit `port_type=`.
        cross_section: Cross-section the route is drawn with.
    """
    if port_type != "electrical":
        return "bend_euler"
    return "wire_corner" if len(cross_section.sections) == 1 else "wire_corner_sections"


def validate_bend90(
    bend90: Component, port_type: str, default_bend: ComponentSpec
) -> None:
    """Raises if a bend cannot be placed on a route of `port_type`.

    Args:
        bend90: Bend cell, built the way the router will place it.
        port_type: Port type the route is placed on.
        default_bend: The route's default from `get_default_bend`, suggested instead
            unless `bend90` is that bend already.

    Raises:
        BendPortTypeError: If the bend does not have exactly two `port_type` ports.
    """
    matching = bend90.ports.filter(port_type=port_type)
    if len(matching) == 2:
        return

    port_types = [port.port_type for port in bend90.ports]
    hint = (
        ""
        if bend90.function_name == default_bend
        else f" Use bend={default_bend!r}, the default for this route."
    )
    raise BendPortTypeError(
        f"Bend {bend90.name!r} has {len(matching)} {port_type!r} ports, but needs "
        f"2 to route {port_type!r} ports. Its port types are {port_types}.{hint}"
    )


def direction_ports_from_list_ports(
    optical_ports: Sequence[Port],
) -> dict[str, list[Port]]:
    """Returns a dict of WENS ports."""
    direction_ports: dict[str, list[Port]] = {x: [] for x in ["E", "N", "W", "S"]}
    for p in optical_ports:
        orientation = (p.orientation + 360.0) % 360
        if orientation <= 45.0 or orientation >= 315:
            direction_ports["E"].append(p)
        elif orientation <= 135.0:
            direction_ports["N"].append(p)
        elif orientation <= 225.0:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)

    for direction, list_ports in list(direction_ports.items()):
        if direction in ["E", "W"]:
            list_ports.sort(key=lambda p: p.y)

        if direction in ["S", "N"]:
            list_ports.sort(key=lambda p: p.x)

    return direction_ports


def check_ports_have_equal_spacing(list_ports: Sequence[Port]) -> float:
    """Returns port separation.

    Raises error if not constant.

    """
    if not isinstance(list_ports, list):
        raise ValueError(f"list_ports should be a list of ports, got {list_ports}")
    if not list_ports:
        raise ValueError("list_ports should not be empty")

    orientation = get_list_ports_angle(list_ports)
    if orientation in [0, 180]:
        xys = [p.y for p in list_ports]
    else:
        xys = [p.x for p in list_ports]

    seps = [round(abs(c2 - c1), 5) for c1, c2 in zip(xys[1:], xys[:-1], strict=False)]
    different_seps = set(seps)
    if len(different_seps) > 1:
        raise ValueError(f"Ports should have the same separation. Got {different_seps}")
    return float(seps[0])


def get_list_ports_angle(list_ports: Sequence[Port]) -> float | None:
    """Returns the orientation/angle (in degrees) of a list of ports."""
    if not list_ports:
        return None
    if len({p.orientation for p in list_ports}) > 1:
        raise ValueError(f"All port angles should be the same. Got {list_ports}")
    return list_ports[0].orientation

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

from gdsfactory.typings import ComponentSpec, Port

if TYPE_CHECKING:
    from gdsfactory.component import Component


class RouteWarning(UserWarning):
    pass


def get_default_bend(port_type: str) -> ComponentSpec:
    """Returns the bend for a Manhattan route whose caller did not choose one.

    Keyed on the port type being routed rather than on the cross-section, because the
    port type is what the placer selects the bend's own ports by, so the default is one
    that can be placed by construction.

    `wire_corner` draws the main section only, so a multi-section electrical
    cross-section (GS, GSG) needs a bend that draws them all like `wire_corner_sections`.

    Args:
        port_type: Port type the route is placed on, normally `ports1[0].port_type`
            or an explicit `port_type=`.
    """
    return "wire_corner" if port_type == "electrical" else "bend_euler"


def validate_bend90(bend90: Component, port_type: str) -> None:
    """Raises if a bend cannot be placed on a route of `port_type`.

    Args:
        bend90: Bend cell, built the way the router will place it.
        port_type: Port type the route is placed on.

    Raises:
        ValueError: If the bend does not have exactly two `port_type` ports.
    """
    matching = bend90.ports.filter(port_type=port_type)
    if len(matching) == 2:
        return

    counts = Counter(port.port_type for port in bend90.ports)
    instead = next(
        (
            f"pass port_type={other!r} to route the ports it does have"
            for other, count in counts.items()
            if count == 2
        ),
        "pass a bend that has them",
    )
    has = ", ".join(f"{count} {name}" for name, count in counts.items()) or "none"
    raise ValueError(
        f"Cannot route {port_type!r} ports with bend {bend90.name!r}: it has "
        f"{len(matching)} {port_type!r} ports, and a 90 degree bend needs exactly 2, "
        f"one to enter the turn and one to leave it. Its ports are: {has}. Use "
        f"bend={get_default_bend(port_type)!r}, the default for {port_type!r} routes, "
        f"or {instead}."
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

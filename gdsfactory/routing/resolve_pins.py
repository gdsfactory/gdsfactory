"""Pin-to-Port resolution for electrical routing.

Resolves Pin pairs to Port pairs by selecting the ports that minimize
Euclidean distance between connection points. Cell designers control
which directions are routable by choosing which ports to include in a Pin.
"""

from __future__ import annotations

import math

from gdsfactory.typings import Pin, Port


def resolve_pin_pair(pin_a: Pin, pin_b: Pin) -> tuple[Port, Port]:
    """Resolve a single Pin pair to a Port pair by minimum Euclidean distance.

    For each combination of ports from pin_a and pin_b, computes the
    Euclidean distance between port centers and returns the pair with
    the smallest distance.

    Args:
        pin_a: first Pin.
        pin_b: second Pin.

    Returns:
        Tuple of (port_from_a, port_from_b) with minimum distance.

    Raises:
        ValueError: if either Pin has no ports.
    """
    ports_a = pin_a.ports
    ports_b = pin_b.ports

    if not ports_a:
        raise ValueError(
            f"Pin {pin_a.name!r} has no ports. "
            "Each Pin must contain at least one Port."
        )
    if not ports_b:
        raise ValueError(
            f"Pin {pin_b.name!r} has no ports. "
            "Each Pin must contain at least one Port."
        )

    best_dist = math.inf
    best_pair: tuple[Port, Port] | None = None

    for pa in ports_a:
        ax, ay = pa.center
        for pb in ports_b:
            bx, by = pb.center
            dist = math.hypot(ax - bx, ay - by)
            if dist < best_dist:
                best_dist = dist
                best_pair = (pa, pb)

    assert best_pair is not None
    return best_pair


def resolve_pins(
    pins1: list[Pin],
    pins2: list[Pin],
) -> tuple[list[Port], list[Port]]:
    """Resolve lists of Pin pairs to lists of Port pairs.

    Each Pin in pins1 is paired with the corresponding Pin in pins2
    by index. For each pair, selects the ports that minimize Euclidean
    distance.

    Args:
        pins1: list of source Pins.
        pins2: list of destination Pins. Must be same length as pins1.

    Returns:
        Tuple of (resolved_ports1, resolved_ports2).

    Raises:
        ValueError: if lists differ in length or any Pin has no ports.
    """
    if len(pins1) != len(pins2):
        raise ValueError(
            f"pins1 has {len(pins1)} pins and pins2 has {len(pins2)} pins. "
            "Lists must be equal length."
        )

    ports1: list[Port] = []
    ports2: list[Port] = []

    for pa, pb in zip(pins1, pins2, strict=True):
        resolved_a, resolved_b = resolve_pin_pair(pa, pb)
        ports1.append(resolved_a)
        ports2.append(resolved_b)

    return ports1, ports2

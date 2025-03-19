import warnings

import pytest

import gdsfactory as gf
from gdsfactory.routing.utils import RouteWarning
from gdsfactory.typings import CrossSectionSpec

MANHATTAN_ANGLES = [0, 90, 180, 270]


@gf.cell
def port_bank(
    count: int = 4, spacing: float = 3.0, cross_section: CrossSectionSpec = "strip"
) -> gf.Component:
    """Create a bank of ports.

    Args:
        count: The number of ports to create.
        spacing: The spacing between the ports.
        cross_section: The cross section of the ports.

    """
    c = gf.Component()
    xs = [spacing * i for i in range(count)]
    ys = [0 for _ in xs]
    for i, (x, y) in enumerate(zip(xs, ys)):
        c.add_port(
            f"o{i + 1}", center=(x, y), cross_section=cross_section, orientation=90
        )
        _ = c << gf.c.text(
            str(i + 1), position=(x, y - 2), size=1.5, layer=(2, 0), justify="center"
        )

    c.add_polygon([(xs[0], 0), (xs[-1], 0), ((xs[0] + xs[-1]) * 0.5, -2)], layer=(1, 0))
    return c


def connection_tuple(port1: gf.Port, port2: gf.Port) -> tuple[tuple[float, float], ...]:
    return (port1.center, port2.center)


def make_bundle(
    angle: float = 0, reverse_ports: bool = False, sort_ports: bool = True
) -> None:
    """Create a bundle of ports and route them together.

    Args:
        angle: The angle of the second port bank relative to the first.
        reverse_ports: Whether to reverse the order of the ports in the second bank.
        sort_ports: Whether to sort the ports in the second bank by their x-coordinate.

    """
    c = gf.Component()
    b = port_bank()
    r1 = c.add_ref(b, "r1")
    r2 = c.add_ref(b, "r2")
    r2.rotate(angle)
    r2.dmove((40, 60))
    ports1_names = ["o1", "o2", "o3", "o4"]
    ports2_names = ["o1", "o2", "o3", "o4"]

    if reverse_ports:
        ports2_names.reverse()
    ports1 = [r1.ports[n] for n in ports1_names]
    ports2 = [r2.ports[n] for n in ports2_names]
    gf.routing.route_bundle(
        c, ports1, ports2, sort_ports=sort_ports, separation=2.5, cross_section="strip"
    )


@pytest.mark.parametrize("angle", MANHATTAN_ANGLES)
def test_good_bundle_passes_sorted(angle: float) -> None:
    with warnings.catch_warnings(record=True) as ws:
        make_bundle(angle, reverse_ports=True, sort_ports=True)
        for w in ws:
            if issubclass(w.category, RouteWarning):
                raise AssertionError(f"Routing warning was raised: {w}")


if __name__ == "__main__":
    make_bundle(angle=0, reverse_ports=False, sort_ports=True)

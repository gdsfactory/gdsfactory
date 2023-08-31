import warnings

import pytest

import gdsfactory as gf
from gdsfactory.routing.manhattan import RouteWarning
from gdsfactory.typings import CrossSectionSpec

MANHATTAN_ANGLES = [0, 90, 180, 270]


@gf.cell
def port_bank(
    count: int = 4, spacing: float = 3.0, cross_section: CrossSectionSpec = "strip"
) -> gf.Component:
    c = gf.Component()
    xs = [spacing * i for i in range(count)]
    ys = [0 for _ in xs]
    for i, (x, y) in enumerate(zip(xs, ys)):
        c.add_port(
            f"o{i+1}", center=(x, y), cross_section=cross_section, orientation=90
        )
        _ = c << gf.c.text(
            str(i + 1), position=(x, y - 2), size=1.5, layer=(2, 0), justify="center"
        )

    c.add_polygon([[xs[0], 0], [xs[-1], 0], [(xs[0] + xs[-1]) * 0.5, -2]])
    return c


def connection_tuple(port1: gf.Port, port2: gf.Port) -> tuple:
    return (tuple(port1.center), tuple(port2.center))


def make_bundle(
    angle: float = 0, reverse_ports: bool = False, sort_ports: bool = False
) -> None:
    c = gf.Component()
    b = port_bank()
    r1 = c.add_ref(b, "r1")
    r2 = c.add_ref(b, "r2")
    r2.rotate(angle, center="o1")
    r2.move((40, 30))
    ports1_names = ["o1", "o2", "o3", "o4"]
    ports2_names = ["o1", "o2", "o3", "o4"]
    if reverse_ports:
        ports2_names.reverse()
    ports1 = [r1.ports[n] for n in ports1_names]
    ports2 = [r2.ports[n] for n in ports2_names]
    port1_lookup = {tuple(p.center): p.name for p in ports1}
    port2_lookup = {tuple(p.center): p.name for p in ports2}
    connections_expected = {connection_tuple(p1, p2) for p1, p2 in zip(ports1, ports2)}
    bundle = gf.routing.get_bundle(ports1, ports2, sort_ports=sort_ports)

    for route in bundle:
        c.add(route.references)
        connection = connection_tuple(*route.ports)
        if connection not in connections_expected:
            port1_pt, port2_pt = connection
            port1_name = port1_lookup.get(port1_pt, "N/A")
            port2_name = port2_lookup.get(port2_pt, "N/A")
            print(f"not a specified connection! r1.{port1_name} -> r2.{port2_name}")


@pytest.mark.parametrize("angle", MANHATTAN_ANGLES)
def test_bad_bundle_fails(angle: float) -> None:
    with pytest.warns(RouteWarning):
        make_bundle(angle, reverse_ports=False, sort_ports=False)


@pytest.mark.parametrize("angle", MANHATTAN_ANGLES)
def test_bad_bundle_fails_sorted(angle: float):
    with pytest.warns(RouteWarning):
        make_bundle(angle, reverse_ports=False, sort_ports=True)


@pytest.mark.parametrize("angle", MANHATTAN_ANGLES)
def test_good_bundle_passes(angle: float):
    if angle == 270:
        pytest.skip(
            "Skipping test for now... This is technically a routable bundle, but the router is not yet capable."
        )
    with warnings.catch_warnings(record=True) as ws:
        make_bundle(angle, reverse_ports=True, sort_ports=False)
        for w in ws:
            if issubclass(w.category, RouteWarning):
                raise AssertionError(f"Routing warning was raised: {w}")


@pytest.mark.parametrize("angle", MANHATTAN_ANGLES)
def test_good_bundle_passes_sorted(angle: float):
    if angle == 270:
        pytest.skip(
            "Skipping test for now... This is technically a routable bundle, but the router is not yet capable."
        )
    with warnings.catch_warnings(record=True) as ws:
        make_bundle(angle, reverse_ports=True, sort_ports=True)
        for w in ws:
            if issubclass(w.category, RouteWarning):
                raise AssertionError(f"Routing warning was raised: {w}")


if __name__ == "__main__":
    make_bundle(angle=180, reverse_ports=False, sort_ports=False)

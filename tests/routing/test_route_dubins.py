import pytest
from kfactory.conf import CheckInstances

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell(check_instances=CheckInstances.IGNORE)
def sample_route_dubins_basic() -> gf.Component:
    """Basic test showing Dubins path routing between two straight waveguides."""
    c = gf.Component()

    # Create two straight waveguides with different orientations
    wg1 = c << gf.components.straight(length=100, width=3.2)
    wg2 = c << gf.components.straight(length=100, width=3.2)

    # Move and rotate the second waveguide
    wg2.move((300, 50))
    wg2.rotate(45)

    # Route between the output of wg1 and input of wg2
    gf.routing.route_dubins(
        c,
        port1=wg1.ports["o2"],
        port2=wg2.ports["o1"],
        cross_section=gf.cross_section.strip(width=3.2),
        radius=100,
    )
    return c


@gf.cell(check_instances=CheckInstances.IGNORE)
def sample_route_dubins_array() -> gf.Component:
    """Test showing Dubins path routing between arrays of ports."""
    c = Component()

    # Create two multi-port components
    comp1 = c << gf.components.nxn(
        west=0, east=10, xsize=10, ysize=100, layer=(30, 0), wg_width=3.2
    )
    comp2 = c << gf.components.nxn(
        west=0, east=10, xsize=10, ysize=100, layer=(30, 0), wg_width=3.2
    )

    # Position second component
    comp2.rotate(30)
    comp2.move((500, -100))

    # Route between corresponding ports
    for i in range(10):
        port1_name = f"o{10 - i}"  # Inverted port id for port1
        port2_name = f"o{i + 1}"  # Adjusted to match available ports
        gf.routing.route_dubins(
            c,
            port1=comp1.ports[port1_name],
            port2=comp2.ports[port2_name],
            cross_section=comp1.ports[port1_name].cross_section,
            radius=100 + i * 10,
        )
    return c


def test_route_dubins_basic() -> None:
    sample_route_dubins_basic()


def test_route_dubins_array() -> None:
    sample_route_dubins_array()


@pytest.mark.parametrize("radius", [20.0, 50.0])
def test_radius_override_reaches_the_target(radius: float) -> None:
    xs = gf.cross_section.strip()
    c = gf.Component()
    start = c.add_port("start", center=(0, 0), orientation=0, cross_section=xs)
    end = c.add_port("end", center=(150, 80), orientation=210, cross_section=xs)
    route = gf.routing.route_dubins(c, start, end, xs, radius=radius)
    last = route.instances[-1].ports["o2"]
    assert last.center == pytest.approx(end.center, abs=0.002)
    assert last.orientation == pytest.approx((end.orientation + 180) % 360)
    for instance in route.instances:
        if "radius" in instance.cell.info:
            assert instance.cell.info["radius"] == radius
    assert route.length == pytest.approx(
        sum(instance.cell.info["length"] for instance in route.instances), abs=0.002
    )
    assert xs.radius == 10

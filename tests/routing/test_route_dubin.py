import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell(check_instances=False)
def sample_route_dubin_basic() -> gf.Component:
    """Basic test showing Dubins path routing between two straight waveguides."""
    c = gf.Component()

    # Create two straight waveguides with different orientations
    wg1 = c << gf.components.straight(length=100, width=3.2, layer=(30, 0))
    wg2 = c << gf.components.straight(length=100, width=3.2, layer=(30, 0))

    # Move and rotate the second waveguide
    wg2.move((300, 50))
    wg2.rotate(45)

    # Route between the output of wg1 and input of wg2
    gf.routing.route_dubin(
        c,
        port1=wg1.ports["o2"],
        port2=wg2.ports["o1"],
        cross_section=gf.cross_section.strip(width=3.2, layer=(30, 0), radius=100),
    )
    return c


@gf.cell(check_instances=False)
def sample_route_dubin_array() -> gf.Component:
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
    comp2.drotate(30)
    comp2.move((500, -100))

    # Route between corresponding ports
    for i in range(10):
        port1_name = f"o{10-i}"  # Inverted port id for port1
        port2_name = f"o{i+1}"  # Adjusted to match available ports
        gf.routing.route_dubin(
            c,
            port1=comp1.ports[port1_name],
            port2=comp2.ports[port2_name],
            cross_section=gf.cross_section.strip(
                width=3.2, layer=(30, 0), radius=100 + i * 10
            ),
        )
    return c


def test_route_dubin_basic() -> None:
    sample_route_dubin_basic()


def test_route_dubin_array() -> None:
    sample_route_dubin_array()


if __name__ == "__main__":
    # c = sample_route_dubin_basic()
    c = sample_route_dubin_array()
    c.show()

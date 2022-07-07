import gdsfactory as gf
from gdsfactory.add_loopback import add_loopback
from gdsfactory.component import Component
from gdsfactory.difftest import difftest


@gf.cell
def straight_with_loopback():
    c = gf.Component()
    wg = c << gf.components.straight()
    c.add(
        add_loopback(
            wg.ports["o1"],
            wg.ports["o2"],
            grating=gf.components.grating_coupler_te,
            inside=False,
        )
    )
    return c


def test_add_loopback() -> Component:
    c = straight_with_loopback()
    difftest(c)
    return c


if __name__ == "__main__":
    c = test_add_loopback()
    c.show(show_ports=True)

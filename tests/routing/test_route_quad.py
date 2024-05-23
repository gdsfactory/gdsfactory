import gdsfactory as gf


def test_route_quad() -> None:
    c = gf.Component()
    pad1 = c << gf.components.pad(size=(50, 50))
    pad2 = c << gf.components.pad(size=(10, 10))
    pad2.d.movex(100)
    pad2.d.movey(50)
    gf.routing.route_quad(
        c,
        pad1.ports["e2"],
        pad2.ports["e4"],
        width1=None,
        width2=None,
        manhattan_target_step=0.1,
    )

"""LVS demo."""

import gdsfactory as gf


@gf.cell
def pads_with_routes(radius: float = 10):
    """Returns MZI interferometer with bend."""
    c = gf.Component()
    pad = gf.components.pad()

    tl = c << pad
    bl = c << pad

    tr = c << pad
    br = c << pad

    tl.move((0, 300))
    br.move((500, 0))
    tr.move((500, 500))

    ports1 = [bl.ports["e3"], tl.ports["e3"]]
    ports2 = [br.ports["e1"], tr.ports["e1"]]
    routes = gf.routing.get_bundle(ports1, ports2, cross_section="metal3")

    for route in routes:
        c.add(route.references)

    return c


if __name__ == "__main__":
    c = pads_with_routes(radius=100)
    c.show(show_ports=True)

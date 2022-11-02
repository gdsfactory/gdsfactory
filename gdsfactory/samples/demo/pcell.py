"""Pcell demo."""

import gdsfactory as gf


@gf.cell
def mzi_with_bend(radius: float = 10):
    """Returns MZI interferometer with bend."""
    c = gf.Component()
    mzi = c.add_ref(gf.components.mzi())
    bend = c.add_ref(gf.components.bend_euler(radius=radius))
    bend.connect("o1", mzi.ports["o2"])
    c.add_port("o1", port=mzi.ports["o1"])
    c.add_port("o2", port=bend.ports["o2"])
    return c


if __name__ == "__main__":
    c = mzi_with_bend(radius=5)
    cc = gf.routing.add_fiber_single(c)
    cc.show(show_ports=True)

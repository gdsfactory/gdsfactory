"""PCell demo."""

from __future__ import annotations

import gdsfactory as gf


@gf.cell
def mzi_with_bend(radius: float = 10) -> gf.Component:
    """Returns MZI interferometer with bend."""
    c = gf.Component()
    mzi = c.add_ref(gf.components.mzi())
    bend = c.add_ref(gf.components.bend_euler(radius=radius))
    bend.connect("o1", mzi.ports["o2"])
    c.add_port("o1", port=mzi.ports["o1"])
    c.add_port("o2", port=bend.ports["o2"])
    return c


if __name__ == "__main__":
    c = mzi_with_bend(radius=50)
    # c = gf.routing.add_fiber_single(c)
    c.show()

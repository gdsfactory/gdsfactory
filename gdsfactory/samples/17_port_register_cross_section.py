"""Ports carry registered kfactory profiles, retrievable by their stored name."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component

gf.gpdk.PDK.activate()


@gf.cell
def component_with_registered_cross_section(
    length: float = 5.0,
    width: float = 2.5,
    layer: gf.typings.LayerSpec = "WG",
) -> Component:
    """Returns a component whose port cross section is registered in the layout.

    Args:
        length: in um.
        width: waveguide width in um.
        layer: layer.
    """
    xs = gf.cross_section.pin(width=width, layer=layer)

    p = gf.path.straight(length=length)
    c = p.extrude(xs)

    # No separate PDK factory registration is needed.
    xs_name = c["o1"].info["cross_section"]
    assert gf.get_cross_section(xs_name) == xs
    c = gf.c.extend_ports(c, cross_section=xs)
    return c


def test_component_with_registered_cross_section() -> None:
    assert component_with_registered_cross_section()


if __name__ == "__main__":
    c = component_with_registered_cross_section()
    c.show()

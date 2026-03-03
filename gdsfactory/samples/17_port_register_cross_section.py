"""Register a CrossSection factory when adding a port.

When adding a port with a cross_section, you can set
``register_cross_section_factory=True`` so the CrossSection is registered
in the active PDK.  This lets you later retrieve it by name with
``gf.get_cross_section``.
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.gpdk import PDK

PDK.activate()


@gf.cell
def component_with_registered_cross_section(
    length: float = 5.0,
    width: float = 2.5,
    layer: gf.typings.LayerSpec = "WG",
) -> Component:
    """Returns a component whose port cross_section is registered in the PDK.

    Args:
        length: in um.
        width: waveguide width in um.
        layer: layer.
    """
    xs = gf.cross_section.pin(width=width, layer=layer)

    c = gf.Component()
    p = gf.path.straight(length=length)
    c = p.extrude(
        xs, register_cross_section_factory=True
    )  # extrude the path to create a component with a polygon geometry

    # c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    # c.add_port(
    #     name="o1",
    #     center=(0, width / 2),
    #     orientation=180,
    #     cross_section=xs,
    #     register_cross_section_factory=True,
    # )

    # The cross_section is now retrievable from the PDK by name.
    xs_name = c["o1"].info["cross_section"]
    assert gf.get_cross_section(xs_name) == xs
    # c = gf.c.extend_ports(c, cross_section=c['o1'].info['cross_section'])
    c = gf.c.extend_ports(c, cross_section=xs)
    return c


def test_component_with_registered_cross_section() -> None:
    assert component_with_registered_cross_section()


if __name__ == "__main__":
    c = component_with_registered_cross_section()
    c.show()

"""based on phidl.geometry."""

from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.text import text
from gdsfactory.typings import Anchor, ComponentSpec, LayerSpec

big_square = partial(rectangle, size=(1300, 2600), centered=False)


@gf.cell
def die_bbox(
    component: ComponentSpec = big_square,
    street_width: float = 100.0,
    street_length: float | None = None,
    die_name: str | None = None,
    text_size: float = 100.0,
    text_anchor: Anchor = "sw",
    layer: LayerSpec = "MTOP",
    padding: float = 10.0,
) -> gf.Component:
    """Returns component with boundary box frame around it.

    Perfect for defining the boundary of the chip/die
    it can also add a label with the name of the die.
    similar to die and bbox.

    Args:
        component: to frame.
        street_width: Width of the boundary box.
        street_length: length of the boundary box.
        die_name: Label text.
        text_size: Label text size.
        text_anchor: {'nw', 'nc', 'ne', 'sw', 'sc', 'se'} text location.
        layer: Specific layer(s) to put polygon geometry on.
        padding: adds padding.
    """
    c = gf.Component()
    component = gf.get_component(component)

    c.copy_child_info(component)
    cref = c.add_ref(component)
    cref.dmovex(-(cref.dbbox().right + cref.dbbox().left) // 2)
    cref.dmovey(-(cref.dbbox().bottom + cref.dbbox().top) // 2)
    sx, sy = component.dxsize, component.dysize

    sx += street_width + padding
    sy += street_width + padding

    street_length = street_length or max([sx, sy])

    xpts = np.array(
        [
            sx,
            sx,
            sx - street_width,
            sx - street_width,
            sx - street_length,
            sx - street_length,
        ]
    )
    ypts = np.array(
        [
            sy,
            sy - street_length,
            sy - street_length,
            sy - street_width,
            sy - street_width,
            sy,
        ]
    )
    c.add_polygon(list(zip(+xpts, +ypts)), layer=layer)
    c.add_polygon(list(zip(-xpts, +ypts)), layer=layer)
    c.add_polygon(list(zip(+xpts, -ypts)), layer=layer)
    c.add_polygon(list(zip(-xpts, -ypts)), layer=layer)

    if die_name:
        t = c.add_ref(text(text=die_name, size=text_size, layer=layer))

        d = street_width + 20
        if text_anchor == "nw":
            t.dxmin, t.dymax = [-sx + d, sy - d]
        elif text_anchor == "nc":
            t.dx, t.dymax = [0, sy - d]
        elif text_anchor == "ne":
            t.dxmax, t.dymax = [sx - d, sy - d]
        if text_anchor == "sw":
            t.dxmin, t.dymin = [-sx + d, -sy + d]
        elif text_anchor == "sc":
            t.dx, t.dymin = [0, -sy + d]
        elif text_anchor == "se":
            t.dxmax, t.dymin = [sx - d, -sy + d]

    return c


if __name__ == "__main__":
    mask = gf.components.array(rows=10, columns=10)
    # c = die_bbox(component=mask, die_name="chip99")
    c = die_bbox(component=mask, die_name="chip99", text_anchor="se")
    c.show()

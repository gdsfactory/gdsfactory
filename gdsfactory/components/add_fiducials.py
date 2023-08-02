from __future__ import annotations

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.pad import pad_array
from gdsfactory.typings import ComponentSpec, Coordinates, Float2


@cell
def add_fiducials(
    component: ComponentSpec = pad_array,
    gap: float = 50,
    left: ComponentSpec | None = "cross",
    right: ComponentSpec | None = "cross",
    top: ComponentSpec | None = None,
    bottom: ComponentSpec | None = None,
    offset: Float2 = (0, 0),
    **kwargs,
) -> Component:
    """Return component with fiducials.

    Args:
        component: component to add to the new component.
        gap: from component to fiducial edge.
        left: optional left fiducial.
        right: optional right fiducial.
        top: optional top fiducial.
        bottom: optional bottom fiducial.
        offset: component offset coordinate (x, y).
        kwargs: fiducial settings.
    """
    c = Component()
    component = gf.get_component(component, **kwargs)
    r = c << component
    r.move(offset)

    if left:
        x1 = c << gf.get_component(left)
        x1.xmax = r.xmin - gap
        c.add_ports(x1.ports, prefix="l")

    if right:
        x2 = c << gf.get_component(right)
        x2.xmin = r.xmax + gap
        c.add_ports(x2.ports, prefix="r")

    if top:
        y1 = c << gf.get_component(top)
        y1.ymin = r.ymax + gap
        c.add_ports(y1.ports, prefix="t")

    if bottom:
        y2 = c << gf.get_component(bottom)
        y2.ymax = r.ymin - gap
        c.add_ports(y2.ports, prefix="b")

    c.add_ports(r.ports)
    c.copy_child_info(component)
    return c


@cell
def add_fiducials_offsets(
    component: ComponentSpec = pad_array,
    fiducial: ComponentSpec = "cross",
    offsets: Coordinates = ((0, 100), (0, -100)),
) -> Component:
    """Returns new component with fiducials from a list of offsets.

    Args:
        component: add reference to component to the new Component.
        fiducial: function to return fiducial.
        offsets: list of offsets.
    """
    c = Component()
    component = gf.get_component(component)
    fiducial = gf.get_component(fiducial)
    r = c << component
    c.add_ports(r.ports)
    c.copy_child_info(component)

    for offset in offsets:
        f = c << fiducial
        f.move(offset)

    return c


if __name__ == "__main__":
    from gdsfactory.generic_tech import get_generic_pdk

    PDK = get_generic_pdk()
    PDK.activate()
    # c = add_fiducials(top='cross')
    c = add_fiducials_offsets()
    c.show(show_ports=True)

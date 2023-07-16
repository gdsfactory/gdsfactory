from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.components.bbox import bbox
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler import coupler
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def add_trenches(
    component: ComponentSpec = coupler,
    cross_section: CrossSectionSpec = "rib_with_trenches",
    top: bool = True,
    bot: bool = True,
    right: bool = False,
    left: bool = False,
    **kwargs,
) -> gf.Component:
    """Return component with trenches.

    Args:
        component: component to add to the trenches.
        cross_section: spec (CrossSection, string or dict).
        top: add top trenches.
        bot: add bot trenches.
        right: add right trenches.
        left: add left trenches.
        kwargs: component settings.
    """
    c = gf.Component()
    component = gf.get_component(component, **kwargs)
    xs = gf.get_cross_section(cross_section)

    layer_trench = xs.info["settings"]["layer_trench"]
    width_trench = xs.info["settings"]["width_trench"]

    top = width_trench if top else 0
    bot = width_trench if bot else 0
    left = width_trench if left else 0
    right = width_trench if right else 0

    core = component
    clad = bbox(
        core.bbox, layer=layer_trench, top=top, bottom=bot, left=left, right=right
    )
    ref = c << gf.geometry.boolean(clad, core, operation="not", layer=layer_trench)

    if xs.add_bbox:
        c = xs.add_bbox(c) or c
    if xs.add_pins:
        c = xs.add_pins(c) or c

    c.add_ports(component.ports, cross_section=xs)
    c.copy_child_info(component)
    c.absorb(ref)

    return c


add_trenches90 = partial(
    add_trenches, component=bend_euler, top=False, bot=True, right=True, left=False
)

if __name__ == "__main__":
    from gdsfactory.generic_tech import get_generic_pdk

    PDK = get_generic_pdk()
    PDK.activate()
    c = add_trenches()
    c.show(show_ports=True)

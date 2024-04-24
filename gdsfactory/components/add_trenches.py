from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.components.bbox import bbox
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler import coupler
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec


@gf.cell
def add_trenches(
    component: ComponentSpec = coupler,
    layer_component: LayerSpec = "WG",
    layer_trench: LayerSpec = "DEEP_ETCH",
    width_trench: float = 2.0,
    cross_section: CrossSectionSpec = "xs_rc_with_trenches",
    top: bool = True,
    bot: bool = True,
    right: bool = False,
    left: bool = False,
    **kwargs,
) -> gf.Component:
    """Return component with trenches.

    Args:
        component: component to add to the trenches.
        layer_component: layer of the component.
        layer_trench: layer of the trenches.
        width_trench: width of the trenches.
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

    top = width_trench if top else 0
    bot = width_trench if bot else 0
    left = width_trench if left else 0
    right = width_trench if right else 0

    core = component
    clad = bbox(core, layer=layer_trench, top=top, bottom=bot, left=left, right=right)
    trenches = gf.boolean(
        clad,
        core,
        operation="not",
        layer=layer_trench,
        layer1=layer_trench,
        layer2=layer_component,
    )

    _ = c << trenches
    c.add_ports(component.ports)
    c.copy_child_info(component)
    xs.add_bbox(c)
    return c


add_trenches90 = partial(
    add_trenches, component=bend_euler, top=False, bot=True, right=True, left=False
)

if __name__ == "__main__":
    c = add_trenches()
    c.show()

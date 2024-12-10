from __future__ import annotations

from functools import partial
from typing import Any

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec


@gf.cell
def add_trenches(
    component: ComponentSpec = "coupler",
    layer_component: LayerSpec = "WG",
    layer_trench: LayerSpec = "DEEP_ETCH",
    width_trench: float = 2.0,
    cross_section: CrossSectionSpec = "rib_with_trenches",
    top: float | None = None,
    bot: float | None = None,
    right: float | None = 0,
    left: float | None = 0,
    **kwargs: Any,
) -> gf.Component:
    """Return component with trenches.

    Args:
        component: component to add to the trenches.
        layer_component: layer of the component.
        layer_trench: layer of the trenches.
        width_trench: width of the trenches.
        cross_section: spec (CrossSection, string or dict).
        top: width of the trench on the top. If None uses width_trench.
        bot: width of the trench on the bottom. If None uses width_trench.
        right: width of the trench on the right. If None uses width_trench.
        left: width of the trench on the left. If None uses width_trench.
        kwargs: component settings.
    """
    component = gf.get_component(component, **kwargs)
    xs = gf.get_cross_section(cross_section)

    top = top if top is not None else width_trench
    bot = bot if bot is not None else width_trench
    right = right if right is not None else width_trench
    left = left if left is not None else width_trench

    core = component
    clad = gf.c.bbox(
        core, layer=layer_trench, top=top, bottom=bot, left=left, right=right
    )
    c = gf.boolean(
        clad,
        core,
        operation="not",
        layer=layer_trench,
        layer1=layer_trench,
        layer2=layer_component,
    )

    c.add_ports(component.ports)
    c.copy_child_info(component)
    xs.add_bbox(c)
    return c


add_trenches90 = partial(
    add_trenches, component="bend_euler", top=0, left=0, right=None
)

if __name__ == "__main__":
    c = add_trenches90()
    c.show()

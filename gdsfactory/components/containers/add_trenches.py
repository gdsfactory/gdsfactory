from __future__ import annotations

__all__ = ["add_trenches", "add_trenches90"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec


@gf.cell_with_module_name(tags=["containers"])
def add_trenches(
    component: ComponentSpec = "coupler",
    layer_component: LayerSpec = "WG",
    layer_trench: LayerSpec = "DEEP_ETCH",
    width_trench: float = 2.0,
    cross_section: CrossSectionSpec | None = None,
    top: float | None = None,
    bot: float | None = None,
    right: float | None = 0,
    left: float | None = 0,
) -> gf.Component:
    """Return inverted component with trenches.

    Args:
        component: component to add to the trenches.
        layer_component: layer of the component to invert.
        layer_trench: layer of the trenches.
        width_trench: width of the trenches.
        cross_section: spec (CrossSection, string or dict).
        top: width of the trench on the top. If None uses width_trench.
        bot: width of the trench on the bottom. If None uses width_trench.
        right: width of the trench on the right. If None uses width_trench.
        left: width of the trench on the left. If None uses width_trench.
    """
    return cf.add_trenches(
        component=component,
        layer_component=layer_component,
        layer_trench=layer_trench,
        width_trench=width_trench,
        cross_section=cross_section,
        top=top,
        bot=bot,
        right=right,
        left=left,
    )


add_trenches90 = CellAlias(
    add_trenches, component="bend_euler", top=0, left=0, right=None
)

if __name__ == "__main__":
    c = add_trenches(bot=300)
    c.show()

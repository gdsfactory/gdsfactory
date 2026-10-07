"""Via chain."""

from __future__ import annotations

__all__ = ["via_chain"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpecs


@gf.cell_with_module_name(tags=["vias"])
def via_chain(
    num_vias: int = 100,
    cols: int = 10,
    via: ComponentSpec = "via1",
    contact: ComponentSpec = "via_stack_m2_m3",
    layers_bot: LayerSpecs = ("M1",),
    layers_top: LayerSpecs = ("M2",),
    offsets_top: tuple[float, ...] = (0,),
    offsets_bot: tuple[float, ...] = (0,),
    via_min_enclosure: float = 1.0,
    min_metal_spacing: float = 1.0,
    contact_offset: float = 0.0,
) -> Component:
    """Via chain to extract via resistance.

    Args:
        num_vias: number of vias.
        cols: number of column pairs.
        via: via component.
        contact: contact component.
        layers_bot: list of bottom layers.
        layers_top: list of top layers.
        offsets_top: list of top layer offsets.
        offsets_bot: list of bottom layer offsets.
        via_min_enclosure: via_min_enclosure.
        min_metal_spacing: min_metal_spacing.
        contact_offset: contact offset.

    ```text
        side view:
                                              min_metal_spacing
           ┌────────────────────────────────────┐              ┌────────────────────────────────────┐
           │  layers_top                        │              │                                    │
           │                                    │◄───────────► │                                    │
           └─────────────┬─────┬────────────────┘              └───────────────┬─────┬──────────────┘
                         │     │         via_enclosure                         │     │
                         │     │◄───────────────►                              │     │
                         │     │                                               │     │
                         │     │                                               │     │
                         │width│                                               │     │
                         ◄─────►                                               │     │
                         │     │                                               │     │
           ┌─────────────┴─────┴───────────────────────────────────────────────┴─────┴───────────────┐
           │ layers_bot                                                                              │
           │                                                                                         │
           └─────────────────────────────────────────────────────────────────────────────────────────┘
    ```

    ```text
           ◄─────────────────────────────────────────────────────────────────────────────────────────►
                                         2*e + w + min_metal_spacing + 2*e + w
    ```

    """
    return cf.via_chain(
        num_vias=num_vias,
        cols=cols,
        via=via,
        contact=contact,
        layers_bot=layers_bot,
        layers_top=layers_top,
        offsets_top=offsets_top,
        offsets_bot=offsets_bot,
        via_min_enclosure=via_min_enclosure,
        min_metal_spacing=min_metal_spacing,
        contact_offset=contact_offset,
    )

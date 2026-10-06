from __future__ import annotations

__all__ = ["transmon", "transmon_circular"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["quantum"])
def transmon(
    pad_width: float = 200.0,
    pad_height: float = 100.0,
    pad_gap: float = 6.0,
    junction_width: float = 0.15,
    junction_height: float = 0.3,
    island_width: float = 10.0,
    island_height: float = 4.0,
    layer_metal: LayerSpec = (1, 0),
    layer_junction: LayerSpec = (2, 0),
    layer_island: LayerSpec = (1, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates a transmon qubit with Josephson junction.

    A transmon qubit consists of two capacitor pads connected by a Josephson junction.
    The junction creates an anharmonic oscillator that can be used as a qubit.

    Args:
        pad_width: Width of each capacitor pad in μm.
        pad_height: Height of each capacitor pad in μm.
        pad_gap: Gap between the two pads in μm.
        junction_width: Width of the Josephson junction in μm.
        junction_height: Height of the Josephson junction in μm.
        island_width: Width of the central island in μm.
        island_height: Height of the central island in μm.
        layer_metal: Layer for the metal pads.
        layer_junction: Layer for the Josephson junction.
        layer_island: Layer for the central island.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the transmon geometry.
    """
    return cf.transmon(
        pad_width=pad_width,
        pad_height=pad_height,
        pad_gap=pad_gap,
        junction_width=junction_width,
        junction_height=junction_height,
        island_width=island_width,
        island_height=island_height,
        layer_metal=layer_metal,
        layer_junction=layer_junction,
        layer_island=layer_island,
        port_type=port_type,
    )


@gf.cell_with_module_name(tags=["quantum"])
def transmon_circular(
    pad_radius: float = 100.0,
    pad_gap: float = 6.0,
    junction_width: float = 0.15,
    junction_height: float = 0.3,
    island_radius: float = 5.0,
    layer_metal: LayerSpec = (1, 0),
    layer_junction: LayerSpec = (2, 0),
    layer_island: LayerSpec = (1, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates a circular transmon qubit with Josephson junction.

    A circular variant of the transmon qubit with circular capacitor pads.

    Args:
        pad_radius: Radius of each circular capacitor pad in μm.
        pad_gap: Gap between the two pads in μm.
        junction_width: Width of the Josephson junction in μm.
        junction_height: Height of the Josephson junction in μm.
        island_radius: Radius of the central circular island in μm.
        layer_metal: Layer for the metal pads.
        layer_junction: Layer for the Josephson junction.
        layer_island: Layer for the central island.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the circular transmon geometry.
    """
    return cf.transmon_circular(
        pad_radius=pad_radius,
        pad_gap=pad_gap,
        junction_width=junction_width,
        junction_height=junction_height,
        island_radius=island_radius,
        layer_metal=layer_metal,
        layer_junction=layer_junction,
        layer_island=layer_island,
        port_type=port_type,
    )

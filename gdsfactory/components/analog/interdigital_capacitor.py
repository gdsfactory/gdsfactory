from __future__ import annotations

__all__ = ["interdigital_capacitor"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec

from .._schematic import capacitor_schematic


@gf.cell_with_module_name(schematic_function=capacitor_schematic, tags=["analog"])
def interdigital_capacitor(
    fingers: int = 4,
    finger_length: float | int = 20.0,
    finger_gap: float | int = 2.0,
    thickness: float | int = 5.0,
    layer: LayerSpec = "M1",
) -> Component:
    """Generate an interdigital capacitor component with ports on both ends.

    An interdigital capacitor consists of interleaved metal fingers that create
    a distributed capacitance. This component creates a planar capacitor with
    two sets of interleaved fingers extending from opposite ends.

    See for example Zhu et al., `Accurate circuit model of interdigital
    capacitor and its application to design of new quasi-lumped miniaturized
    filters with suppression of harmonic resonance`, doi: 10.1109/22.826833.

    Note:
        ``finger_length=0`` effectively provides a parallel plate capacitor.
        The capacitance scales approximately linearly with the number of fingers
        and finger length.

    Args:
        fingers: Total number of fingers of the capacitor (must be >= 1).
        finger_length: Length of each finger in μm.
        finger_gap: Gap between adjacent fingers in μm.
        thickness: Thickness of fingers and the base section in μm.
        layer: Layer specification for the capacitor geometry.

    Returns:
        Component: A gdsfactory component with the interdigital capacitor geometry
        and two electrical ports ('e1' and 'e2') on opposing sides.
    """
    return cf.interdigital_capacitor(
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=thickness,
        layer=layer,
    )

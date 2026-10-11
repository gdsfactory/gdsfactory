from __future__ import annotations

__all__ = ["flux_qubit", "flux_qubit_asymmetric"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["quantum"])
def flux_qubit(
    loop_width: float = 50.0,
    loop_height: float = 50.0,
    junction_width: float = 0.15,
    junction_height: float = 0.3,
    alpha_junction_width: float = 0.12,
    alpha_junction_height: float = 0.25,
    wire_width: float = 2.0,
    layer_metal: LayerSpec = (1, 0),
    layer_junction: LayerSpec = (2, 0),
    layer_alpha_junction: LayerSpec = (3, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates a flux qubit (persistent current qubit).

    A flux qubit consists of a superconducting loop interrupted by three Josephson junctions.
    Two junctions are identical (beta junctions) while the third is smaller (alpha junction)
    with roughly 0.5-0.8 times the critical current.

    Args:
        loop_width: Width of the superconducting loop in μm.
        loop_height: Height of the superconducting loop in μm.
        junction_width: Width of the beta Josephson junctions in μm.
        junction_height: Height of the beta Josephson junctions in μm.
        alpha_junction_width: Width of the alpha Josephson junction in μm.
        alpha_junction_height: Height of the alpha Josephson junction in μm.
        wire_width: Width of the superconducting wires in μm.
        layer_metal: Layer for the metal wires.
        layer_junction: Layer for the beta Josephson junctions.
        layer_alpha_junction: Layer for the alpha Josephson junction.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the flux qubit geometry.
    """
    return cf.flux_qubit(
        loop_width=loop_width,
        loop_height=loop_height,
        junction_width=junction_width,
        junction_height=junction_height,
        alpha_junction_width=alpha_junction_width,
        alpha_junction_height=alpha_junction_height,
        wire_width=wire_width,
        layer_metal=layer_metal,
        layer_junction=layer_junction,
        layer_alpha_junction=layer_alpha_junction,
        port_type=port_type,
    )


@gf.cell_with_module_name(tags=["quantum"])
def flux_qubit_asymmetric(
    loop_width: float = 60.0,
    loop_height: float = 40.0,
    junction_width: float = 0.15,
    junction_height: float = 0.3,
    alpha_junction_width: float = 0.12,
    alpha_junction_height: float = 0.25,
    wire_width: float = 2.0,
    asymmetry_angle: float = 15.0,
    layer_metal: LayerSpec = (1, 0),
    layer_junction: LayerSpec = (2, 0),
    layer_alpha_junction: LayerSpec = (3, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates an asymmetric flux qubit for reduced flux noise sensitivity.

    An asymmetric flux qubit has a loop geometry that is not perfectly symmetric,
    which can help reduce sensitivity to flux noise while maintaining controllability.

    Args:
        loop_width: Width of the superconducting loop in μm.
        loop_height: Height of the superconducting loop in μm.
        junction_width: Width of the beta Josephson junctions in μm.
        junction_height: Height of the beta Josephson junctions in μm.
        alpha_junction_width: Width of the alpha Josephson junction in μm.
        alpha_junction_height: Height of the alpha Josephson junction in μm.
        wire_width: Width of the superconducting wires in μm.
        asymmetry_angle: Angle of asymmetry in degrees.
        layer_metal: Layer for the metal wires.
        layer_junction: Layer for the beta Josephson junctions.
        layer_alpha_junction: Layer for the alpha Josephson junction.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the asymmetric flux qubit geometry.
    """
    return cf.flux_qubit_asymmetric(
        loop_width=loop_width,
        loop_height=loop_height,
        junction_width=junction_width,
        junction_height=junction_height,
        alpha_junction_width=alpha_junction_width,
        alpha_junction_height=alpha_junction_height,
        wire_width=wire_width,
        asymmetry_angle=asymmetry_angle,
        layer_metal=layer_metal,
        layer_junction=layer_junction,
        layer_alpha_junction=layer_alpha_junction,
        port_type=port_type,
    )

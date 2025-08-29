from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
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
    c = Component()

    # Create the superconducting loop as a hollow rectangle
    outer_rect = gf.components.rectangle(
        size=(loop_width, loop_height),
        layer=layer_metal,
    )

    inner_rect = gf.components.rectangle(
        size=(loop_width - 2 * wire_width, loop_height - 2 * wire_width),
        layer=layer_metal,
    )

    # Create the loop by boolean difference
    loop = gf.boolean(
        outer_rect,
        inner_rect,
        operation="not",
        layer=layer_metal,
    )

    loop_ref = c.add_ref(loop)
    loop_ref.move((-loop_width / 2, -loop_height / 2))

    # Create gaps for junctions
    # Bottom gap for alpha junction
    alpha_gap = gf.components.rectangle(
        size=(alpha_junction_width + 0.1, wire_width + 0.1),
        layer=layer_metal,
    )
    alpha_gap_ref = c.add_ref(alpha_gap)
    alpha_gap_ref.move((-alpha_junction_width / 2 - 0.05, -loop_height / 2 - 0.05))

    # Left gap for beta junction
    beta_gap_left = gf.components.rectangle(
        size=(wire_width + 0.1, junction_height + 0.1),
        layer=layer_metal,
    )
    beta_gap_left_ref = c.add_ref(beta_gap_left)
    beta_gap_left_ref.move((-loop_width / 2 - 0.05, -junction_height / 2 - 0.05))

    # Right gap for beta junction
    beta_gap_right = gf.components.rectangle(
        size=(wire_width + 0.1, junction_height + 0.1),
        layer=layer_metal,
    )
    beta_gap_right_ref = c.add_ref(beta_gap_right)
    beta_gap_right_ref.move(
        (loop_width / 2 - wire_width - 0.05, -junction_height / 2 - 0.05)
    )

    # Remove gaps from the loop
    # Need to perform boolean operations sequentially
    loop_with_gaps = gf.boolean(
        loop_ref,
        alpha_gap_ref,
        operation="not",
        layer=layer_metal,
    )
    loop_with_gaps = gf.boolean(
        loop_with_gaps,
        beta_gap_left_ref,
        operation="not",
        layer=layer_metal,
    )
    loop_with_gaps = gf.boolean(
        loop_with_gaps,
        beta_gap_right_ref,
        operation="not",
        layer=layer_metal,
    )
    c.add_ref(loop_with_gaps)

    # Create the alpha junction (smaller)
    alpha_junction = gf.components.rectangle(
        size=(alpha_junction_width, alpha_junction_height),
        layer=layer_alpha_junction,
    )
    alpha_junction_ref = c.add_ref(alpha_junction)
    alpha_junction_ref.move(
        (
            -alpha_junction_width / 2,
            -loop_height / 2 + wire_width / 2 - alpha_junction_height / 2,
        )
    )

    # Create the beta junctions (larger, identical)
    beta_junction_left = gf.components.rectangle(
        size=(junction_width, junction_height),
        layer=layer_junction,
    )
    beta_junction_left_ref = c.add_ref(beta_junction_left)
    beta_junction_left_ref.move(
        (-loop_width / 2 + wire_width / 2 - junction_width / 2, -junction_height / 2)
    )

    beta_junction_right = gf.components.rectangle(
        size=(junction_width, junction_height),
        layer=layer_junction,
    )
    beta_junction_right_ref = c.add_ref(beta_junction_right)
    beta_junction_right_ref.move(
        (loop_width / 2 - wire_width / 2 - junction_width / 2, -junction_height / 2)
    )

    # Add control lines for flux bias
    control_line_left = gf.components.rectangle(
        size=(20.0, 2.0),
        layer=layer_metal,
    )
    control_line_left_ref = c.add_ref(control_line_left)
    control_line_left_ref.move((-loop_width / 2 - 30.0, -1.0))

    control_line_right = gf.components.rectangle(
        size=(20.0, 2.0),
        layer=layer_metal,
    )
    control_line_right_ref = c.add_ref(control_line_right)
    control_line_right_ref.move((loop_width / 2 + 10.0, -1.0))
    c.flatten()

    # Add ports for flux control
    c.add_port(
        name="flux_control_left",
        center=(-loop_width / 2 - 40.0, 0),
        width=2.0,
        orientation=180,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="flux_control_right",
        center=(loop_width / 2 + 40.0, 0),
        width=2.0,
        orientation=0,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add readout connection
    c.add_port(
        name="readout",
        center=(0, loop_height / 2 + 10.0),
        width=wire_width,
        orientation=90,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add metadata
    c.info["qubit_type"] = "flux_qubit"
    c.info["loop_width"] = loop_width
    c.info["loop_height"] = loop_height
    c.info["beta_junction_area"] = junction_width * junction_height
    c.info["alpha_junction_area"] = alpha_junction_width * alpha_junction_height
    c.info["alpha_beta_ratio"] = (alpha_junction_width * alpha_junction_height) / (
        junction_width * junction_height
    )
    return c


@gf.cell_with_module_name
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
    c = Component()

    # Create the main loop structure
    points = []
    angle_rad = np.radians(asymmetry_angle)

    # Create asymmetric loop shape
    # Bottom side
    points.extend(
        [
            (-loop_width / 2, -loop_height / 2),
            (loop_width / 2, -loop_height / 2),
            (loop_width / 2, -loop_height / 2 + wire_width),
            (-loop_width / 2, -loop_height / 2 + wire_width),
        ]
    )

    # Right side with angle
    x_offset = loop_height / 2 * np.tan(angle_rad)
    points.extend(
        [
            (loop_width / 2 - wire_width, -loop_height / 2 + wire_width),
            (loop_width / 2 - wire_width, loop_height / 2 - wire_width),
            (loop_width / 2 - wire_width + x_offset, loop_height / 2 - wire_width),
            (loop_width / 2 + x_offset, loop_height / 2 - wire_width),
            (loop_width / 2 + x_offset, loop_height / 2),
            (loop_width / 2, loop_height / 2),
            (loop_width / 2, -loop_height / 2),
        ]
    )

    # Top side
    points.extend(
        [
            (loop_width / 2 + x_offset, loop_height / 2),
            (-loop_width / 2, loop_height / 2),
            (-loop_width / 2, loop_height / 2 - wire_width),
            (loop_width / 2 + x_offset, loop_height / 2 - wire_width),
        ]
    )

    # Left side
    points.extend(
        [
            (-loop_width / 2 + wire_width, loop_height / 2 - wire_width),
            (-loop_width / 2 + wire_width, -loop_height / 2 + wire_width),
            (-loop_width / 2, -loop_height / 2 + wire_width),
            (-loop_width / 2, loop_height / 2),
        ]
    )

    # Create the loop polygon
    c.add_polygon(points, layer=layer_metal)

    # Create junctions in the gaps
    # alpha junction at bottom
    alpha_junction = gf.components.rectangle(
        size=(alpha_junction_width, alpha_junction_height),
        layer=layer_alpha_junction,
    )
    alpha_junction_ref = c.add_ref(alpha_junction)
    alpha_junction_ref.move(
        (
            -alpha_junction_width / 2,
            -loop_height / 2 + wire_width / 2 - alpha_junction_height / 2,
        )
    )

    # beta junctions at sides
    beta_junction_left = gf.components.rectangle(
        size=(junction_width, junction_height),
        layer=layer_junction,
    )
    beta_junction_left_ref = c.add_ref(beta_junction_left)
    beta_junction_left_ref.move(
        (-loop_width / 2 + wire_width / 2 - junction_width / 2, 0)
    )

    beta_junction_right = gf.components.rectangle(
        size=(junction_width, junction_height),
        layer=layer_junction,
    )
    beta_junction_right_ref = c.add_ref(beta_junction_right)
    beta_junction_right_ref.move(
        (loop_width / 2 - wire_width / 2 - junction_width / 2, 0)
    )

    # Add control and readout ports
    c.add_port(
        name="flux_control",
        center=(0, loop_height / 2 + 10.0),
        width=wire_width,
        orientation=90,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="readout",
        center=(loop_width / 2 + x_offset + 10.0, 0),
        width=wire_width,
        orientation=0,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add metadata
    c.info["qubit_type"] = "flux_qubit_asymmetric"
    c.info["loop_width"] = loop_width
    c.info["loop_height"] = loop_height
    c.info["asymmetry_angle"] = asymmetry_angle
    c.info["beta_junction_area"] = junction_width * junction_height
    c.info["alpha_junction_area"] = alpha_junction_width * alpha_junction_height
    c.info["alpha_beta_ratio"] = (alpha_junction_width * alpha_junction_height) / (
        junction_width * junction_height
    )

    return c

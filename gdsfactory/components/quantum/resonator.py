from __future__ import annotations

__all__ = ["resonator_cpw", "resonator_lumped", "resonator_quarter_wave"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def resonator_cpw(
    length: float = 1000.0,
    width: float = 10.0,
    gap: float = 6.0,
    meander_pitch: float = 50.0,
    meander_width: float = 200.0,
    coupling_gap: float = 5.0,
    coupling_length: float = 100.0,
    layer_metal: LayerSpec = (1, 0),
    layer_gap: LayerSpec = (2, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates a coplanar waveguide (CPW) resonator.

    A CPW resonator consists of a meandered coplanar waveguide with coupling gaps
    for capacitive coupling to feedlines or qubits.

    Args:
        length: Total length of the resonator in μm.
        width: Width of the center conductor in μm.
        gap: Gap width on each side of the center conductor in μm.
        meander_pitch: Pitch between meander segments in μm.
        meander_width: Width of each meander section in μm.
        coupling_gap: Gap for capacitive coupling in μm.
        coupling_length: Length of the coupling region in μm.
        layer_metal: Layer for the metal conductor.
        layer_gap: Layer for the gaps (ground plane).
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the CPW resonator geometry.
    """
    c = Component()

    # Calculate number of meander sections needed
    num_meanders = int(length / meander_width) if meander_width > 0 else 1
    actual_length = num_meanders * meander_width

    # Create meander path
    path_points = []
    x, y = 0.0, 0.0

    for i in range(num_meanders):
        if i == 0:
            # First segment
            path_points.extend([(x, y), (x + meander_width, y)])
            x += meander_width
        else:
            # Alternate up and down
            if i % 2 == 1:
                y += meander_pitch
                path_points.extend([(x, y), (x - meander_width, y)])
                x -= meander_width
            else:
                y += meander_pitch
                path_points.extend([(x, y), (x + meander_width, y)])
                x += meander_width

    # Create the CPW path
    path = gf.Path(np.array(path_points))

    # Create cross section for CPW
    cpw_xs = gf.cross_section.strip(width=width, layer=layer_metal)

    # Create the resonator structure
    resonator_path = gf.path.extrude(path, cpw_xs)
    c.add_ref(resonator_path)

    # Create ground plane with gaps
    total_width = meander_width + 2 * meander_pitch
    total_height = (num_meanders - 1) * meander_pitch + width + 2 * gap

    # Ground plane
    ground_plane = gf.components.rectangle(
        size=(total_width + 2 * gap, total_height + 2 * gap),
        layer=layer_metal,
    )
    ground_ref = c.add_ref(ground_plane)
    ground_ref.move((-gap, -gap))

    # Create gaps by boolean subtraction
    gap_width = width + 2 * gap

    # Create gap path along the resonator
    gap_path = gf.Path(np.array(path_points))
    gap_xs = gf.cross_section.strip(width=gap_width, layer=layer_gap)
    gap_structure = gf.path.extrude(gap_path, gap_xs)

    # Subtract gaps from ground plane
    gf.boolean(
        ground_ref,
        gap_structure,
        operation="not",
        layer=layer_metal,
    )

    # Add coupling regions
    # Input coupling
    coupling_in = gf.components.rectangle(
        size=(coupling_length, coupling_gap),
        layer=layer_gap,
    )
    coupling_in_ref = c.add_ref(coupling_in)
    coupling_in_ref.move((-coupling_length / 2, -width / 2 - gap - coupling_gap / 2))

    # Output coupling
    coupling_out = gf.components.rectangle(
        size=(coupling_length, coupling_gap),
        layer=layer_gap,
    )
    coupling_out_ref = c.add_ref(coupling_out)
    coupling_out_ref.move(
        (x - coupling_length / 2, y + width / 2 + gap + coupling_gap / 2)
    )

    # Add ports for coupling
    c.add_port(
        name="input",
        center=(0, -width / 2 - gap - coupling_gap),
        width=coupling_length,
        orientation=270,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="output",
        center=(x, y + width / 2 + gap + coupling_gap),
        width=coupling_length,
        orientation=90,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add metadata
    c.info["resonator_type"] = "cpw"
    c.info["length"] = actual_length
    c.info["width"] = width
    c.info["gap"] = gap
    c.info["frequency_estimate"] = (
        3e8 / (2 * actual_length * 1e-6) / 1e9
    )  # GHz, rough estimate
    c.flatten()
    return c


@gf.cell_with_module_name
def resonator_lumped(
    capacitor_fingers: int = 4,
    capacitor_finger_length: float = 20.0,
    capacitor_finger_gap: float = 2.0,
    capacitor_thickness: float = 5.0,
    inductor_width: float = 2.0,
    inductor_turns: int = 3,
    inductor_radius: float = 20.0,
    coupling_gap: float = 5.0,
    layer_metal: LayerSpec = (1, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates a lumped element resonator with interdigital capacitor and spiral inductor.

    A lumped resonator consists of a capacitive element (interdigital capacitor)
    and an inductive element (spiral inductor) forming an LC circuit.

    Args:
        capacitor_fingers: Number of fingers in the interdigital capacitor.
        capacitor_finger_length: Length of each capacitor finger in μm.
        capacitor_finger_gap: Gap between capacitor fingers in μm.
        capacitor_thickness: Thickness of capacitor fingers in μm.
        inductor_width: Width of the inductor wire in μm.
        inductor_turns: Number of turns in the spiral inductor.
        inductor_radius: Radius of the spiral inductor in μm.
        coupling_gap: Gap for capacitive coupling in μm.
        layer_metal: Layer for the metal structures.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the lumped resonator geometry.
    """
    c = Component()

    # Create interdigital capacitor
    capacitor = gf.get_component(
        "interdigital_capacitor",
        fingers=capacitor_fingers,
        finger_length=capacitor_finger_length,
        finger_gap=capacitor_finger_gap,
        thickness=capacitor_thickness,
        layer=layer_metal,
    )
    c.add_ref(capacitor)

    # Create spiral inductor
    # Create a cross section with the specified width
    inductor_cross_section = gf.cross_section.strip(
        width=inductor_width,
        layer=layer_metal,
    )
    inductor = gf.components.spiral(
        n_loops=inductor_turns,
        cross_section=inductor_cross_section,
    )
    ind_ref = c.add_ref(inductor)

    # Position inductor next to capacitor
    cap_width = 2 * capacitor_thickness + capacitor_finger_length + capacitor_finger_gap
    ind_ref.move((cap_width + 20.0, 0))

    # Connect capacitor and inductor
    connection = gf.components.rectangle(
        size=(20.0, inductor_width),
        layer=layer_metal,
    )
    conn_ref = c.add_ref(connection)
    conn_ref.move((cap_width, -inductor_width / 2))

    # Connect to inductor input
    connection2 = gf.components.rectangle(
        size=(inductor_width, 20.0),
        layer=layer_metal,
    )
    conn2_ref = c.add_ref(connection2)
    conn2_ref.move((cap_width + 20.0 - inductor_width / 2, -20.0))

    # Add coupling elements for input/output
    coupling_cap_in = gf.components.rectangle(
        size=(capacitor_thickness, coupling_gap),
        layer=layer_metal,
    )
    coupling_in_ref = c.add_ref(coupling_cap_in)
    coupling_in_ref.move(
        (
            -coupling_gap - capacitor_thickness,
            capacitor_fingers * capacitor_thickness / 2,
        )
    )

    coupling_cap_out = gf.components.rectangle(
        size=(capacitor_thickness, coupling_gap),
        layer=layer_metal,
    )
    coupling_out_ref = c.add_ref(coupling_cap_out)
    coupling_out_ref.move(
        (cap_width + coupling_gap, capacitor_fingers * capacitor_thickness / 2)
    )

    # Add ports
    c.add_port(
        name="input",
        center=(
            -coupling_gap - capacitor_thickness / 2,
            capacitor_fingers * capacitor_thickness / 2,
        ),
        width=coupling_gap,
        orientation=180,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="output",
        center=(
            cap_width + coupling_gap + capacitor_thickness / 2,
            capacitor_fingers * capacitor_thickness / 2,
        ),
        width=coupling_gap,
        orientation=0,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add metadata
    c.info["resonator_type"] = "lumped"
    c.info["capacitor_fingers"] = capacitor_fingers
    c.info["inductor_turns"] = inductor_turns
    c.info["inductor_radius"] = inductor_radius

    return c


@gf.cell_with_module_name
def resonator_quarter_wave(
    length: float = 2500.0,
    width: float = 10.0,
    gap: float = 6.0,
    short_stub_length: float = 50.0,
    coupling_gap: float = 5.0,
    coupling_length: float = 100.0,
    layer_metal: LayerSpec = (1, 0),
    layer_gap: LayerSpec = (2, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates a quarter-wave coplanar waveguide resonator.

    A quarter-wave resonator is shorted at one end and has maximum electric field
    at the open end, making it suitable for capacitive coupling.

    Args:
        length: Length of the quarter-wave resonator in μm.
        width: Width of the center conductor in μm.
        gap: Gap width on each side of the center conductor in μm.
        short_stub_length: Length of the shorting stub in μm.
        coupling_gap: Gap for capacitive coupling in μm.
        coupling_length: Length of the coupling region in μm.
        layer_metal: Layer for the metal conductor.
        layer_gap: Layer for the gaps.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the quarter-wave resonator geometry.
    """
    c = Component()

    # Create main resonator line
    main_line = gf.components.rectangle(
        size=(length, width),
        layer=layer_metal,
    )
    c.add_ref(main_line)

    # Create shorting stub at one end
    short_stub = gf.components.rectangle(
        size=(short_stub_length, width + 2 * gap),
        layer=layer_metal,
    )
    short_ref = c.add_ref(short_stub)
    short_ref.move((length, -gap))

    # Create ground planes
    ground_top = gf.components.rectangle(
        size=(length + short_stub_length + 2 * gap, gap),
        layer=layer_metal,
    )
    ground_top_ref = c.add_ref(ground_top)
    ground_top_ref.move((-gap, width))

    ground_bottom = gf.components.rectangle(
        size=(length + short_stub_length + 2 * gap, gap),
        layer=layer_metal,
    )
    ground_bottom_ref = c.add_ref(ground_bottom)
    ground_bottom_ref.move((-gap, -gap))

    # Create coupling region at open end
    coupling_region = gf.components.rectangle(
        size=(coupling_length, coupling_gap),
        layer=layer_gap,
    )
    coupling_ref = c.add_ref(coupling_region)
    coupling_ref.move((-coupling_length, width / 2 - coupling_gap / 2))

    # Add port for coupling
    c.add_port(
        name="coupling",
        center=(-coupling_length / 2, width / 2),
        width=coupling_gap,
        orientation=180,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add metadata
    c.info["resonator_type"] = "quarter_wave"
    c.info["length"] = length
    c.info["width"] = width
    c.info["gap"] = gap
    c.info["frequency_estimate"] = (
        3e8 / (4 * length * 1e-6) / 1e9
    )  # GHz, rough estimate

    return c

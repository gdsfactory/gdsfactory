from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
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
    c = Component()

    # Create left capacitor pad
    left_pad = gf.components.rectangle(
        size=(pad_width, pad_height),
        layer=layer_metal,
        port_type=port_type,
    )
    left_pad_ref = c.add_ref(left_pad)
    left_pad_ref.move((-pad_width - pad_gap / 2, -pad_height / 2))

    # Create right capacitor pad
    right_pad = gf.components.rectangle(
        size=(pad_width, pad_height),
        layer=layer_metal,
        port_type=port_type,
    )
    right_pad_ref = c.add_ref(right_pad)
    right_pad_ref.move((pad_gap / 2, -pad_height / 2))

    # Create central island
    island = gf.components.rectangle(
        size=(island_width, island_height),
        layer=layer_island,
    )
    island_ref = c.add_ref(island)
    island_ref.move((-island_width / 2, -island_height / 2))

    # Create Josephson junction
    junction = gf.components.rectangle(
        size=(junction_width, junction_height),
        layer=layer_junction,
    )
    junction_ref = c.add_ref(junction)
    junction_ref.move((-junction_width / 2, -junction_height / 2))

    # Add connection lines from pads to island
    # Only add connections if there is a gap between island and pads
    connection_width = abs(pad_gap / 2 - island_width / 2)
    if pad_gap / 2 > island_width / 2:
        left_connection = gf.components.rectangle(
            size=(connection_width, junction_height / 2),
            layer=layer_metal,
        )
        left_conn_ref = c.add_ref(left_connection)
        left_conn_ref.move((-pad_gap / 2, -junction_height / 4))

        right_connection = gf.components.rectangle(
            size=(connection_width, junction_height / 2),
            layer=layer_metal,
        )
        right_conn_ref = c.add_ref(right_connection)
        right_conn_ref.move((island_width / 2, -junction_height / 4))

    # Add ports for connections
    c.add_port(
        name="left_pad",
        center=(-pad_width - pad_gap / 2, 0),
        width=pad_height,
        orientation=180,
        layer=layer_metal,
    )

    c.add_port(
        name="right_pad",
        center=(pad_width + pad_gap / 2, 0),
        width=pad_height,
        orientation=0,
        layer=layer_metal,
    )

    # Add metadata
    c.info["qubit_type"] = "transmon"
    c.info["pad_width"] = pad_width
    c.info["pad_height"] = pad_height
    c.info["pad_gap"] = pad_gap
    c.info["junction_area"] = junction_width * junction_height

    return c


@gf.cell_with_module_name
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
    c = Component()

    # Create left circular pad
    left_pad = gf.components.circle(
        radius=pad_radius,
        layer=layer_metal,
    )
    left_pad_ref = c.add_ref(left_pad)
    left_pad_ref.move((-pad_radius - pad_gap / 2, 0))

    # Create right circular pad
    right_pad = gf.components.circle(
        radius=pad_radius,
        layer=layer_metal,
    )
    right_pad_ref = c.add_ref(right_pad)
    right_pad_ref.move((pad_radius + pad_gap / 2, 0))

    # Create central circular island
    island = gf.components.circle(
        radius=island_radius,
        layer=layer_island,
    )
    c.add_ref(island)

    # Create Josephson junction
    junction = gf.components.rectangle(
        size=(junction_width, junction_height),
        layer=layer_junction,
    )
    junction_ref = c.add_ref(junction)
    junction_ref.move((-junction_width / 2, -junction_height / 2))

    # Add connection lines from pads to island
    # Only add connections if there is a gap between island and pads
    connection_width = abs(pad_gap / 2 - island_radius)
    if pad_gap / 2 > island_radius:
        left_connection = gf.components.rectangle(
            size=(connection_width, junction_height / 2),
            layer=layer_metal,
        )
        left_conn_ref = c.add_ref(left_connection)
        left_conn_ref.move((-pad_gap / 2, -junction_height / 4))

        right_connection = gf.components.rectangle(
            size=(connection_width, junction_height / 2),
            layer=layer_metal,
        )
        right_conn_ref = c.add_ref(right_connection)
        right_conn_ref.move((island_radius, -junction_height / 4))

    # Add ports for connections
    c.add_port(
        name="left_pad",
        center=(-2 * pad_radius - pad_gap / 2, 0),
        width=2 * pad_radius,
        orientation=180,
        layer=layer_metal,
    )

    c.add_port(
        name="right_pad",
        center=(2 * pad_radius + pad_gap / 2, 0),
        width=2 * pad_radius,
        orientation=0,
        layer=layer_metal,
    )

    # Add metadata
    c.info["qubit_type"] = "transmon_circular"
    c.info["pad_radius"] = pad_radius
    c.info["pad_gap"] = pad_gap
    c.info["junction_area"] = junction_width * junction_height

    return c

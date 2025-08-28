from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def coupler_capacitive(
    pad_width: float = 20.0,
    pad_height: float = 50.0,
    gap: float = 2.0,
    feed_width: float = 10.0,
    feed_length: float = 30.0,
    layer_metal: LayerSpec = (1, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates a capacitive coupler for quantum circuits.

    A capacitive coupler consists of two metal pads separated by a small gap,
    providing capacitive coupling between circuit elements like qubits and resonators.

    .. code::
                    ______               ______
          _______  |      |             |      | _______
         |       | |      |             |      ||       |
         | feed1 | | pad1 | ====gap==== | pad2 || feed2 |
         |       | |      |             |      ||       |
         |_______| |      |             |      ||_______|
                   |______|             |______|

    Args:
        pad_width: Width of each coupling pad in μm.
        pad_height: Height of each coupling pad in μm.
        gap: Gap between the coupling pads in μm.
        feed_width: Width of the feed lines in μm.
        feed_length: Length of the feed lines in μm.
        layer_metal: Layer for the metal structures.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the capacitive coupler geometry.
    """
    c = Component()

    # Create left coupling pad
    left_pad = gf.components.rectangle(
        size=(pad_width, pad_height),
        layer=layer_metal,
    )
    left_pad_ref = c.add_ref(left_pad)
    left_pad_ref.move((-pad_width - gap / 2, -pad_height / 2))

    # Create right coupling pad
    right_pad = gf.components.rectangle(
        size=(pad_width, pad_height),
        layer=layer_metal,
    )
    right_pad_ref = c.add_ref(right_pad)
    right_pad_ref.move((gap / 2, -pad_height / 2))

    # Create left feed line
    left_feed = gf.components.rectangle(
        size=(feed_length, feed_width),
        layer=layer_metal,
    )
    left_feed_ref = c.add_ref(left_feed)
    left_feed_ref.move((-pad_width - gap / 2 - feed_length, -feed_width / 2))

    # Create right feed line
    right_feed = gf.components.rectangle(
        size=(feed_length, feed_width),
        layer=layer_metal,
    )
    right_feed_ref = c.add_ref(right_feed)
    right_feed_ref.move((gap / 2 + pad_width, -feed_width / 2))

    # Add ports
    c.add_port(
        name="left",
        center=(-pad_width - gap / 2 - feed_length, 0),
        width=feed_width,
        orientation=180,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="right",
        center=(gap / 2 + pad_width + feed_length, 0),
        width=feed_width,
        orientation=0,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add metadata
    c.info["coupler_type"] = "capacitive"
    c.info["pad_width"] = pad_width
    c.info["pad_height"] = pad_height
    c.info["gap"] = gap
    c.info["coupling_area"] = pad_width * pad_height

    return c


@gf.cell_with_module_name
def coupler_interdigital(
    fingers: int = 6,
    finger_length: float = 30.0,
    finger_width: float = 2.0,
    finger_gap_vertical: float = 2.0,
    finger_gap_horizontal: float = 3.0,
    feed_width: float = 10.0,
    feed_length: float = 30.0,
    layer_metal: LayerSpec = (1, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates an interdigital capacitive coupler.

    Each side includes a base column (a vertical metal block) to which the fingers are attached.
    - The width of the base column is equal to the height of the fingers.
    - The finger_length parameter refers only to the length of the fingers *extending from the base*,
      and does NOT include the base column width

    Args:
        fingers: Number of fingers per side.
        finger_length: Length of each finger in μm (see note above).
        finger_width: Width of each finger in μm.
        finger_gap_vertical: Vertical gap between fingers in μm (g1).
        finger_gap_horizontal: Horizontal gap between fingers in μm (g2).
        feed_width: Width of the feed lines in μm.
        feed_length: Length of the feed lines in μm.
        layer_metal: Layer for the metal structures.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the interdigital coupler geometry.

    .. code::
                    ┌────────┐
                   base columns
                   ↓                    ↓
         ┌────────┐                      ┌────────┐
         │        │█████████████        █│        │
         │        │█        g1          █│        │
         │        │█ <─g2─> █████████████│        │
         │        │█                    █│        │
         │ feed1  │█████████████        █│ feed2  │
         │        │█                    █│        │
         │        │█        █████████████│        │
         │        │█                    █│        │
         │        │█████████████        █│        │
         └────────┘█                    █└────────┘

    """
    c = Component()

    # Calculate total dimensions
    total_width = finger_length + finger_gap_horizontal
    total_height = fingers * finger_width + (fingers - 1) * finger_gap_vertical

    # Create left side base column
    left_base = gf.components.rectangle(
        size=(finger_width, total_height),
        layer=layer_metal,
    )
    left_base_ref = c.add_ref(left_base)
    left_base_ref.move((-total_width / 2 - finger_width, -total_height / 2))

    # Create right side base column
    right_base = gf.components.rectangle(
        size=(finger_width, total_height),
        layer=layer_metal,
    )
    right_base_ref = c.add_ref(right_base)
    right_base_ref.move((total_width / 2, -total_height / 2))

    # Create interdigital fingers
    for i in range(fingers):
        left_finger = gf.components.rectangle(
            size=(finger_length, finger_width),
            layer=layer_metal,
        )
        left_finger_ref = c.add_ref(left_finger)

        # We start from a left finger
        x_pos = -finger_length / 2 + (-1) ** (i + 1) * finger_gap_horizontal / 2
        y_pos = (
            total_height / 2 - finger_width - i * (finger_width + finger_gap_vertical)
        )
        left_finger_ref.move((x_pos, y_pos))

    # Create feed lines
    left_feed = gf.components.rectangle(
        size=(feed_length, feed_width),
        layer=layer_metal,
    )
    left_feed_ref = c.add_ref(left_feed)
    left_feed_ref.move((-total_width / 2 - finger_width - feed_length, -feed_width / 2))

    right_feed = gf.components.rectangle(
        size=(feed_length, feed_width),
        layer=layer_metal,
    )
    right_feed_ref = c.add_ref(right_feed)
    right_feed_ref.move((total_width / 2 + finger_width, -feed_width / 2))

    # Add ports
    c.add_port(
        name="left",
        center=(-total_width / 2 - finger_width - feed_length, 0),
        width=feed_width,
        orientation=180,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="right",
        center=(total_width / 2 + finger_width + feed_length, 0),
        width=feed_width,
        orientation=0,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add metadata
    c.info["coupler_type"] = "interdigital"
    c.info["fingers"] = fingers
    c.info["finger_length"] = finger_length
    c.info["finger_width"] = finger_width
    c.info["finger_gap_horizontal"] = finger_gap_horizontal
    c.info["finger_gap_vertical"] = finger_gap_vertical

    return c


@gf.cell_with_module_name
def coupler_tunable(
    pad_width: float = 30.0,
    pad_height: float = 40.0,
    gap: float = 3.0,
    tuning_pad_width: float = 15.0,
    tuning_pad_height: float = 20.0,
    tuning_gap: float = 1.0,
    feed_width: float = 10.0,
    feed_length: float = 30.0,
    layer_metal: LayerSpec = (1, 0),
    layer_tuning: LayerSpec = (3, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates a tunable capacitive coupler with voltage control.

    A tunable coupler includes additional electrodes that can be voltage-biased
    to change the coupling strength dynamically.


    Args:
        pad_width: Width of main coupling pads in μm.
        pad_height: Height of main coupling pads in μm.
        gap: Gap between main coupling pads in μm.
        tuning_pad_width: Width of tuning pads in μm.
        tuning_pad_height: Height of tuning pads in μm.
        tuning_gap: Gap to tuning pads in μm.
        feed_width: Width of feed lines in μm.
        feed_length: Length of feed lines in μm.
        layer_metal: Layer for main metal structures.
        layer_tuning: Layer for tuning electrodes.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the tunable coupler geometry.


    .. code::

                    (connected to feed)
                         _______
                        |       |
                        | tpad1 |
                        |       |
                        |_______|
                        tuning gap
                   ______        ______
         _______  |      |      |      | _______
        |       | |      |      |      ||       |
        | feed1 | | pad1 | gap  | pad2 || feed2 |
        |       | |      |      |      ||       |
        |_______| |      |      |      ||_______|
                  |______|      |______|
                        tuning gap
                         _______
                        |       |
                        | tpad2 |
                        |       |
                        |_______|
                    (connected to feed)
    """
    c = Component()

    # Create main coupling pads
    left_pad = gf.components.rectangle(
        size=(pad_width, pad_height),
        layer=layer_metal,
    )
    left_pad_ref = c.add_ref(left_pad)
    left_pad_ref.move((-pad_width - gap / 2, -pad_height / 2))

    right_pad = gf.components.rectangle(
        size=(pad_width, pad_height),
        layer=layer_metal,
    )
    right_pad_ref = c.add_ref(right_pad)
    right_pad_ref.move((gap / 2, -pad_height / 2))

    # Create tuning pads above and below
    top_tuning_pad = gf.components.rectangle(
        size=(tuning_pad_width, tuning_pad_height),
        layer=layer_tuning,
    )
    top_tuning_ref = c.add_ref(top_tuning_pad)
    top_tuning_ref.move((-tuning_pad_width / 2, pad_height / 2 + tuning_gap))

    bottom_tuning_pad = gf.components.rectangle(
        size=(tuning_pad_width, tuning_pad_height),
        layer=layer_tuning,
    )
    bottom_tuning_ref = c.add_ref(bottom_tuning_pad)
    bottom_tuning_ref.move(
        (-tuning_pad_width / 2, -pad_height / 2 - tuning_gap - tuning_pad_height)
    )

    # Create feed lines for main pads
    left_feed = gf.components.rectangle(
        size=(feed_length, feed_width),
        layer=layer_metal,
    )
    left_feed_ref = c.add_ref(left_feed)
    left_feed_ref.move((-pad_width - gap / 2 - feed_length, -feed_width / 2))

    right_feed = gf.components.rectangle(
        size=(feed_length, feed_width),
        layer=layer_metal,
    )
    right_feed_ref = c.add_ref(right_feed)
    right_feed_ref.move((gap / 2 + pad_width, -feed_width / 2))

    # Create tuning feed lines
    top_tuning_feed = gf.components.rectangle(
        size=(feed_width, feed_length),
        layer=layer_tuning,
    )
    top_tuning_feed_ref = c.add_ref(top_tuning_feed)
    top_tuning_feed_ref.move(
        (-feed_width / 2, pad_height / 2 + tuning_gap + tuning_pad_height)
    )

    bottom_tuning_feed = gf.components.rectangle(
        size=(feed_width, feed_length),
        layer=layer_tuning,
    )
    bottom_tuning_feed_ref = c.add_ref(bottom_tuning_feed)
    bottom_tuning_feed_ref.move(
        (
            -feed_width / 2,
            -pad_height / 2 - tuning_gap - tuning_pad_height - feed_length,
        )
    )

    # Add ports
    c.add_port(
        name="left",
        center=(-pad_width - gap / 2 - feed_length, 0),
        width=feed_width,
        orientation=180,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="right",
        center=(gap / 2 + pad_width + feed_length, 0),
        width=feed_width,
        orientation=0,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="tuning_top",
        center=(0, pad_height / 2 + tuning_gap + tuning_pad_height + feed_length),
        width=feed_width,
        orientation=90,
        layer=layer_tuning,
        port_type=port_type,
    )

    c.add_port(
        name="tuning_bottom",
        center=(0, -pad_height / 2 - tuning_gap - tuning_pad_height - feed_length),
        width=feed_width,
        orientation=270,
        layer=layer_tuning,
        port_type=port_type,
    )

    # Add metadata
    c.info["coupler_type"] = "tunable"
    c.info["pad_width"] = pad_width
    c.info["pad_height"] = pad_height
    c.info["gap"] = gap
    c.info["tuning_pad_width"] = tuning_pad_width
    c.info["tuning_pad_height"] = tuning_pad_height
    c.info["tuning_gap"] = tuning_gap

    return c

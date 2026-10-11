from __future__ import annotations

__all__ = ["coupler_capacitive", "coupler_interdigital", "coupler_tunable"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["quantum"])
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

    ```text
                    ______               ______
          _______  |      |             |      | _______
         |       | |      |             |      ||       |
         | feed1 | | pad1 | ====gap==== | pad2 || feed2 |
         |       | |      |             |      ||       |
         |_______| |      |             |      ||_______|
                   |______|             |______|
    ```

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
    return cf.coupler_capacitive(
        pad_width=pad_width,
        pad_height=pad_height,
        gap=gap,
        feed_width=feed_width,
        feed_length=feed_length,
        layer_metal=layer_metal,
        port_type=port_type,
    )


@gf.cell_with_module_name(tags=["quantum"])
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

    ```text
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
    ```

    """
    return cf.coupler_interdigital(
        fingers=fingers,
        finger_length=finger_length,
        finger_width=finger_width,
        finger_gap_vertical=finger_gap_vertical,
        finger_gap_horizontal=finger_gap_horizontal,
        feed_width=feed_width,
        feed_length=feed_length,
        layer_metal=layer_metal,
        port_type=port_type,
    )


@gf.cell_with_module_name(tags=["quantum"])
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

    ```text
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
    ```
    """
    return cf.coupler_tunable(
        pad_width=pad_width,
        pad_height=pad_height,
        gap=gap,
        tuning_pad_width=tuning_pad_width,
        tuning_pad_height=tuning_pad_height,
        tuning_gap=tuning_gap,
        feed_width=feed_width,
        feed_length=feed_length,
        layer_metal=layer_metal,
        layer_tuning=layer_tuning,
        port_type=port_type,
    )

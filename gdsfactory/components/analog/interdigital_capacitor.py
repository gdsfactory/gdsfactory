from __future__ import annotations

from itertools import chain
from math import ceil, floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def interdigital_capacitor(
    fingers: int = 4,
    finger_length: float | int = 20.0,
    finger_gap: float | int = 2.0,
    thickness: float | int = 5.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Generates an interdigital capacitor with ports on both ends.

    See for example Zhu et al., `Accurate circuit model of interdigital
    capacitor and its application to design of new uasi-lumped miniaturized
    filters with suppression of harmonic resonance`, doi: 10.1109/22.826833.

    Note:
        ``finger_length=0`` effectively provides a plate capacitor.

    Args:
        fingers: total fingers of the capacitor.
        finger_length: length of the probing fingers.
        finger_gap: length of gap between the fingers.
        thickness: Thickness of fingers and section before the fingers.
        layer: spec.
    """
    c = Component()

    assert fingers >= 1, "Must have at least 1 finger"

    width = 2 * thickness + finger_length + finger_gap  # total length
    height = fingers * thickness + (fingers - 1) * finger_gap  # total height
    points_1 = [
        (0, 0),
        (0, height),
        (thickness + finger_length, height),
        (thickness + finger_length, height - thickness),
        (thickness, height - thickness),
        *chain.from_iterable(
            (
                (thickness, height - (2 * i) * (thickness + finger_gap)),
                (
                    thickness + finger_length,
                    height - (2 * i) * (thickness + finger_gap),
                ),
                (
                    thickness + finger_length,
                    height - (2 * i) * (thickness + finger_gap) - thickness,
                ),
                (thickness, height - (2 * i) * (thickness + finger_gap) - thickness),
            )
            for i in range(ceil(fingers / 2))
        ),
        (thickness, 0),
        (0, 0),
    ]

    points_2 = [
        (width, 0),
        (width, height),
        (width - thickness, height),
        *chain.from_iterable(
            (
                (
                    width - thickness,
                    height - (1 + 2 * i) * thickness - (1 + 2 * i) * finger_gap,
                ),
                (
                    width - (thickness + finger_length),
                    height - (1 + 2 * i) * thickness - (1 + 2 * i) * finger_gap,
                ),
                (
                    width - (thickness + finger_length),
                    height - (2 + 2 * i) * thickness - (1 + 2 * i) * finger_gap,
                ),
                (
                    width - thickness,
                    height - (2 + 2 * i) * thickness - (1 + 2 * i) * finger_gap,
                ),
            )
            for i in range(floor(fingers / 2))
        ),
        (width - thickness, 0),
        (width, 0),
    ]

    c.add_polygon(points_1, layer=layer)
    c.add_polygon(points_2, layer=layer)
    c.add_port(
        name="o1",
        center=(0, height / 2),
        width=thickness,
        orientation=180,
        layer=layer,
    )
    c.add_port(
        name="o2",
        center=(width, height / 2),
        width=thickness,
        orientation=0,
        layer=layer,
    )
    return c


if __name__ == "__main__":
    c = interdigital_capacitor()
    c.show()

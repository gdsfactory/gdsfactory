from __future__ import annotations

__all__ = ["comb_drive"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["mems"])
def comb_drive(
    finger_width: float = 0.5,
    finger_length: float = 10.0,
    finger_gap: float = 0.5,
    n_fingers: int = 20,
    finger_overlap: float = 5.0,
    shuttle_width: float = 5.0,
    shuttle_length: float = 30.0,
    spring_width: float = 0.5,
    spring_length: float = 20.0,
    n_spring_folds: int = 4,
    anchor_size: float = 10.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a comb drive actuator with interdigitated fingers and folded springs.

    A central shuttle with comb fingers on both sides, fixed electrodes
    with interleaving fingers, and folded springs connecting the shuttle
    to corner anchor pads.

    Args:
        finger_width: width of each comb finger.
        finger_length: length of each comb finger.
        finger_gap: gap between adjacent moving and fixed fingers.
        n_fingers: number of moving fingers on each side.
        finger_overlap: overlap length between moving and fixed fingers in the actuation direction.
        shuttle_width: width (vertical) of the shuttle mass.
        shuttle_length: length (horizontal) of the shuttle mass.
        spring_width: width of spring beam segments.
        spring_length: length of each spring fold segment.
        n_spring_folds: number of folds in each folded spring.
        anchor_size: size of each square anchor pad.
        layer: layer spec.
    """
    return cf.comb_drive(
        finger_width=finger_width,
        finger_length=finger_length,
        finger_gap=finger_gap,
        n_fingers=n_fingers,
        finger_overlap=finger_overlap,
        shuttle_width=shuttle_width,
        shuttle_length=shuttle_length,
        spring_width=spring_width,
        spring_length=spring_length,
        n_spring_folds=n_spring_folds,
        anchor_size=anchor_size,
        layer=layer,
    )


if __name__ == "__main__":
    c = comb_drive()
    c.show()

from __future__ import annotations

__all__ = [
    "via_stack",
    "via_stack_corner45",
    "via_stack_corner45_extended",
    "via_stack_heater_m2",
    "via_stack_heater_m3",
    "via_stack_heater_mtop",
    "via_stack_heater_mtop_mini",
    "via_stack_m1_m3",
    "via_stack_m1_mtop",
    "via_stack_m2_m3",
    "via_stack_npp_m1",
    "via_stack_slab_m1",
    "via_stack_slab_m1_horizontal",
    "via_stack_slab_m2",
    "via_stack_slab_m3",
    "via_stack_slab_npp_m3",
]

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, Floats, Ints, LayerSpec, LayerSpecs, Size


@gf.cell_with_module_name(tags=["vias"])
def via_stack(
    size: Size = (11.0, 11.0),
    layers: LayerSpecs = ("M1", "M2", "MTOP"),
    layer_offsets: Floats | tuple[float | tuple[float, float], ...] | None = None,
    vias: Sequence[ComponentSpec | None] = ("via1", "via2", None),
    layer_to_port_orientations: dict[LayerSpec, list[int]] | None = None,
    correct_size: bool = False,
    slot_horizontal: bool = False,
    slot_vertical: bool = False,
    port_orientations: Ints | None = (180, 90, 0, -90),
) -> Component:
    """Rectangular via array stack.

    You can use it to connect different metal layers or metals to silicon.
    You can use the naming convention via_stack_layerSource_layerDestination
    contains 4 ports (e1, e2, e3, e4)

    also know as Via array
    <http://www.vlsi-expert.com/2017/12/vias.html>

    Args:
        size: of the layers.
        layers: layers on which to draw rectangles.
        layer_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size. If a tuple, it is the offset in x and y.
        vias: vias to use to fill the rectangles.
        layer_to_port_orientations: dictionary of layer to port_orientations.
        correct_size: if True, if the specified dimensions are too small it increases
            them to the minimum possible to fit a via.
        slot_horizontal: if True, then vias are horizontal.
        slot_vertical: if True, then vias are vertical.
        port_orientations: list of port_orientations to add. None does not add ports.
    """
    return cf.via_stack(
        size=size,
        layers=layers,
        layer_offsets=layer_offsets,
        vias=vias,
        layer_to_port_orientations=layer_to_port_orientations,
        correct_size=correct_size,
        slot_horizontal=slot_horizontal,
        slot_vertical=slot_vertical,
        port_orientations=port_orientations,
    )


@gf.cell_with_module_name(tags=["vias"])
def via_stack_corner45(
    width: float = 10,
    layers: Sequence[LayerSpec | None] = ("M1", "M2", "MTOP"),
    layer_offsets: Floats | None = None,
    vias: Sequence[ComponentSpec | None] = ("via1", "via2", None),
    layer_port: LayerSpec | None = None,
    correct_size: bool = False,
) -> Component:
    """Rectangular via array stack at a 45 degree angle.

    Args:
        width: of the corner45.
        layers: layers on which to draw rectangles.
        layer_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
        vias: vias to use to fill the rectangles.
        layer_port: if None assumes port is on the last layer.
        correct_size: if True, if the specified dimensions are too small it increases
            them to the minimum possible to fit a via.
    """
    return cf.via_stack_corner45(
        width=width,
        layers=layers,
        layer_offsets=layer_offsets,
        vias=vias,
        layer_port=layer_port,
        correct_size=correct_size,
    )


@gf.cell_with_module_name(tags=["vias"])
def via_stack_corner45_extended(
    corner: ComponentSpec = "via_stack_corner45",
    via_stack: ComponentSpec = "via_stack",
    width: float = 3,
    length: float = 10,
) -> Component:
    """Rectangular via array stack at a 45 degree angle.

    Args:
        corner: corner component.
        via_stack: for the via stack.
        width: of the corner45.
        length: of the straight.
    """
    return cf.via_stack_corner45_extended(
        corner=corner,
        via_stack=via_stack,
        width=width,
        length=length,
    )


via_stack_m1_mtop = via_stack_m1_m3 = CellAlias(
    via_stack,
    layers=("M1", "M2", "MTOP"),
    vias=("via1", "via2", None),
)
via_stack_m2_m3 = CellAlias(
    via_stack,
    layers=("M2", "MTOP"),
    vias=("via2", None),
)
via_stack_slab_m1 = CellAlias(
    via_stack,
    layers=("SLAB90", "M1"),
    vias=("viac", "via1"),
)
via_stack_slab_m2 = CellAlias(
    via_stack,
    layers=("SLAB90", "M1", "M2"),
    vias=("viac", "via1", None),
)

via_stack_slab_m3 = CellAlias(
    via_stack,
    layers=("SLAB90", "M1", "M2", "MTOP"),
    vias=("viac", "via1", "via2", None),
)
via_stack_npp_m1 = CellAlias(
    via_stack,
    layers=("WG", "NPP", "M1"),
    vias=(None, None, "viac"),
)
via_stack_slab_npp_m3 = CellAlias(
    via_stack,
    layers=("SLAB90", "NPP", "M1"),
    vias=(None, None, "viac"),
)
via_stack_heater_mtop = via_stack_heater_m3 = CellAlias(
    via_stack, layers=("HEATER", "M2", "MTOP"), vias=(None, "via1", "via2")
)
via_stack_heater_mtop_mini = CellAlias(via_stack_heater_mtop, size=(4, 4))

via_stack_heater_m2 = CellAlias(via_stack, layers=("HEATER", "M2"), vias=(None, "via1"))

via_stack_slab_m1_horizontal = CellAlias(via_stack_slab_m1, slot_horizontal=True)


if __name__ == "__main__":
    c = via_stack_heater_mtop_mini(size=(1, 1), correct_size=True)
    c.show()

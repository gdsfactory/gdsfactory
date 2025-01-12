from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.superconductors.optimal_hairpin import optimal_hairpin
from gdsfactory.typings import LayerSpec, Port, Size


@gf.cell
def snspd(
    wire_width: float = 0.2,
    wire_pitch: float = 0.6,
    size: Size = (10, 8),
    num_squares: int | None = None,
    turn_ratio: float = 4,
    terminals_same_side: bool = False,
    layer: LayerSpec = (1, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates an optimally-rounded SNSPD.

    Args:
        wire_width: Width of the wire.
        wire_pitch: Distance between two adjacent wires. Must be greater than `width`.
        size: Float2
            (width, height) of the rectangle formed by the outer boundary of the
            SNSPD.
        num_squares: int | None = None
            Total number of squares inside the SNSPD length.
        turn_ratio: float
            Specifies how much of the SNSPD width is dedicated to the 180 degree
            turn. A `turn_ratio` of 10 will result in 20% of the width being
            comprised of the turn.
        terminals_same_side: If True, both ports will be located on the same side of the SNSPD.
        layer: layer spec to put polygon geometry on.
        port_type: type of port to add to the component.

    """
    if num_squares is not None:
        xy = np.sqrt(num_squares * wire_pitch * wire_width)
        size = (xy, xy)
        num_squares = None

    xsize, ysize = size
    if num_squares is not None:
        if xsize is None:
            xsize = num_squares * wire_pitch * wire_width / ysize
        elif ysize is None:
            ysize = num_squares * wire_pitch * wire_width / xsize

    num_meanders = int(np.ceil(ysize / wire_pitch))

    D = Component()
    hairpin = optimal_hairpin(
        width=wire_width,
        pitch=wire_pitch,
        turn_ratio=turn_ratio,
        length=xsize / 2,
        num_pts=20,
        layer=layer,
    )

    if not terminals_same_side and (num_meanders % 2) == 0:
        num_meanders += 1
    elif terminals_same_side and (num_meanders % 2) == 1:
        num_meanders += 1

    port_type = "electrical"

    start_nw = D.add_ref(
        gf.c.compass(size=(xsize / 2, wire_width), layer=layer, port_type=port_type)
    )
    hp_prev = D.add_ref(hairpin)
    hp_prev.connect("e1", start_nw.ports["e3"])
    alternate = True
    last_port: Port | None = None
    for _n in range(2, num_meanders):
        hp = D.add_ref(hairpin)
        if alternate:
            hp.connect("e2", hp_prev.ports["e2"])
        else:
            hp.connect("e1", hp_prev.ports["e1"])
        last_port = hp.ports["e2"] if terminals_same_side else hp.ports["e1"]
        hp_prev = hp
        alternate = not alternate

    finish_se = D.add_ref(
        gf.c.compass(size=(xsize / 2, wire_width), layer=layer, port_type=port_type)
    )
    if last_port is not None:
        finish_se.connect("e3", last_port)

    D.add_port(port=start_nw.ports["e1"], name="e1")
    D.add_port(port=finish_se.ports["e1"], name="e2")

    D.info["num_squares"] = num_meanders * (xsize / wire_width)
    D.info["area"] = xsize * ysize
    D.info["xsize"] = xsize
    D.info["ysize"] = ysize
    D.flatten()
    return D


if __name__ == "__main__":
    c = snspd()
    c.show()

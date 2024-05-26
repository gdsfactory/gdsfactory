from __future__ import annotations

import numpy as np

from gdsfactory import cell
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.optimal_hairpin import optimal_hairpin
from gdsfactory.typings import Float2, LayerSpec


@cell
def snspd(
    wire_width: float = 0.2,
    wire_pitch: float = 0.6,
    size: Float2 = (10, 8),
    num_squares: int | None = None,
    turn_ratio: float = 4,
    terminals_same_side: bool = False,
    layer: LayerSpec = (1, 0),
) -> Component:
    """Creates an optimally-rounded SNSPD.

    Args:
        wire_width: Width of the wire.
        wire_pitch: Distance between two adjacent wires. Must be greater than `width`.
        size: None or array-like[2] of int or float
            (width, height) of the rectangle formed by the outer boundary of the
            SNSPD. Must be none if `num_squares` is specified.
        num_squares: int or None
            Total number of squares inside the SNSPD length. Must be none if
            `size` is specified.
        turn_ratio: int or float
            Specifies how much of the SNSPD width is dedicated to the 180 degree
            turn. A `turn_ratio` of 10 will result in 20% of the width being
            comprised of the turn.
        terminals_same_side: If True, both ports will be located on the same side of the SNSPD.
        layer: layer spec to put polygon geometry on.

    """
    # Convenience tests to auto-shape the size based
    # on the number of squares
    if num_squares is not None and (
        (size is None) or ((size[0] is None) and (size[1]) is None)
    ):
        xy = np.sqrt(num_squares * wire_pitch * wire_width)
        size = (xy, xy)
        num_squares = None
    if [size[0], size[1], num_squares].count(None) != 1:
        raise ValueError(
            "snspd() requires that exactly ONE value of "
            "the arguments ``num_squares`` and ``size`` be None "
            "to prevent overconstraining, for example:\n"
            ">>> snspd(size = (3, None), num_squares = 2000)"
        )
    if size[0] is None:
        ysize = size[1]
        xsize = num_squares * wire_pitch * wire_width / ysize
    elif size[1] is None:
        xsize = size[0]
        ysize = num_squares * wire_pitch * wire_width / xsize
    else:
        xsize = size[0]
        ysize = size[1]

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

    port_type = "optical"
    start_nw = D.add_ref(
        compass(size=(xsize / 2, wire_width), layer=layer, port_type=port_type)
    )
    hp_prev = D.add_ref(hairpin)
    hp_prev.connect("o1", start_nw.ports["o3"])
    alternate = True
    for _n in range(2, num_meanders):
        hp = D.add_ref(hairpin)
        if alternate:
            hp.connect("o2", hp_prev.ports["o2"])
        else:
            hp.connect("o1", hp_prev.ports["o1"])
        last_port = hp.ports["o1"]
        hp_prev = hp
        alternate = not alternate

    finish_se = D.add_ref(
        compass(size=(xsize / 2, wire_width), layer=layer, port_type=port_type)
    )
    finish_se.connect("o3", last_port)

    D.add_port(port=start_nw.ports["o1"], name="o1")
    D.add_port(port=finish_se.ports["o1"], name="o2")

    D.info["num_squares"] = num_meanders * (xsize / wire_width)
    D.info["area"] = xsize * ysize
    D.info["xsize"] = xsize
    D.info["ysize"] = ysize
    return D


if __name__ == "__main__":
    c = snspd()
    c.show()

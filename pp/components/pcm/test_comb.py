from typing import Tuple

import pp
from pp import components as pc
from pp.component import Component


@pp.cell
def test_comb(
    pad_size: Tuple[int, int] = (200, 200),
    wire_width: int = 1,
    wire_gap: int = 3,
    comb_layer: Tuple[int, int] = pp.LAYER.M1,
    overlap_zigzag_layer: Tuple[int, int] = pp.LAYER.HEATER,
    comb_pad_layer: Tuple[int, int] = pp.LAYER.M3,
    comb_gnd_layer: Tuple[int, int] = pp.LAYER.M3,
    overlap_pad_layer: None = None,
) -> Component:
    """Superconducting heater component from phidl.geometry

    Args:
        pad_size=(200, 200)
        wire_width=1
        wire_gap=3
        comb_layer=0
        overlap_zigzag_layer=1
        comb_pad_layer=None
        comb_gnd_layer=None
        overlap_pad_layer=None

    """
    CI = pp.Component()

    if comb_pad_layer is None:
        comb_pad_layer = comb_layer
    if comb_gnd_layer is None:
        comb_gnd_layer = comb_layer
    if overlap_pad_layer is None:
        overlap_pad_layer = overlap_zigzag_layer
    wire_spacing = wire_width + wire_gap * 2

    # %% pad overlays
    overlay_padb = CI.add_ref(
        pc.rectangle(
            size=(pad_size[0] * 9 / 10, pad_size[1] * 9 / 10), layer=overlap_pad_layer
        )
    )
    overlay_padl = CI.add_ref(
        pc.rectangle(
            size=(pad_size[0] * 9 / 10, pad_size[1] * 9 / 10), layer=comb_pad_layer
        )
    )
    overlay_padt = CI.add_ref(
        pc.rectangle(
            size=(pad_size[0] * 9 / 10, pad_size[1] * 9 / 10), layer=comb_pad_layer
        )
    )
    overlay_padr = CI.add_ref(
        pc.rectangle(
            size=(pad_size[0] * 9 / 10, pad_size[1] * 9 / 10), layer=comb_gnd_layer
        )
    )

    overlay_padl.xmin = 0
    overlay_padl.ymin = 0
    overlay_padb.ymax = 0
    overlay_padb.xmin = overlay_padl.xmax + pad_size[1] / 5
    overlay_padr.ymin = overlay_padl.ymin
    overlay_padr.xmin = overlay_padb.xmax + pad_size[1] / 5
    overlay_padt.xmin = overlay_padl.xmax + pad_size[1] / 5
    overlay_padt.ymin = overlay_padl.ymax

    # %% pads
    padl = CI.add_ref(pc.rectangle(size=pad_size, layer=comb_layer))
    padt = CI.add_ref(pc.rectangle(size=pad_size, layer=comb_layer))
    padr = CI.add_ref(pc.rectangle(size=pad_size, layer=comb_layer))
    padb = CI.add_ref(pc.rectangle(size=pad_size, layer=overlap_zigzag_layer))
    padl_nub = CI.add_ref(
        pc.rectangle(size=(pad_size[0] / 4, pad_size[1] / 2), layer=comb_layer)
    )
    padr_nub = CI.add_ref(
        pc.rectangle(size=(pad_size[0] / 4, pad_size[1] / 2), layer=comb_layer)
    )

    padl.xmin = overlay_padl.xmin
    padl.center = [padl.center[0], overlay_padl.center[1]]
    padt.ymax = overlay_padt.ymax
    padt.center = [overlay_padt.center[0], padt.center[1]]
    padr.xmax = overlay_padr.xmax
    padr.center = [padr.center[0], overlay_padr.center[1]]
    padb.ymin = overlay_padb.ymin
    padb.center = [overlay_padb.center[0], padb.center[1]]
    padl_nub.xmin = padl.xmax
    padl_nub.center = [padl_nub.center[0], padl.center[1]]
    padr_nub.xmax = padr.xmin
    padr_nub.center = [padr_nub.center[0], padr.center[1]]

    # %% connected zig

    head = CI.add_ref(pc.compass(size=(pad_size[0] / 12, wire_width), layer=comb_layer))
    head.xmin = padl_nub.xmax
    head.ymax = padl_nub.ymax
    connector = CI.add_ref(pc.compass(size=(wire_width, wire_width), layer=comb_layer))
    connector.connect(port="W", destination=head.ports["E"])
    old_port = connector.ports["S"]
    top = True
    obj = connector
    while obj.xmax + pad_size[0] / 12 < padr_nub.xmin:
        # long zig zag rectangle
        obj = CI.add_ref(
            pc.compass(
                size=(pad_size[1] / 2 - 2 * wire_width, wire_width), layer=comb_layer
            )
        )
        obj.connect(port="W", destination=old_port)
        old_port = obj.ports["E"]
        if top:
            # zig zag edge rectangle
            obj = CI.add_ref(
                pc.compass(size=(wire_width, wire_width), layer=comb_layer)
            )
            obj.connect(port="N", destination=old_port)
            top = False
        else:
            # zig zag edge rectangle
            obj = CI.add_ref(
                pc.compass(size=(wire_width, wire_width), layer=comb_layer)
            )
            obj.connect(port="S", destination=old_port)
            top = True
            # comb rectange
            comb = CI.add_ref(
                pc.rectangle(
                    size=(
                        (padt.ymin - head.ymax)
                        + pad_size[1] / 2
                        - (wire_spacing + wire_width) / 2,
                        wire_width,
                    ),
                    layer=comb_layer,
                )
            )
            comb.rotate(90)
            comb.ymax = padt.ymin
            comb.xmax = obj.xmax - (wire_spacing + wire_width) / 2
        old_port = obj.ports["E"]
        obj = CI.add_ref(pc.compass(size=(wire_spacing, wire_width), layer=comb_layer))
        obj.connect(port="W", destination=old_port)
        old_port = obj.ports["E"]
        obj = CI.add_ref(pc.compass(size=(wire_width, wire_width), layer=comb_layer))
        obj.connect(port="W", destination=old_port)
        if top:
            old_port = obj.ports["S"]
        else:
            old_port = obj.ports["N"]
    old_port = obj.ports["E"]
    if padr_nub.xmin - obj.xmax > 0:
        tail = CI.add_ref(
            pc.compass(size=(padr_nub.xmin - obj.xmax, wire_width), layer=comb_layer)
        )
    else:
        tail = CI.add_ref(pc.compass(size=(wire_width, wire_width), layer=comb_layer))
    tail.connect(port="W", destination=old_port)

    # %% disconnected zig

    dhead = CI.add_ref(
        pc.compass(
            size=(padr_nub.ymin - padb.ymax - wire_width, wire_width),
            layer=overlap_zigzag_layer,
        )
    )
    dhead.rotate(90)
    dhead.ymin = padb.ymax
    dhead.xmax = tail.xmin - (wire_spacing + wire_width) / 2
    connector = CI.add_ref(
        pc.compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer)
    )
    connector.connect(port="S", destination=dhead.ports["E"])
    old_port = connector.ports["N"]
    right = True
    obj = connector
    while obj.ymax + wire_spacing + wire_width < head.ymax:
        obj = CI.add_ref(
            pc.compass(size=(wire_spacing, wire_width), layer=overlap_zigzag_layer)
        )
        obj.connect(port="W", destination=old_port)
        old_port = obj.ports["E"]
        if right:
            obj = CI.add_ref(
                pc.compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer)
            )
            obj.connect(port="W", destination=old_port)
            right = False
        else:
            obj = CI.add_ref(
                pc.compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer)
            )
            obj.connect(port="E", destination=old_port)
            right = True
        old_port = obj.ports["N"]
        obj = CI.add_ref(
            pc.compass(
                size=(
                    dhead.xmin - (head.xmax + head.xmin + wire_width) / 2,
                    wire_width,
                ),
                layer=overlap_zigzag_layer,
            )
        )
        obj.connect(port="E", destination=old_port)
        old_port = obj.ports["W"]
        obj = CI.add_ref(
            pc.compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer)
        )
        obj.connect(port="S", destination=old_port)
        if right:
            old_port = obj.ports["W"]
        else:
            old_port = obj.ports["E"]

    return CI


if __name__ == "__main__":
    c = test_comb()
    c.show()

from typing import Tuple

import pp
from pp import components as pc
from pp.component import Component


@pp.cell
def _via_iterable(
    via_spacing: int,
    wire_width: int,
    wiring1_layer: Tuple[int, int],
    wiring2_layer: Tuple[int, int],
    via_layer: Tuple[int, int],
    via_width: int,
) -> Component:
    VI = pp.Component()
    wire1 = VI.add_ref(pc.compass(size=(via_spacing, wire_width), layer=wiring1_layer))
    wire2 = VI.add_ref(pc.compass(size=(via_spacing, wire_width), layer=wiring2_layer))
    via1 = VI.add_ref(pc.compass(size=(via_width, via_width), layer=via_layer))
    via2 = VI.add_ref(pc.compass(size=(via_width, via_width), layer=via_layer))
    wire1.connect(port="E", destination=wire2.ports["W"], overlap=wire_width)
    via1.connect(
        port="W", destination=wire1.ports["E"], overlap=(wire_width + via_width) / 2
    )
    via2.connect(
        port="W", destination=wire2.ports["E"], overlap=(wire_width + via_width) / 2
    )
    VI.add_port(name="W", port=wire1.ports["W"])
    VI.add_port(name="E", port=wire2.ports["E"])
    VI.add_port(
        name="S",
        midpoint=[(1 * wire_width) + wire_width / 2, -wire_width / 2],
        width=wire_width,
        orientation=-90,
    )
    VI.add_port(
        name="N",
        midpoint=[(1 * wire_width) + wire_width / 2, wire_width / 2],
        width=wire_width,
        orientation=90,
    )

    return VI


@pp.cell
def test_via(
    num_vias: int = 100,
    wire_width: int = 10,
    via_width: int = 15,
    via_spacing: int = 40,
    pad_size: Tuple[int, int] = (300, 300),
    min_pad_spacing: int = 0,
    pad_layer: Tuple[int, int] = pp.LAYER.M3,
    wiring1_layer: Tuple[int, int] = pp.LAYER.HEATER,
    wiring2_layer: Tuple[int, int] = pp.LAYER.M1,
    via_layer: Tuple[int, int] = pp.LAYER.VIA1,
) -> Component:
    """ Via cutback to extract via resistance
    from phidl.geometry

    Args:
        num_vias=100
        wire_width=10
        via_width=15
        via_spacing=40
        pad_size=(300, 300)
        min_pad_spacing=0
        pad_layer=0
        wiring1_layer=1
        wiring2_layer=2
        via_layer=3

    Usage:
        Call via_route_test_structure() by indicating the number of vias you want drawn. You can also change the other parameters however
        if you do not specifiy a value for a parameter it will just use the default value
        Ex::

            via_route_test_structure(num_vias=54)

        - or -::

            via_route_test_structure(num_vias=12, pad_size=(100,100),wire_width=8)

        total requested vias (num_vias) -> this needs to be even
        pad size (pad_size) -> given in a pair (width, height)
        wire_width -> how wide each wire should be
        pad_layer -> GDS layer number of the pads
        wiring1_layer -> GDS layer number of the top wiring
        wiring2_layer -> GDS layer number of the bottom wiring
        via_layer -> GDS layer number of the vias
        ex: via_route(54, min_pad_spacing=300)

    .. plot::
      :include-source:

      import pp

      c = pp.c.test_via()
      pp.plotgds(c)
    """

    VR = pp.Component()
    pad1 = VR.add_ref(pc.rectangle(size=pad_size, layer=pad_layer))
    pad1_overlay = VR.add_ref(pc.rectangle(size=pad_size, layer=wiring1_layer))
    pad2 = VR.add_ref(pc.rectangle(size=pad_size, layer=pad_layer))
    pad2_overlay = VR.add_ref(pc.rectangle(size=pad_size, layer=wiring1_layer))
    nub = VR.add_ref(pc.compass(size=(3 * wire_width, wire_width), layer=pad_layer))
    nub_overlay = VR.add_ref(
        pc.compass(size=(3 * wire_width, wire_width), layer=wiring1_layer)
    )
    head = VR.add_ref(pc.compass(size=(wire_width, wire_width), layer=pad_layer))
    head_overlay = VR.add_ref(
        pc.compass(size=(wire_width, wire_width), layer=wiring1_layer)
    )
    nub.ymax = pad1.ymax - 5
    nub.xmin = pad1.xmax
    nub_overlay.ymax = pad1.ymax - 5
    nub_overlay.xmin = pad1.xmax
    head.connect(port="W", destination=nub.ports["E"])
    head_overlay.connect(port="W", destination=nub_overlay.ports["E"])
    pad1_overlay.xmin = pad1.xmin
    pad1_overlay.ymin = pad1.ymin

    old_port = head.ports["S"]
    count = 0
    width_via_iter = 2 * via_spacing - 2 * wire_width

    pad2.xmin = pad1.xmax + min_pad_spacing
    up = False
    down = True
    edge = True
    current_width = 3 * wire_width + wire_width  # width of nub and 1 overlap
    obj_old = head
    obj = head
    via_iterable = _via_iterable(
        via_spacing=via_spacing,
        wire_width=wire_width,
        wiring1_layer=wiring1_layer,
        wiring2_layer=wiring2_layer,
        via_layer=via_layer,
        via_width=via_width,
    )
    while (count + 2) <= num_vias:
        obj = VR.add_ref(via_iterable)
        obj.connect(port="W", destination=old_port, overlap=wire_width)
        old_port = obj.ports["E"]
        edge = False
        if obj.ymax > pad1.ymax:
            obj.connect(port="W", destination=obj_old.ports["S"], overlap=wire_width)
            old_port = obj.ports["S"]
            current_width += width_via_iter
            down = True
            up = False
            edge = True

        elif obj.ymin < pad1.ymin:
            obj.connect(port="W", destination=obj_old.ports["N"], overlap=wire_width)
            old_port = obj.ports["N"]
            current_width += width_via_iter
            up = True
            down = False
            edge = True
        count = count + 2
        obj_old = obj

    if (
        current_width < min_pad_spacing
        and (min_pad_spacing - current_width) > 3 * wire_width
    ):
        tail = VR.add_ref(
            pc.compass(
                size=(min_pad_spacing - current_width + wire_width, wire_width),
                layer=wiring1_layer,
            )
        )
        tail_overlay = VR.add_ref(
            pc.compass(
                size=(min_pad_spacing - current_width + wire_width, wire_width),
                layer=pad_layer,
            )
        )
    else:
        tail = VR.add_ref(
            pc.compass(size=(3 * wire_width, wire_width), layer=wiring1_layer)
        )
        tail_overlay = VR.add_ref(
            pc.compass(size=(3 * wire_width, wire_width), layer=wiring1_layer)
        )

    if up and not edge:
        tail.connect(port="W", destination=obj.ports["S"], overlap=wire_width)
        tail_overlay.connect(port="W", destination=obj.ports["S"], overlap=wire_width)
    elif down and not edge:
        tail.connect(port="W", destination=obj.ports["N"], overlap=wire_width)
        tail_overlay.connect(port="W", destination=obj.ports["N"], overlap=wire_width)
    else:
        tail.connect(port="W", destination=obj.ports["E"], overlap=wire_width)
        tail_overlay.connect(port="W", destination=obj.ports["E"], overlap=wire_width)

    pad2.xmin = tail.xmax
    pad2_overlay.xmin = pad2.xmin
    pad2_overlay.ymin = pad2.ymin

    return VR


if __name__ == "__main__":
    c = test_via()
    pp.show(c)

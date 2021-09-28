"""Via cutback."""

from typing import Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.via_stack import via_stack_heater
from gdsfactory.types import ComponentFactory, Float2


@gf.cell
def _via_iterable(
    via_spacing: float,
    wire_width: float,
    layer1: Tuple[int, int],
    layer2: Tuple[int, int],
    via_layer: Tuple[int, int],
    via_width: float,
) -> Component:
    """Via"""
    c = gf.Component()
    wire1 = c.add_ref(compass(size=(via_spacing, wire_width), layer=layer1))
    wire2 = c.add_ref(compass(size=(via_spacing, wire_width), layer=layer2))
    viac = c.add_ref(compass(size=(via_width, via_width), layer=via_layer))
    via1 = c.add_ref(compass(size=(via_width, via_width), layer=via_layer))
    wire1.connect(port="e3", destination=wire2.ports["e1"], overlap=wire_width)
    viac.connect(
        port="e1", destination=wire1.ports["e3"], overlap=(wire_width + via_width) / 2
    )
    via1.connect(
        port="e1", destination=wire2.ports["e3"], overlap=(wire_width + via_width) / 2
    )
    c.add_port(name="e1", port=wire1.ports["e1"], port_type="electrical")
    c.add_port(name="e3", port=wire2.ports["e3"], port_type="electrical")
    c.add_port(
        name="e4",
        midpoint=[(1 * wire_width) + wire_width / 2, -wire_width / 2],
        width=wire_width,
        orientation=-90,
        port_type="electrical",
    )
    c.add_port(
        name="e2",
        midpoint=[(1 * wire_width) + wire_width / 2, wire_width / 2],
        width=wire_width,
        orientation=90,
        port_type="electrical",
    )

    return c


@gf.cell
def via_cutback(
    num_vias: float = 100.0,
    wire_width: float = 10.0,
    via_width: float = 5.0,
    via_spacing: float = 40.0,
    min_pad_spacing: float = 0.0,
    pad: ComponentFactory = via_stack_heater,
    pad_size: Float2 = (150, 150),
    layer1: Tuple[int, int] = gf.LAYER.HEATER,
    layer2: Tuple[int, int] = gf.LAYER.M1,
    via_layer: Tuple[int, int] = gf.LAYER.VIAC,
    wire_pad_inclusion: float = 12.0,
) -> Component:
    """Via cutback to extract via resistance

    adapted from phidl.geometry

    Args:
        num_vias: total requested vias needs to be even
        wire_width: width of wire
        via_width: width of via
        via_spacing: via_spacing
        pad_size: (width, height)
        min_pad_spacing
        pad_layer
        layer1: top wiring
        layer2: bottom wiring
        via_layer
        wire_pad_inclusion:

    """

    c = gf.Component()

    pad_component = pad(size=pad_size)
    pad1 = c.add_ref(pad_component)
    pad2 = c.add_ref(pad_component)

    nub = c.add_ref(pad(size=(3 * wire_width, wire_width)))
    head = c.add_ref(pad(size=(wire_width, wire_width)))
    nub.ymax = pad1.ymax - 5
    nub.xmin = pad1.xmax - wire_pad_inclusion
    head.connect(port="e1", destination=nub.ports["e3"])

    old_port = head.ports["e4"]
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
        layer1=layer1,
        layer2=layer2,
        via_layer=via_layer,
        via_width=via_width,
    )
    while (count + 2) <= num_vias:
        obj = c.add_ref(via_iterable)
        obj.connect(port="e1", destination=old_port, overlap=wire_width)
        old_port = obj.ports["e3"]
        edge = False
        if obj.ymax > pad1.ymax:
            obj.connect(port="e1", destination=obj_old.ports["e4"], overlap=wire_width)
            old_port = obj.ports["e4"]
            current_width += width_via_iter
            down = True
            up = False
            edge = True

        elif obj.ymin < pad1.ymin:
            obj.connect(port="e1", destination=obj_old.ports["e2"], overlap=wire_width)
            old_port = obj.ports["e2"]
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
        tail = c.add_ref(
            pad(
                size=(min_pad_spacing - current_width + wire_width, wire_width),
            )
        )
    else:
        tail = c.add_ref(pad(size=(3 * wire_width, wire_width)))

    if up and not edge:
        tail.connect(port="e1", destination=obj.ports["e4"], overlap=wire_width)
    elif down and not edge:
        tail.connect(port="e1", destination=obj.ports["e2"], overlap=wire_width)
    else:
        tail.connect(port="e1", destination=obj.ports["e3"], overlap=wire_width)

    pad2.xmin = tail.xmax - wire_pad_inclusion
    return c


if __name__ == "__main__":
    c = via_cutback()
    c.show()

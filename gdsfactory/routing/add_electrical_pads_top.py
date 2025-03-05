from __future__ import annotations

from typing import Any, Literal

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import select_ports_electrical
from gdsfactory.routing.route_quad import route_quad
from gdsfactory.routing.sort_ports import sort_ports_x
from gdsfactory.typings import ComponentSpec, Float2, LayerSpec, SelectPorts, Strs


def add_electrical_pads_top(
    component: ComponentSpec,
    direction: Literal["top", "right"] = "top",
    spacing: Float2 = (0.0, 100.0),
    pad_array: ComponentSpec = "pad_array",
    select_ports: SelectPorts = select_ports_electrical,
    port_names: Strs | None = None,
    layer: LayerSpec = "MTOP",
    **kwargs: Any,
) -> Component:
    """Returns new component with electrical ports connected to top pad array.

    Args:
        component: to route.
        direction: sets direction of the array (top or right).
        spacing: component to pad spacing.
        pad_array: function for pad_array.
        select_ports: function to select electrical ports.
        port_names: optional port names. Overrides select_ports.
        layer: for the routes.
        kwargs: additional arguments.

    Keyword Args:
        ports: Dict[str, Port] a port dict {port name: port}.
        prefix: select ports with port name prefix.
        suffix: select ports with port name suffix.
        orientation: select ports with orientation in degrees.
        width: select ports with port width.
        layers_excluded: List of layers to exclude.
        port_type: select ports with port type (optical, electrical, vertical_te).
        clockwise: if True, sort ports clockwise, False: counter-clockwise.


    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.wire_straight(length=200.)
        cc = gf.routing.add_electrical_pads_top(component=c, spacing=(-150, 30))
        cc.plot()

    """
    c = Component()
    component = gf.get_component(component)
    ref = c << component

    ports = [ref[port_name] for port_name in port_names] if port_names else None
    ports_electrical = ports or select_ports(ref.ports, **kwargs)

    if not ports_electrical:
        raise ValueError("No electrical ports found")

    if direction == "top":
        pads = c << gf.get_component(
            pad_array, columns=len(ports_electrical), rows=1, port_orientation=270
        )
    elif direction == "right":
        pads = c << gf.get_component(
            pad_array, columns=1, rows=len(ports_electrical), orientation=270
        )
    else:
        raise ValueError(f"Invalid direction {direction}")

    pads.dx = ref.dx + spacing[0]
    pads.dymin = ref.dymax + spacing[1]

    ports_pads = sort_ports_x(pads.ports)
    ports_component = sort_ports_x(ports_electrical)

    for p1, p2 in zip(ports_component, ports_pads):
        route_quad(c, p1, p2, layer=layer)

    for port in ref.ports:
        if port.port_type != "electrical":
            c.add_port(name=port.name, port=port)

    c.add_ports(pads.ports)
    c.copy_child_info(component)
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    from gdsfactory.components import straight_heater_metal

    c = straight_heater_metal()
    # c = gf.components.mzi_phase_shifter_top_heater_metal()
    # cc = gf.routing.add_electrical_pads_top(component=c, spacing=(-150, 30))
    c = add_electrical_pads_top(c)
    # c = _wire_long()
    c.pprint_ports()
    c.show()

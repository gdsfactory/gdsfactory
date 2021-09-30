from typing import Optional

import gdsfactory as gf
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.straight_heater_metal import straight_heater_metal
from gdsfactory.types import ComponentFactory, ComponentOrFactory, CrossSectionFactory


@gf.cell
def bend_port(
    component: ComponentOrFactory = straight_heater_metal,
    port_name: str = "e1",
    port_name2: str = "e2",
    port_name1_bend: Optional[str] = None,
    port_name2_bend: Optional[str] = None,
    cross_section: CrossSectionFactory = gf.cross_section.metal3,
    bend: ComponentFactory = bend_circular,
    angle: int = 180,
    extension_length: Optional[float] = None,
    **kwargs,
):
    """
    Returns a component that contains a component with a bend and a straight

    Args:
        component:
        port_name: of the component
        port_name2: of the component, to extend to
        port_name1_bend:
        port_name2_bend:
        cross_section: for the bend
        bend: factory for the bend
        angle: for the bend
        extension_length: for the straight after the bend
        **kwargs: cross_section settings

    """
    c = gf.Component()
    component = component() if callable(component) else component
    c.component = component

    if port_name not in component.ports:
        raise ValueError(f"port_name {port_name} not in {list(component.ports.keys())}")

    extension_length = extension_length or abs(
        component.ports[port_name2].midpoint[0] - component.ports[port_name].midpoint[0]
    )

    ref = c << component
    b = c << bend(angle=angle, cross_section=cross_section, **kwargs)
    bend_ports = b.get_ports_list()

    port_name1_bend = port_name1_bend or bend_ports[0].name
    port_name2_bend = port_name2_bend or bend_ports[1].name

    b.connect(port_name1_bend, ref.ports[port_name])

    s = c << gf.c.straight(
        length=extension_length, cross_section=cross_section, **kwargs
    )
    straight_ports = s.get_ports_list()
    o2 = straight_ports[1].name
    o1 = straight_ports[0].name
    s.connect(o2, b.ports[port_name2_bend])

    c.add_ports(ref.get_ports_list())
    c.ports.pop(port_name)
    c.add_port(port_name, port=s.ports[o1])
    return c


if __name__ == "__main__":
    # c = gf.c.straight_pin()
    c = gf.c.straight_heater_metal()
    c = bend_port(component=c, port_name="e1")
    c.show()

import gdsfactory as gf
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.types import (
    ComponentFactory,
    ComponentOrFactory,
    CrossSectionFactory,
    PortName,
)


@gf.cell
def bend_port(
    component: ComponentOrFactory,
    port_name: PortName = 1,
    cross_section: CrossSectionFactory = gf.cross_section.metal3,
    bend: ComponentFactory = bend_circular,
    angle: int = 180,
    length: float = 1.0,
    **kwargs
):
    """
    Returns a component that contains a component with a bend and a straight

    Args:
        component:
        port_name: of the component
        cross_section: for the bend
        angle: for the bend
        length: for the straight after the bend
        **kwargs: cross_section settings

    """
    c = gf.Component()
    component = component() if callable(component) else component
    ref = c << component
    b = c << bend(angle=angle, cross_section=cross_section, **kwargs)
    b.connect(1, ref.ports[port_name])

    s = c << gf.c.straight(length=length, cross_section=cross_section)
    s.connect(2, b.ports[2])

    c.add_ports(ref.get_ports_list())
    c.ports.pop(port_name)
    c.add_port(port_name, port=s.ports[1])
    return c


if __name__ == "__main__":
    component = gf.c.straight_heater_metal()
    c = bend_port(component=component, length=component.settings["length"], port_name=1)
    c.show()

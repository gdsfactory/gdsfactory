import gdsfactory as gf
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.types import ComponentFactory, ComponentOrFactory, CrossSectionFactory


@gf.cell
def bend_port(
    component: ComponentOrFactory,
    port_name: str = "DC_0",
    cross_section: CrossSectionFactory = gf.cross_section.metal3,
    bend: ComponentFactory = bend_circular,
    angle: int = 180,
    length: float = 1.0,
    port_type: str = "dc",
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
        port_type: for the new port
        **kwargs: cross_section settings

    """
    c = gf.Component()
    component = component() if callable(component) else component
    ref = c << component
    b = c << bend(angle=angle, cross_section=cross_section, **kwargs)
    b.connect("W0", ref.ports[port_name])

    bend_ports = b.ports.copy()
    bend_ports.pop("W0")
    bend_port_name = list(bend_ports.keys())[0]

    s = c << gf.c.straight(length=length, cross_section=cross_section)
    s.connect("E0", b.ports[bend_port_name])

    ports = ref.ports.copy()
    ports.pop(port_name)
    c.add_ports(ports)
    c.add_port("W1", port=s.ports["W0"])
    c.ports["W1"].port_type = port_type
    gf.port.auto_rename_ports(c)
    return c


if __name__ == "__main__":
    component = gf.c.straight_heater_metal()
    c = bend_port(component=component, length=component.settings["length"])
    c.show()

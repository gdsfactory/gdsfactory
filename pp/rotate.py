from pp.component import Component
from pp.container import container
from pp.port import auto_rename_ports


@container
def rotate(
    component: Component, angle: int = 90, rename_ports: bool = True,
) -> Component:
    """Returns rotated component inside a container

    Args:
        component:
        angle: in degrees
        rename_ports: rename_ports_by_orientation
    """
    c = Component(f"{component.name}_r")
    cr = c.add_ref(component)
    cr.rotate(angle)
    c.ports = cr.ports
    if rename_ports:
        auto_rename_ports(c)
    return c


if __name__ == "__main__":
    import pp

    component = pp.c.mzi2x2()
    component_rotated = rotate(component, flatten=False)
    component_rotated.show()
    print(component_rotated)

    # component_rotated.pprint()
    # component_netlist = component.get_netlist()
    # component.pprint_netlist()

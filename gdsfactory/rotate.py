from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import auto_rename_ports


@cell
def rotate(
    component: Component,
    angle: int = 90,
    rename_ports: bool = True,
) -> Component:
    """Returns rotated component inside a new component.

    Most times you just need to place a reference and rotate it.
    This rotate function just encapsulates the rotated reference into a new component.

    Args:
        component:
        angle: in degrees
        rename_ports: renames ports by orientation (E,W,S,N)
    """
    c = Component()
    cr = c.add_ref(component)
    cr.rotate(angle)
    c.ports = cr.ports
    if rename_ports:
        auto_rename_ports(c)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    component = gf.components.mmi1x2()
    component_rotated = rotate(component=component)
    component_rotated.show()
    # print(component_rotated)

    # component_rotated.pprint()
    # component_netlist = component.get_netlist()
    # component.pprint_netlist()

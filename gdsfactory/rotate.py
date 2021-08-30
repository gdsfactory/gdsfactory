from gdsfactory.cell import cell
from gdsfactory.component import Component


@cell
def rotate(
    component: Component,
    angle: int = 90,
) -> Component:
    """Returns rotated component inside a new component.

    Most times you just need to place a reference and rotate it.
    This rotate function just encapsulates the rotated reference into a new component.

    Args:
        component:
        angle: in degrees
    """
    component_new = Component()
    component_new.component = component
    ref = component_new.add_ref(component)
    ref.rotate(angle)
    component_new.add_ports(ref.ports)
    return component_new


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.mmi1x2()
    cr = rotate(component=c)
    cr.show()
    # print(component_rotated)

    # component_rotated.pprint
    # component_netlist = component.get_netlist()
    # component.pprint_netlist()

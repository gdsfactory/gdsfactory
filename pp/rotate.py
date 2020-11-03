from pp.port import deco_rename_ports
from pp.container import container
from pp.component import Component


@container
@deco_rename_ports
def rotate(component: Component, angle: int = 90) -> Component:
    """ returns rotated component
    """
    c = Component(f"{component.name}_r")
    cr = c.add_ref(component)
    cr.rotate(angle)
    c.ports = cr.ports
    return c


if __name__ == "__main__":
    import pp

    c = pp.c.mmi1x2()
    cr = rotate(c)
    pp.show(cr)
    print(cr.ports)

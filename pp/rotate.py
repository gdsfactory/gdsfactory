from pp.port import deco_rename_ports
from pp.container import container
from pp.component import Component


@container
@deco_rename_ports
def rotate(component: Component, angle: int = 90) -> Component:
    """ returns rotated component
    """
    c = Component(
        settings=component.get_settings(),
        test_protocol=component.test_protocol,
        data_analysis_protocol=component.data_analysis_protocol,
    )
    cr = c.add_ref(component)
    cr.rotate(angle)
    c.ports = cr.ports
    c.name = component.name + "_r"
    return c


if __name__ == "__main__":
    import pp

    c = pp.c.mmi1x2()
    cr = rotate(c)
    pp.show(cr)
    print(cr.ports)

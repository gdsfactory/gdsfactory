import pathlib
import gdspy
import pp


def add_monitor(component, port, port_margin=1, layer=200, offset=(0, 0)):
    width = port.width + 2 * port_margin
    monitor = gdspy.Path(
        width, initial_point=port.midpoint + offset, number_of_paths=1, distance=0,
    )
    monitor.segment(length=0, layer=layer)
    component.add(monitor)


def add_monitors(component, port_margin=1):
    """ add port monitors to GDS detects ports by marker layer
    """
    if not isinstance(component, pp.Component):
        component = pp.load_component(component)

    for i, port in enumerate(component.ports.values()):
        if i == 0:
            add_monitor(
                component=component,
                port=port,
                port_margin=port_margin,
                layer=200,
                offset=(-0.2, 0),
            )
        add_monitor(
            component=component, port=port, port_margin=port_margin, layer=201 + i
        )

    return component


def extension_factory(length, width, layer=(1, 0)):
    return pp.c.hline(length=length, width=width, layer=layer)


def extend_ports(
    component,
    length=1,
    extension_factory=extension_factory,
    input_port_ext=None,
    output_port_ext=None,
):
    c = component
    port_list = list(c.ports.keys())

    dummy_ext = extension_factory(length=length, width=0.5)
    port_labels = list(dummy_ext.ports.keys())

    input_port_ext = input_port_ext or port_labels[0]
    output_port_ext = output_port_ext or port_labels[-1]

    for port_label in port_list:
        port = c.ports.pop(port_label)

        extension = c << extension_factory(
            length=length, width=port.width, layer=port.layer
        )
        extension.connect(input_port_ext, port)
        c.add_port(port_label, port=extension.ports[output_port_ext])

    return c


if __name__ == "__main__":
    gdspath = pathlib.Path.cwd() / "waveguide.gds"
    c = pp.c.waveguide(length=2)
    # cm = add_monitors(c)
    cm = extend_ports(c)
    pp.show(c)

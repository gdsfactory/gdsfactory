"""" add electrical labels to each device port

"""


import pp
from pp.ports.add_port_markers import get_input_label_electrical


def add_labels(component):
    c = pp.Component()
    electrical_ports = component.get_electrical_ports()
    c.add(component.ref())

    for i, port in enumerate(electrical_ports):
        label = get_input_label_electrical(port, i, component_name=component.name)
        c.add(label)

    return c


if __name__ == "__main__":
    from pp.components.electrical.pad import pad

    c = pad(width=10, height=10)
    print(c.ports)
    c2 = add_labels(c)
    pp.show(c2)

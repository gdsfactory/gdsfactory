import gdsfactory as gf
from gdsfactory.typings import LayerSpec, PortsOrList


def add_port_labels(
    component: gf.Component,
    ports: PortsOrList,
    layer: LayerSpec,
    texts: list[str] | None = None,
) -> gf.Component:
    """Add port labels to a component.

    Args:
        component: to add the labels.
        ports: list of ports to add the labels.
        layer: layer to add the labels.
        texts: text to add to the labels. Defaults to port names.
    """
    texts = texts or [port.name for port in ports]

    for text, port in zip(texts, ports):
        component.add_label(
            text=text,
            position=port.dcenter,
            layer=layer,
        )
    return component


if __name__ == "__main__":
    c = gf.Component()
    # ref = c << gf.components.straight()
    # ref = c << gf.routing.add_fiber_array(gf.components.straight())
    # ref = c << gf.routing.add_fiber_array(gf.components.nxn())
    ref = c << gf.routing.add_pads_top(gf.components.wire_straight())
    # c = add_port_labels(c, [ref.ports["o1"]], layer=(2, 0))
    # c = add_port_labels(c, ref.ports, layer=(2, 0))
    # c = label_farthest_right_port(c, ref.ports, layer=(2, 0), text="output")
    c.show()

import pathlib
from typing import List, Optional, Tuple

import numpy as np
from pydantic import validate_arguments

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.tech import LAYER


@validate_arguments
def _add_pin_square(
    component: Component,
    port: Port,
    pin_length: float = 0.1,
    layer: Tuple[int, int] = LAYER.PORT,
    label_layer: Optional[Tuple[int, int]] = LAYER.PORT,
    port_margin: float = 0.0,
) -> None:
    """Add half out pin to a component.

    Args:
        component:
        port: Port
        pin_length: length of the pin marker for the port
        layer: for the pin marker
        label_layer: for the label
        port_margin: margin to port edge


    .. code::

           _______________
          |               |
          |               |
          |               |
         |||              |
         |||              |
          |               |
          |      __       |
          |_______________|
                 __

    """
    p = port
    a = p.orientation
    ca = np.cos(a * np.pi / 180)
    sa = np.sin(a * np.pi / 180)
    rot_mat = np.array([[ca, -sa], [sa, ca]])

    d = p.width / 2 + port_margin

    dbot = np.array([pin_length / 2, -d])
    dtop = np.array([pin_length / 2, d])
    dbotin = np.array([-pin_length / 2, -d])
    dtopin = np.array([-pin_length / 2, +d])

    p0 = p.position + _rotate(dbot, rot_mat)
    p1 = p.position + _rotate(dtop, rot_mat)
    ptopin = p.position + _rotate(dtopin, rot_mat)
    pbotin = p.position + _rotate(dbotin, rot_mat)
    polygon = [p0, p1, ptopin, pbotin]
    component.add_polygon(polygon, layer=layer)

    if label_layer:
        component.add_label(
            text=str(p.name),
            position=p.midpoint,
            layer=label_layer,
        )


def _rotate(v, m):
    return np.dot(m, v)


def add_monitors_and_extend_ports(
    component: Component,
    port_labels: Optional[List[Port]] = None,
    extension_length: float = 0.2,
    port_margin: float = 1.0,
    layer: Tuple[int, int] = (101, 0),
) -> Component:
    """Add pins for port_labels
    each port increments the port layer

    Args:
        component: Component
        port_labels: list of labels
        extension_length: length to extend each port
        port_margin: margin from edge of the waveguide to edge of the port
        layer: monitor layer

    Returns: component with extended ports and monitors

    """

    port_labels = port_labels or component.ports.keys()

    for i, port_label in enumerate(port_labels):
        _add_pin_square(
            component=component,
            port=component.ports[port_label],
            pin_length=0,
            port_margin=port_margin,
            layer=(layer[0] + i, layer[1]),
        )

    return gf.components.extension.extend_ports(
        component=component, length=extension_length
    )


@cell
def add_monitors(
    component: Component,
    source_port_name: str = "o1",
    extension_length: float = 1.0,
    source_distance_to_monitors: float = 0.2,
    port_margin: float = 1.0,
    layer_monitor: Tuple[int, int] = (101, 0),
    layer_source: Tuple[int, int] = (110, 0),
    layer_simulation_region: Tuple[int, int] = (2, 0),
    top: float = 0.0,
    bottom: float = 0.0,
    right: float = 0.0,
    left: float = 0.0,
) -> Component:
    """Add monitors in layer_monitor, incrementing the layer number for each port
    Then extends all ports by source_distance_to_monitors
    Then add source in source_port_name
    Finally extends all ports by extension_length
    returns device centered at x=0, y=0

    Args:
        component: to add monitors
        source_port_name: name of the port to add the mode source
        extension_length: extension lenght when extending the ports
        port_margin: from waveguide edge to port edge
        layer_monitor: layer for the fist monitor
        layer_source: layer for the source
        layer_simulation_region: layer for the simulation region
        top: simulation region top padding
        bottom: simulation region south padding
        right: simulation region east padding
        left: simulation region west padding

    Returns: component with extended ports and monitors

    """
    c = gf.Component(f"{component.name}_monitors")

    # add monitors
    component_with_monitors = add_monitors_and_extend_ports(
        component=component.copy(),
        extension_length=source_distance_to_monitors,
        port_margin=port_margin,
        layer=layer_monitor,
    )

    # add source
    component_with_source = add_monitors_and_extend_ports(
        component=component_with_monitors.copy(),
        extension_length=extension_length,
        port_labels=[source_port_name],
        layer=layer_source,
    )

    # add simulation region
    component_with_padding = gf.add_padding_container(
        component=component_with_source,
        default=0,
        layers=[layer_simulation_region],
        top=top,
        bottom=bottom,
        right=right,
        left=left,
    )
    c.add(component_with_padding)
    c.ports = component_with_padding.ports
    return c


if __name__ == "__main__":
    gdspath = pathlib.Path.cwd() / "waveguide.gds"
    c = gf.components.bend_circular(radius=5)
    # c = gf.components.waveguide(length=2)
    # cm = extend_ports(component=c)
    cm = add_monitors(component=c)
    gf.show(cm)

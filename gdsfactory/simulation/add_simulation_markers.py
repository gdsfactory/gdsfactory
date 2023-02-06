"""Returns component with simulation markers."""
from __future__ import annotations

import warnings

import numpy as np

import gdsfactory as gf
from gdsfactory.add_pins import add_pin_rectangle
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerLevel
from gdsfactory.typings import ComponentSpec, Layer, LayerSpec


@gf.cell
def add_simulation_markers(
    component: ComponentSpec = bend_circular,
    port_margin: float = 3,
    port_source_name: str = "o1",
    layer_source: Layer = (110, 0),
    layer_monitor: Layer = (101, 0),
    layer_label: LayerSpec = "TEXT",
    port_source_offset: float = 0.2,
) -> Component:
    r"""Returns new Component with simulation markers.

    Args:
        component: component spec.
        port_margin: from port edge in um.
        port_source_name: for input.
        layer_source: for port marker.
        layer_monitor: for port marker.
        layer_label: for labeling the ports.
        port_source_offset: distance from source to monitor in um.

    .. code::

         top view
              ________________________________
             |                               |
             | xmargin_left                  | port_extension
             |<------>          port_margin ||<-->
          ___|___________          _________||___
             |           \        /          |
             |            \      /           |
             |             ======            |
             |            /      \           |
          ___|___________/        \__________|___
             |   |                 <-------->|
             |   |ymargin_bot   xmargin_right|
             |   |                           |
             |___|___________________________|

        side view
              ________________________________
             |                     |         |
             |                     |         |
             |                   zmargin_top |
             |ymargin              |         |
             |<---> _____         _|___      |
             |     |     |       |     |     |
             |     |     |       |     |     |
             |     |_____|       |_____|     |
             |       |                       |
             |       |                       |
             |       |zmargin_bot            |
             |       |                       |
             |_______|_______________________|



    .. plot::
        :include-source:

        import gdsfactory as gf
        from gdsfactory.simulation.add_simulation_markers import add_simulation_markers

        c = gf.components.bend_circular()
        c = add_simulation_markers(c)
        c.plot()
    """
    c = gf.Component()
    component = gf.get_component(component)

    ref = c << component
    port_names = list(ref.ports.keys())

    layer_stack = get_layer_stack()

    if port_source_name not in port_names:
        warnings.warn(f"port_source_name={port_source_name!r} not in {port_names}")
        port_source = ref.get_ports_list()[0]
        port_source_name = port_source.name
        warnings.warn(f"Selecting port_source_name={port_source_name!r} instead.")

    assert isinstance(
        component, Component
    ), f"component needs to be a gf.Component, got Type {type(component)}"

    # Add port monitors
    for port_name in ref.ports.keys():
        port = ref.ports[port_name]
        add_pin_rectangle(c, port=port, port_margin=port_margin, layer=layer_monitor)
        layer_stack.layers["monitor"] = LayerLevel(
            layer=layer_monitor, thickness=2 * port_margin, zmin=-port_margin
        )

    # Add source
    port = ref.ports[port_source_name]
    angle_rad = np.radians(port.orientation)

    port = port.move_polar_copy(angle=angle_rad, d=port_source_offset)
    add_pin_rectangle(c, port=port, port_margin=port_margin, layer=layer_source)

    layer_stack.layers["source"] = LayerLevel(
        layer=layer_source, thickness=2 * port_margin, zmin=-port_margin
    )

    c.add_ports(component.ports)
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    # c = gf.components.coupler_ring()
    c = gf.components.mmi1x2()
    c = add_simulation_markers(c)
    c.show()
    scene = c.to_3d()
    scene.show()

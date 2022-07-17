"""Returns simulation with markers."""
import warnings

import gdsfactory as gf
from gdsfactory.add_pins import add_pin_rectangle
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.types import ComponentSpec, Layer, LayerSpec


@gf.cell
def add_simulation_markers(
    component: ComponentSpec = bend_circular,
    port_margin: float = 3,
    port_source_name: str = "o1",
    layer_source: Layer = (110, 0),
    layer_monitor: Layer = (101, 0),
    layer_label: LayerSpec = "TEXT",
) -> Component:
    r"""Returns new Component with simulation markers from gdsfactory Component.

    Args:
        component: component spec.
        port_margin: from port edge in um.
        port_source_name: for input.
        layer_source: for port marker.
        layer_monitor: for port marker.
        layer_label: LayerSpec = "TEXT",

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
    ref.x = 0
    ref.y = 0
    port_names = list(ref.ports.keys())

    if port_source_name not in port_names:
        warnings.warn(f"port_source_name={port_source_name!r} not in {port_names}")
        port_source = ref.get_ports_list()[0]
        port_source_name = port_source.name
        warnings.warn(f"Selecting port_source_name={port_source_name!r} instead.")

    assert isinstance(
        component, Component
    ), f"component needs to be a gf.Component, got Type {type(component)}"

    # Add source
    port = ref.ports[port_source_name]
    add_pin_rectangle(c, port=port, port_margin=port_margin, layer=layer_source)

    # Add port monitors
    for port_name in ref.ports.keys():
        port = ref.ports[port_name]
        add_pin_rectangle(c, port=port, port_margin=port_margin, layer=layer_monitor)
    c.copy_child_info(component)
    c.add_ports(component.ports)
    return c


if __name__ == "__main__":
    c = gf.components.taper_sc_nc()
    c = add_simulation_markers(c)
    # c.show()
    scene = c.to_3d()
    scene.show()

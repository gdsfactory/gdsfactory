from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec


@gf.cell
def splitter_chain(
    splitter: ComponentSpec = "mmi1x2",
    columns: int = 3,
    bend: ComponentSpec = "bend_s",
) -> Component:
    """Chain of splitters.

    Args:
        splitter: splitter to chain.
        columns: number of splitters to chain.
        bend: bend to connect splitters.

    .. code::

                 __o5
              __|
           __|  |__o4
      o1 _|  |__o3
          |__o2

           __o2
      o1 _|
          |__o3
    """
    c = gf.Component()
    splitter_component = gf.get_component(splitter)
    cref = c.add_ref(splitter_component)

    splitter_ports_east = list(cref.ports.filter(port_type="optical", orientation=0))
    e1_port_name = splitter_ports_east[0].name
    e0_port_name = splitter_ports_east[1].name

    bend = gf.get_component(bend)
    c.add_port(name="o1", port=cref.ports["o1"])
    c.add_port(name="o2", port=cref.ports[e0_port_name])

    for i in range(1, columns):
        bref = c.add_ref(bend)
        bref.connect(port="o1", other=cref.ports[e1_port_name])
        cref = c.add_ref(splitter_component)
        cref.connect(port="o1", other=bref.ports["o2"])
        c.add_port(name=f"o{i + 2}", port=cref.ports[e0_port_name])

    c.add_port(name=f"o{i + 3}", port=cref.ports[e1_port_name])
    c.copy_child_info(splitter_component)
    return c


if __name__ == "__main__":
    # component = splitter_chain(splitter=gf.components.mmi1x2, columns=4)
    component = splitter_chain()
    component.show()

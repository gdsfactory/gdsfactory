import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_s import bend_s
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.types import ComponentFactory


@gf.cell
def splitter_chain(
    splitter: ComponentFactory = mmi1x2,
    n_devices: int = 3,
    bend: ComponentFactory = bend_s,
    **kwargs,
) -> Component:
    """Chain of splitters

    .. code::

                __5
             __|
          __|  |__4
      1 _|  |__3
         |__2


          __E1
      1 _|
         |__E0

    """
    c = gf.Component()
    splitter_component = gf.call_if_func(splitter, **kwargs)
    cref = c.add_ref(splitter_component)

    splitter_ports_east = cref.get_ports_list(port_type="optical", orientation=0)
    e1_port_name = splitter_ports_east[0].name
    e0_port_name = splitter_ports_east[1].name

    bend = bend() if callable(bend) else bend
    c.add_port(name="o1", port=cref.ports["o1"])
    c.add_port(name="o2", port=cref.ports[e0_port_name])

    for i in range(1, n_devices):
        bref = c.add_ref(bend)
        bref.connect(port="o1", destination=cref.ports[e1_port_name])

        cref = c.add_ref(splitter_component)

        cref.connect(port="o1", destination=bref.ports["o2"])
        c.add_port(name=f"o{i+2}", port=cref.ports[e0_port_name])

    c.add_port(name=f"o{i+3}", port=cref.ports[e1_port_name])
    c.settings["component"] = splitter_component.get_settings()
    return c


if __name__ == "__main__":
    # component = splitter_chain(splitter=gf.components.mmi1x2, n_devices=4)
    component = splitter_chain()
    component.show()
    component.pprint

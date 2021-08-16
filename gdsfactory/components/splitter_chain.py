import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.types import ComponentFactory


@gf.cell
def splitter_chain(
    splitter: ComponentFactory = mmi1x2, n_devices: int = 3, **kwargs
) -> Component:
    """Chain of splitters

    .. code::

                __
             __|
          __|  |__
        _|  |__
         |__


          __E1
        _|
         |__E0

    """
    c = gf.Component()
    splitter_component = gf.call_if_func(splitter, **kwargs)
    cref = c.add_ref(splitter_component)
    e1_port_name = len(splitter_component.ports) - 1
    e0_port_name = len(splitter_component.ports)

    bend = gf.components.bezier()
    c.add_port(name=1, port=cref.ports[1])
    c.add_port(name=2, port=cref.ports[e0_port_name])

    for i in range(1, n_devices):
        bref = c.add_ref(bend)
        bref.connect(port=1, destination=cref.ports[e1_port_name])

        cref = c.add_ref(splitter_component)

        cref.connect(port=1, destination=bref.ports[2])
        c.add_port(name=i + 2, port=cref.ports[e0_port_name])

    c.add_port(name=i + 3, port=cref.ports[e1_port_name])
    c.settings["component"] = splitter_component.get_settings()
    return c


if __name__ == "__main__":
    component = splitter_chain(splitter=gf.components.mmi1x2, n_devices=4)
    component.show()
    component.pprint()

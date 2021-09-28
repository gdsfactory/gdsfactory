from typing import List, Optional

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.types import ComponentOrFactory


@cell
def extend_ports_list(
    ports: List[Port],
    extension_factory: ComponentOrFactory,
    extension_port_name: Optional[str] = None,
) -> Component:
    """Returns a component with the extensions for a list of ports.

    Args:
        ports: list of ports
        extension_factory: function for extension
        extension_port_name: to connect extension

    """
    c = Component()
    extension = (
        extension_factory() if callable(extension_factory) else extension_factory
    )

    extension_port_name = extension_port_name or list(extension.ports.keys())[0]

    for i, port in enumerate(ports):
        extension_ref = c << extension
        extension_ref.connect(extension_port_name, port)

        for port_name, port in extension_ref.ports.items():
            # if port_name not in extension_port_name:
            c.add_port(f"{i}_{port_name}", port=port)

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.mmi1x2()
    t = gf.partial(gf.c.taper, width2=0.1)

    cr = extend_ports_list(
        ports=c.get_ports_list(), extension_factory=t, extension_port_name="o1"
    )
    c.add_ref(cr)
    c.show()

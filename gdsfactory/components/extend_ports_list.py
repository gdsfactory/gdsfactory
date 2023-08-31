from __future__ import annotations

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.typings import ComponentSpec, Strs


@cell
def extend_ports_list(
    ports: list[Port],
    extension: ComponentSpec,
    extension_port_name: str | None = None,
    ignore_ports: Strs | None = None,
) -> Component:
    """Returns a component with the extensions for a list of ports.

    Args:
        ports: list of ports.
        extension: function for extension.
        extension_port_name: to connect extension.
        ignore_ports: list of port names to ignore.
    """
    from gdsfactory.pdk import get_component

    c = Component()
    extension = get_component(extension)

    extension_port_name = extension_port_name or list(extension.ports.keys())[0]
    ignore_ports = ignore_ports or ()

    for i, port in enumerate(ports):
        extension_ref = c << extension
        extension_ref.connect(extension_port_name, port)

        for port_name, port in extension_ref.ports.items():
            if port_name not in ignore_ports:
                c.add_port(f"{i}_{port_name}", port=port)

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component("taper_extended")
    c0 = gf.components.taper(width2=10)
    e = extend_ports_list(c0.get_ports_list(), extension="straight")
    c << c0
    c << e
    c.show(show_ports=True)

    # c = gf.Component("mmi_extended")
    # m = gf.components.mmi1x2()
    # t = partial(gf.components.taper, width2=0.1)
    # e = extend_ports_list(
    #     ports=m.get_ports_list(), extension=t, extension_port_name="o1"
    # )

    # c << m
    # c << e
    # c.show(show_ports=True)

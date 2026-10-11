from __future__ import annotations

__all__ = ["extend_ports"]

import warnings
from typing import Any, cast

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.component_functions._get_component import get_component
from gdsfactory.port import Port
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, PortNames


def extend_ports(
    component: ComponentSpec = "mmi1x2",
    port_names: PortNames | None = None,
    length: float = 5.0,
    extension: ComponentSpec | None = None,
    port1: str | None = None,
    port2: str | None = None,
    port_type: str = "optical",
    centered: bool = False,
    cross_section: CrossSectionSpec | None = None,
    extension_port_names: list[str] | None = None,
    allow_width_mismatch: bool = False,
    auto_taper: bool = True,
    **kwargs: Any,
) -> Component:
    """Returns a new component with some ports extended.

    You can define extension Spec
    defaults to port cross_section of each port to extend.

    Args:
        component: component to extend ports.
        port_names: list of ports names to extend, if None it extends all ports.
        length: extension length, added after any automatic taper, so the total
            extension is taper_length + length when a taper is inserted.
        extension: function to extend ports (defaults to a straight).
        port1: extension input port name.
        port2: extension output port name.
        port_type: type of the ports to extend.
        centered: if True centers rectangle at (0, 0).
        cross_section: extension cross_section, defaults to port cross_section
            if port has no cross_section it creates one using width and layer.
        extension_port_names: extension port names add to the new component.
        allow_width_mismatch: allow width mismatches.
        auto_taper: if True adds automatic tapers.
        kwargs: cross_section settings.

    Keyword Args:
        layer: port GDS layer.
        prefix: port name prefix.
        orientation: in degrees.
        width: port width.
        layers_excluded: List of layers to exclude.
        port_type: optical, electrical, ....
        clockwise: if True, sort ports clockwise, False: counter-clockwise.
    """
    c = gf.Component()
    component = get_component(component)

    cref = c << component
    if centered:
        cref.x = 0
        cref.y = 0

    ports_all = cref.ports
    ports_all_names = [p.name for p in ports_all if p.name is not None]

    ports_to_extend = gf.port.get_ports_list(ports_all, port_type=port_type, **kwargs)
    ports_to_extend_names = [p.name for p in ports_to_extend if p.name is not None]
    ports_to_extend_names = cast("list[str]", port_names or ports_to_extend_names)

    ports_to_connect: dict[str, Port] = {}
    for port_name_to_extend in ports_to_extend_names:
        if port_name_to_extend in ports_all_names:
            ports_to_connect[port_name_to_extend] = ports_all[port_name_to_extend]
        else:
            warnings.warn(
                f"Port Name {port_name_to_extend!r} not in {ports_all_names}",
                stacklevel=3,
                category=UserWarning,
            )

    if auto_taper and cross_section:
        from gdsfactory.routing.auto_taper import add_auto_tapers

        tapered = add_auto_tapers(
            component=c,
            ports=list(ports_to_connect.values()),
            cross_section=cross_section,
        )
        ports_to_connect = dict(zip(ports_to_connect, tapered, strict=True))

    for port in ports_all:
        port_name = port.name

        if port_name in ports_to_connect:
            if extension:
                extension_component = get_component(extension)
            else:
                extension_component = get_component(
                    "straight",
                    length=length,
                    cross_section=cross_section
                    or ports_to_connect[port_name].cross_section,
                )
            port_labels = [p.name for p in extension_component.ports]
            port1 = port1 or port_labels[0]
            port2 = port2 or port_labels[-1]

            assert port1 is not None

            extension_ref = c << extension_component
            extension_ref.connect(
                port1,
                ports_to_connect[port_name],
                allow_width_mismatch=allow_width_mismatch,
                mirror=isinstance(
                    ports_to_connect[port_name].cross_section, gf.AsymmetricCrossSection
                ),
                use_mirror=True,
            )
            c.add_port(port_name, port=extension_ref.ports[port2])
            extension_port_names = extension_port_names or []
            [
                c.add_port(name, port=extension_ref.ports[name])
                for name in extension_port_names
            ]
        else:
            c.add_port(port_name, port=port)

    c.copy_child_info(component)
    return c

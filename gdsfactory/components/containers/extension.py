from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import cross_section as cross_section_function
from gdsfactory.port import Port
from gdsfactory.typings import ComponentSpec, Coordinate, CrossSectionSpec, PortNames

DEG2RAD = np.pi / 180


def line(
    p_start: Port | Coordinate,
    p_end: Port | Coordinate,
    width: float | None = None,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    if isinstance(p_start, gf.Port):
        width = p_start.width
        p_start = p_start.center

    if isinstance(p_end, gf.Port):
        p_end = p_end.center

    w = width
    assert w is not None
    angle = np.arctan2(p_end[1] - p_start[1], p_end[0] - p_start[0])
    a = np.pi / 2
    p0 = move_polar_rad_copy(p_start, angle + a, w / 2)
    p1 = move_polar_rad_copy(p_start, angle - a, w / 2)
    p2 = move_polar_rad_copy(p_end, angle - a, w / 2)
    p3 = move_polar_rad_copy(p_end, angle + a, w / 2)
    return p0, p1, p2, p3


def move_polar_rad_copy(
    pos: Coordinate, angle: float, length: float
) -> npt.NDArray[np.floating[Any]]:
    """Returns the points of a position (pos) with angle, shifted by length.

    Args:
        pos: position.
        angle: in radians.
        length: extension length in um.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return pos + length * np.array([c, s])


@gf.cell
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
        length: extension length.
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
    component = gf.get_component(component)

    cref = c << component
    if centered:
        cref.dx = 0
        cref.dy = 0

    ports_all = cref.ports
    port_names_all = [p.name for p in ports_all if p.name is not None]

    ports_to_extend = list(
        gf.port.get_ports_list(cref.ports, port_type=port_type, **kwargs)
    )
    ports_to_extend_names = [p.name for p in ports_to_extend if p.name is not None]
    ports_to_extend_names = cast(list[str], port_names or ports_to_extend_names)

    if auto_taper and cross_section:
        from gdsfactory.routing.auto_taper import add_auto_tapers

        ports_to_extend = add_auto_tapers(
            component=c, ports=ports_to_extend, cross_section=cross_section
        )

    for port_name_to_extend in ports_to_extend_names:
        if port_name_to_extend not in port_names_all:
            warnings.warn(
                f"Port Name {port_name_to_extend!r} not in {port_names_all}",
                stacklevel=3,
                category=UserWarning,
            )

    for port in ports_all:
        port_name = port.name
        port = cref.ports[port_name]

        if port_name in ports_to_extend_names:
            if extension:
                extension_component = gf.get_component(extension)
            else:
                pdk = gf.get_active_pdk()
                cross_section_names = list(pdk.cross_sections)
                port_xs_name = port.info.get("cross_section", None)

                if port_xs_name and port_xs_name in cross_section_names:
                    cross_section_extension: CrossSectionSpec = gf.get_cross_section(
                        port.info["cross_section"]
                    )

                else:
                    cross_section_extension = cross_section or cross_section_function(
                        layer=gf.get_layer_tuple(port.layer),
                        width=port.width,
                    )

                extension_component = gf.components.straight(
                    length=length,
                    cross_section=cross_section_extension,
                )
            port_labels = [p.name for p in extension_component.ports]
            port1 = port1 or port_labels[0]
            port2 = port2 or port_labels[-1]

            assert port1 is not None

            extension_ref = c << extension_component
            extension_ref.connect(
                port1, port, allow_width_mismatch=allow_width_mismatch
            )
            c.add_port(port_name, port=extension_ref.ports[port2])
            extension_port_names = extension_port_names or []
            [
                c.add_port(name, port=extension_ref.ports[name])
                for name in extension_port_names
            ]
        else:
            c.add_port(port_name, port=component.ports[port_name])

    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    # test_extend_ports()
    # c0 = gf.c.straight(width=5)
    # t = gf.components.taper(length=10, width1=5, width2=0.5)
    # p0 = c0["o1"]
    # c = extend_ports(c0, extension=t)
    # c = extend_ports()
    c = gf.c.mmi1x2(cross_section="rib")
    c = extend_ports(component=c)
    c.pprint_ports()
    c.show()

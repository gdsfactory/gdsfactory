"""This module contains functions to automatically add tapers to a component's ports, and to create tapers between different cross sections."""

from __future__ import annotations

import warnings

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, Port, Ports


def add_auto_tapers(
    component: gf.Component,
    ports: Ports,
    cross_section: CrossSectionSpec,
) -> list[Port]:
    """Adds tapers to the ports of a component (to be used for routing) and returns the new lists of ports.

    Args:
        component: the component to add tapers to
        ports: the list of ports
        cross_section: the cross section to route to

    Returns:
        The new list of ports, on the opposite end of the tapers
    """
    return [auto_taper_to_cross_section(component, p, cross_section) for p in ports]


def auto_taper_to_cross_section(
    component: gf.Component, port: Port, cross_section: CrossSectionSpec
) -> Port:
    """Creates a taper from a port to a given cross section and places it in the component. The opposite port of the taper will be returned.

    Args:
        component: the component to place into
        port: a port to connect to, usually from a ComponentReference
        cross_section: a cross section to transition to

    Returns:
        The port at the opposite (unconnected end) of the taper.
    """
    port_layer = gf.get_layer(port.layer)
    port_width = port.dwidth
    cross_section = gf.get_cross_section(cross_section)
    cs_layer = gf.get_layer(cross_section.layer)
    cs_width = cross_section.width
    pdk = gf.get_active_pdk()

    if port_layer != cs_layer:
        try:
            taper = pdk.layer_transitions[port_layer, cs_layer]
        except KeyError as e:
            raise KeyError(
                f"No registered tapers between routing layers {gf.get_layer_name(port_layer)!r} and {gf.get_layer_name(cs_layer)!r}!"
            ) from e
    elif port_width != cs_width:
        try:
            taper = pdk.layer_transitions[port_layer]
        except KeyError:
            warnings.warn(
                f"No registered width taper for layer {port_layer}. Skipping."
            )
            return port
    else:
        return port
    taper_component = gf.get_component(taper, width1=port_width, width2=cs_width)
    taper_ports = [p for p in taper_component.ports if p.port_type == port.port_type]

    if len(taper_ports) != 2:
        raise ValueError(
            f"Taper component should have two ports of port_type={port.port_type!r}! Got {taper_component.ports}."
        )
    if taper_ports[0].layer == port.layer and taper_ports[0].width == port.width:
        p0, p1 = taper_ports
    elif taper_ports[1].layer == port.layer and taper_ports[1].width == port.width:
        p1, p0 = taper_ports
    else:
        width = port.dwidth
        layer = port.layer
        raise ValueError(
            f"Taper component ports do not match the port's layer and width! Got {taper_ports}. and {layer}, {width}"
        )
    taper_ref = component.add_ref(taper_component)
    assert p0.name is not None
    taper_ref.connect(p0.name, port)
    port_new = taper_ref.ports[p1.name].copy()
    port_new.name = port.name
    return port_new

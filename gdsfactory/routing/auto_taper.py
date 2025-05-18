"""This module contains functions to automatically add tapers to a component's ports, and to create tapers between different cross sections."""

from __future__ import annotations

import warnings

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.typings import CrossSectionSpec, LayerTransitions, Ports


def add_auto_tapers(
    component: Component,
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
    # Prefetch cross_section object and relevant properties just once
    cross_section_obj = gf.get_cross_section(cross_section)
    cs_layer = gf.get_layer(cross_section_obj.layer)
    cs_width = cross_section_obj.width

    # Prefetch active pdk and layer_transitions, for efficiency (if needed)
    pdk = gf.get_active_pdk()
    default_layer_transitions = getattr(pdk, "layer_transitions", None)

    # Cache all port.layer -> port_layer lookups (reduces redundant gf.get_layer calls)
    port_layer_cache = {}
    for port in ports:
        layer = port.layer
        if layer not in port_layer_cache:
            port_layer_cache[layer] = gf.get_layer(layer)

    # Call the helper function, passing cached objects as parameters
    return [
        auto_taper_to_cross_section_fast(
            component,
            port,
            cross_section_obj,
            cs_layer,
            cs_width,
            port_layer_cache[port.layer],
            default_layer_transitions,
        )
        for port in ports
    ]


def auto_taper_to_cross_section(
    component: gf.Component,
    port: Port,
    cross_section: CrossSectionSpec,
    layer_transitions: LayerTransitions | None = None,
) -> Port:
    """Creates a taper from a port to a given cross section and places it in the component. The opposite port of the taper will be returned.

    Args:
        component: the component to place into
        port: a port to connect to, usually from a ComponentReference
        cross_section: a cross section to transition to
        layer_transitions: the layer transitions dictionary to use (use the pdk default if None)

    Returns:
        The port at the opposite (unconnected end) of the taper.
    """
    port_layer = gf.get_layer(port.layer)
    port_width = port.width
    cross_section = gf.get_cross_section(cross_section)
    cs_layer = gf.get_layer(cross_section.layer)
    cs_width = cross_section.width
    if layer_transitions is None:
        pdk = gf.get_active_pdk()
        layer_transitions = pdk.layer_transitions
    reverse = False
    if port_layer != cs_layer:
        try:
            taper = layer_transitions.get((port_layer, cs_layer))
            if taper is None:
                taper = layer_transitions[cs_layer, port_layer]
                reverse = True
        except KeyError as e:
            raise KeyError(
                f"No registered tapers between routing layers {gf.get_layer_name(port_layer)!r} and {gf.get_layer_name(cs_layer)!r}!"
            ) from e
    elif port_width != cs_width:
        try:
            taper = layer_transitions[port_layer]
        except KeyError:
            warnings.warn(
                f"No registered width taper for layer {port_layer}. Skipping.",
                stacklevel=4,
            )
            return port
    else:
        return port
    if reverse:
        taper_component = gf.get_component(taper, width2=port_width, width1=cs_width)
    else:
        taper_component = gf.get_component(taper, width1=port_width, width2=cs_width)

    # ensure we filter out i.e. extra electrical ports if we should be looking at optical ones
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
        width = port.width
        layer = port.layer
        raise ValueError(
            f"Taper component ports do not match the port's layer and width!\nTaper name: {taper_component.name}\nPorts: {taper_ports}\nCross-section: {layer=}, {width=}"
        )
    taper_ref = component.add_ref(taper_component)
    assert p0.name is not None
    taper_ref.connect(p0.name, port)
    port_new = taper_ref.ports[p1.name].copy()
    port_new.name = port.name
    return port_new


def auto_taper_to_cross_section_fast(
    component: Component,
    port: Port,
    cross_section,  # already a CrossSection object
    cs_layer,  # already an int or LayerEnum, computed once for the cs in add_auto_tapers
    cs_width,  # already fetched
    port_layer,  # previously fetched for this port.layer
    default_layer_transitions,  # previously fetched
    layer_transitions: LayerTransitions | None = None,
) -> Port:
    """Creates a taper from a port to a given cross section and places it in the component. The opposite port of the taper will be returned.

    Args:
        component: the component to place into
        port: a port to connect to, usually from a ComponentReference
        cross_section: a cross section to transition to (already resolved)
        cs_layer, cs_width: precomputed to avoid repeated lookups
        port_layer: precomputed gf.get_layer(port.layer)
        default_layer_transitions: pre-fetched from pdk
        layer_transitions: optionally override default transition dictionary

    Returns:
        The port at the opposite (unconnected end) of the taper.
    """
    port_width = port.width
    # Layer transitions default logic, as before
    lt = (
        layer_transitions
        if layer_transitions is not None
        else default_layer_transitions
    )
    reverse = False
    if port_layer != cs_layer:
        try:
            taper = lt.get((port_layer, cs_layer))
            if taper is None:
                taper = lt[cs_layer, port_layer]
                reverse = True
        except KeyError as e:
            raise KeyError(
                f"No registered tapers between routing layers {gf.get_layer_name(port_layer)!r} and {gf.get_layer_name(cs_layer)!r}!"
            ) from e
    elif port_width != cs_width:
        try:
            taper = lt[port_layer]
        except KeyError:
            warnings.warn(
                f"No registered width taper for layer {port_layer}. Skipping.",
                stacklevel=4,
            )
            return port
    else:
        return port

    if reverse:
        taper_component = gf.get_component(taper, width2=port_width, width1=cs_width)
    else:
        taper_component = gf.get_component(taper, width1=port_width, width2=cs_width)

    # Filter ports: only include those with the same port_type
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
        width = port.width
        layer = port.layer
        raise ValueError(
            f"Taper component ports do not match the port's layer and width!\nTaper name: {taper_component.name}\nPorts: {taper_ports}\nCross-section: {layer=}, {width=}"
        )
    taper_ref = component.add_ref(taper_component)
    assert p0.name is not None
    taper_ref.connect(p0.name, port)
    port_new = taper_ref.ports[p1.name].copy()
    port_new.name = port.name
    return port_new

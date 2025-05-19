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
    cross_section_obj = gf.get_cross_section(cross_section)
    cs_layer = gf.get_layer(cross_section_obj.layer)
    cs_width = cross_section_obj.width

    layer_transitions = None
    _pdk = None

    def get_layer_transitions() -> LayerTransitions:
        nonlocal layer_transitions, _pdk
        if layer_transitions is None:
            if _pdk is None:
                _pdk = gf.get_active_pdk()
            layer_transitions = _pdk.layer_transitions
        return layer_transitions

    result = []
    get_port_layer = gf.get_layer

    for p in ports:
        port_layer = get_port_layer(p.layer)
        port_width = p.width
        lt = layer_transitions
        if lt is None:
            lt = get_layer_transitions()
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
                result.append(p)
                continue
        else:
            result.append(p)
            continue

        if reverse:
            taper_component = gf.get_component(
                taper, width2=port_width, width1=cs_width
            )
        else:
            taper_component = gf.get_component(
                taper, width1=port_width, width2=cs_width
            )

        # Ensure we filter out i.e. extra electrical ports if we should be looking at optical ones
        taper_ports = [
            tp for tp in taper_component.ports if tp.port_type == p.port_type
        ]

        if len(taper_ports) != 2:
            raise ValueError(
                f"Taper component should have two ports of port_type={p.port_type!r}! Got {taper_component.ports}."
            )
        if taper_ports[0].layer == p.layer and taper_ports[0].width == p.width:
            p0, p1 = taper_ports
        elif taper_ports[1].layer == p.layer and taper_ports[1].width == p.width:
            p1, p0 = taper_ports
        else:
            width = p.width
            layer = p.layer
            raise ValueError(
                f"Taper component ports do not match the port's layer and width!\nTaper name: {taper_component.name}\nPorts: {taper_ports}\nCross-section: {layer=}, {width=}"
            )
        taper_ref = component.add_ref(taper_component)
        assert p0.name is not None
        taper_ref.connect(p0.name, p)
        port_new = taper_ref.ports[p1.name].copy()
        port_new.name = p.name
        result.append(port_new)
    return result


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
    # Reuse fast version above for legacy compatibility
    # (not used in new add_auto_tapers flow)
    port_layer = gf.get_layer(port.layer)
    port_width = port.width
    cross_section_obj = gf.get_cross_section(cross_section)
    cs_layer = gf.get_layer(cross_section_obj.layer)
    cs_width = cross_section_obj.width
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

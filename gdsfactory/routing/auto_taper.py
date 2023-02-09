import warnings
from typing import Optional, List

from gdsfactory.component import Port, ComponentReference, Component
from gdsfactory.typings import CrossSectionSpec


def taper_to_cross_section(
    port: Port, cross_section: CrossSectionSpec
) -> Optional[ComponentReference]:
    """Returns taper ComponentReference from a port to a given cross-section \
            placed so that it connects to the input port.

    Assumes that the taper component has `width1` and `width2` which map to the input and output port widths.

    Args:
        port: a port to connect to, usually from a ComponentReference
        cross_section: a cross-section to transition to

    .. plot::
        :include-source:

        from gdsfactory.routing.auto_taper import taper_to_cross_section
        from gdsfactory.cross_section import strip
        import gdsfactory as gf

        c = gf.Component()

        # create a component reference to connect to
        wg = c << gf.components.straight()

        # create a taper reference transitioning to strip from the rib waveguide
        taper = taper_to_cross_section(wg.ports['o1'], strip(width=2.0))

        # add the taper reference to the parent component
        c.add(taper)
        c.plot()
    """
    from gdsfactory.pdk import (
        get_cross_section,
        get_active_pdk,
        get_layer,
        get_component,
    )

    port_layer = get_layer(port.layer)
    port_width = port.width
    cross_section = get_cross_section(cross_section)
    cs_layer = get_layer(cross_section.layer)
    cs_width = cross_section.width
    layer_transitions = get_active_pdk().layer_transitions

    if port_layer != cs_layer:
        try:
            taper_name = layer_transitions[(port_layer, cs_layer)]
        except KeyError as e:
            raise KeyError(
                f"No registered tapers between routing layers {port_layer} and {cs_layer}!"
            ) from e
    elif abs(port_width - cs_width) > 0.001:
        try:
            taper_name = layer_transitions[port_layer]
        except KeyError:
            warnings.warn(
                f"No registered width taper for layer {port_layer}. Skipping."
            )
            return None
    else:
        return None
    taper = get_component(taper_name, width1=port_width, width2=cs_width)
    input_port_name = _get_taper_io_port_names(component=taper)[0]
    return ComponentReference(taper).connect(input_port_name, port)


def _get_taper_io_port_names(component: Component) -> List[str]:
    # this is kind of a hack, but o1 < o2, in0 < out0... hopefully nobody has any other wacky conventions!
    return sorted(component.ports.keys())


def _auto_taper(
    routing_func,
    ports1,
    ports2,
    cross_section: CrossSectionSpec,
    auto_taper=True,
    **kwargs,
):
    from gdsfactory.pdk import get_cross_section

    if not auto_taper:
        return routing_func(ports1, ports2, cross_section=cross_section, **kwargs)
    tapers = []
    inner_ports = []
    ports_original = [ports1, ports2]
    x = get_cross_section(cross_section)

    for port_group in ports_original:
        new_port_group = []
        taper_group = []
        for port in port_group:
            taper = taper_to_cross_section(port, x)
            taper_group.append(taper)
            if taper is None:
                new_port_group.append(port)
            else:
                new_port_group.append(taper.ports["out0"])
        inner_ports.append(new_port_group)
        tapers.append(taper_group)
    routes = routing_func(
        inner_ports[0], inner_ports[1], cross_section=cross_section, **kwargs
    )
    for route, port1, port2, taper1, taper2 in zip(
        routes, ports1, ports2, tapers[0], tapers[1]
    ):
        if taper1:
            route.references.append(taper1)
        if taper2:
            route.references.append(taper2)
        route.ports = (port1, port2)
    return routes

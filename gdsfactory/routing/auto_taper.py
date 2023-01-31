###################################################################################################################
# PROPRIETARY AND CONFIDENTIAL
# THIS SOFTWARE IS THE SOLE PROPERTY AND COPYRIGHT (c) 2022 OF ROCKLEY PHOTONICS LTD.
# USE OR REPRODUCTION IN PART OR AS A WHOLE WITHOUT THE WRITTEN AGREEMENT OF ROCKLEY PHOTONICS LTD IS PROHIBITED.
# RPLTD NOTICE VERSION: 1.1.1
###################################################################################################################
import warnings
from typing import Optional

from gdsfactory.component import Port, ComponentReference
from gdsfactory.pdk import get_cross_section, get_active_pdk, get_layer, get_component
from gdsfactory.types import CrossSectionSpec


def taper_to_cross_section(
    port: Port, cross_section: CrossSectionSpec
) -> Optional[ComponentReference]:
    """
    Creates a taper from a port to a given cross section. Port 'in0' of the taper will be connected to the input port
    and 'out0' will be exposed as a port that can be connected to by a device with the given cross section.

    :param port: a port to connect to, usually from a ComponentReference
    :param cross_section: a cross section to transition to
    :return: a ComponentReference for the taper component placed such that it will connect to the input port

    .. plot::
        :include-source:

        import picbuilder.components as pbc
        import picbuilder.cross_sections as cs
        from picbuilder.routing import taper_to_cross_section
        import gdsfactory as gf
        from picbuilder.core import PDK

        PDK.activate()

        c = gf.Component()

        # create a component reference to connect to
        wg = c << pbc.rp_rib_waveguide()
        # create a taper reference transitioning to strip from the rib waveguide
        taper = taper_to_cross_section(wg.ports['out0'], cs.wg_most_cross_section())
        # add the taper reference to the parent component
        c.add(taper)

        c.plot()

    """
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
    taper = get_component(taper_name, width_input=port_width, width_output=cs_width)
    taper_ref = ComponentReference(taper).connect("in0", port)
    return taper_ref


def _auto_taper(
    routing_func,
    ports1,
    ports2,
    cross_section: CrossSectionSpec,
    auto_taper=True,
    **kwargs,
):
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

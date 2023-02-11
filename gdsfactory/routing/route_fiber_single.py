from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import gdstk
import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.component_layout import _rotate_points
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.route_fiber_array import route_fiber_array
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Label


def route_fiber_single(
    component: Component,
    fiber_spacing: float = 50.0,
    grating_coupler: ComponentSpec = grating_coupler_te,
    min_input_to_output_spacing: float = 200.0,
    optical_routing_type: int = 1,
    optical_port_labels: Optional[Tuple[str, ...]] = None,
    excluded_ports: Optional[Tuple[str, ...]] = None,
    component_name: Optional[str] = None,
    select_ports: Callable = select_ports_optical,
    cross_section: CrossSectionSpec = strip,
    **kwargs,
) -> Tuple[List[Union[ComponentReference, Label]], List[ComponentReference]]:
    """Returns route Tuple(references, grating couplers) for single fiber input/output.

    Args:
        component: to add grating couplers.
        fiber_spacing: between grating couplers.
        grating_coupler: grating coupler Spec
        min_input_to_output_spacing: so opposite fibers do not touch
        optical_routing_type: 0 (basic), 1 (standard), 2 (looks at ports)
        optical_port_labels: port labels that need connection
        excluded_ports: ports excluded from routing
        component_name: Optional component name.
        select_ports: function to select ports.
        cross_section: spec.
        kwargs: cross_section settings

    Returns:
        elements: list of ComponentReferences for routes and labels
        grating_couplers: list of grating_couplers references


    .. code::
              _________
             |         |_E1
          W0_|         |
             |         |_E0
             |_________|


         rotates +90 deg and routes West ports to South

         the rest of the original ports (East, North, South) will route south

         it calls route_fiber_array twice

         route_fiber_array is designed to route ports south

               E1  E0
              _|___|_
             |       |
             |       |
             |       |
             |       |
             |       |
             |       |
             |_______|
                 |
                 W0     1st part routes West ports south

        then rotates 180 and routes the rest of the ports North

    """
    if not select_ports(component.ports):
        raise ValueError(f"No ports for {component.name}")

    component = component.copy()
    component_copy = component.copy()

    if optical_port_labels is None:
        optical_ports = select_ports(component.ports)
    else:
        optical_ports = [component.ports[lbl] for lbl in optical_port_labels]

    excluded_ports = excluded_ports or []
    optical_ports = {
        p.name: p for p in optical_ports.values() if p.name not in excluded_ports
    }
    if isinstance(grating_coupler, list):
        grating_couplers = [gf.call_if_func(g) for g in grating_coupler]
        grating_coupler = grating_couplers[0]
    else:
        grating_coupler = gf.call_if_func(grating_coupler)
        N = len(optical_ports)

        grating_couplers = [grating_coupler] * N

    gc_port2center = getattr(grating_coupler, "port2center", grating_coupler.xsize / 2)
    if component.xsize + 2 * gc_port2center < min_input_to_output_spacing:
        fanout_length = (
            gf.snap.snap_to_grid(
                min_input_to_output_spacing - component.xsize - 2 * gc_port2center, 10
            )
            / 2
        )
    else:
        fanout_length = None

    # route WEST ports to south
    component_west_ports = Component()
    ref = component_west_ports << component
    ref.rotate(90)
    south_ports = ref.get_ports_dict(orientation=270)
    component_west_ports.ports = south_ports

    if len(south_ports):
        (
            elements_south,
            gratings_south,
            ports_grating_input_waveguide_south,
            ports_loopback_south,
            ports_component_south,
        ) = route_fiber_array(
            component=component_west_ports,
            with_loopback=False,
            fiber_spacing=fiber_spacing,
            fanout_length=fanout_length,
            grating_coupler=grating_couplers[0],
            optical_routing_type=optical_routing_type,
            component_name=component_name,
            cross_section=cross_section,
            select_ports=select_ports,
            **kwargs,
        )

    # route non WEST ports north
    component = gf.Component()
    component_ref = component << component_copy
    component_ref.rotate(-90)
    component.add_ports(component_ref.ports)
    for port_already_routed in south_ports.values():
        component.ports.pop(port_already_routed.name)

    component.ports = select_ports(component.ports)

    (
        elements_north,
        gratings_north,
        ports_grating_input_waveguide_north,
        ports_loopback_north,
        ports_component_north,
    ) = route_fiber_array(
        component=component,
        with_loopback=False,
        fiber_spacing=fiber_spacing,
        fanout_length=fanout_length,
        grating_coupler=grating_couplers[1:],
        optical_routing_type=optical_routing_type,
        component_name=component_name,
        cross_section=cross_section,
        select_ports=select_ports,
        **kwargs,
    )
    for e in elements_north:
        if isinstance(e, list):
            for ei in e:
                if isinstance(ei, gdstk.Label):
                    ei.rotation = np.mod(ei.rotation + np.pi, 2 * np.pi)
                    print(ei.rotation)
                    ei.origin = _rotate_points(
                        ei.origin,
                        angle=np.rad2deg(ei.rotation),
                    )

                    elements_south.append(ei)
                else:
                    elements_south.append(ei.rotate(180))
        elif isinstance(e, gdstk.Label):
            ei = e
            ei.rotation = np.mod(ei.rotation + np.pi, 2 * np.pi)
            ei.origin = _rotate_points(
                ei.origin,
                angle=np.rad2deg(ei.rotation),
            )
            elements_south.append(e)
        else:
            elements_south.append(e.rotate(180))

    if len(gratings_north) > 0:
        for io in gratings_north[0]:
            gratings_south.append(io.rotate(180))

    ports_grating_input_waveguide = (
        ports_grating_input_waveguide_north + ports_grating_input_waveguide_south
    )
    ports_component = ports_component_north + ports_component_south
    return (
        elements_south,
        gratings_south,
        ports_grating_input_waveguide,
        ports_component,
    )


if __name__ == "__main__":
    gcte = gf.components.grating_coupler_te
    gctm = gf.components.grating_coupler_tm

    c = gf.components.cross(length=500)
    c = gf.components.ring_double()
    c = gf.components.crossing()
    c = gf.components.rectangle()

    # elements, gc = route_fiber_single(
    #     c, grating_coupler=[gcte, gctm, gcte, gctm],
    # )

    layer = (2, 0)
    c = gf.components.straight(width=2, length=500)
    c = gf.components.mmi1x2(length_mmi=167)
    c = gf.components.ring_single(length_x=167)
    c = gf.components.mmi2x2()
    # c = gf.components.spiral_inner_io_fiber_single()

    gc = gf.components.grating_coupler_elliptical_te(layer=layer)
    (
        elements,
        gc,
        ports_grating_input_waveguide,
        ports_component,
    ) = route_fiber_single(
        c,
        # grating_coupler=[gc, gc, gc, gc],
        radius=10,
        layer=layer,
    )

    cc = gf.Component("sample_route_fiber_single")
    cr = cc << c
    cr.rotate(90)

    for e in elements:
        cc.add(e)
    for e in gc:
        cc.add(e)

    cc.add_ports(ports_grating_input_waveguide)
    cc.show(show_ports=True)

    # layer = (31, 0)
    # c = gf.components.mmi2x2()
    # c = gf.components.straight(width=2, length=500)
    # gc = gf.components.grating_coupler_elliptical_te(layer=layer)
    # elements, gc = route_fiber_single(
    #     c,
    #     grating_coupler=[gc, gc, gc, gc],
    #     radius=10,
    #     layer=layer,
    # )
    # cc = gf.Component("sample_route_fiber_single")
    # cr = cc << c.rotate(90)
    # for e in elements:
    #     cc.add(e)
    # for e in gc:
    #     cc.add(e)
    # cc.show(show_ports=True)

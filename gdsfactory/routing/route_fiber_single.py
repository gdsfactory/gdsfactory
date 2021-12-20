from typing import Callable, List, Optional, Tuple, Union

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.route_fiber_array import route_fiber_array
from gdsfactory.types import CrossSectionFactory, Label


def route_fiber_single(
    component: Component,
    fiber_spacing: float = 50.0,
    grating_coupler: Callable = grating_coupler_te,
    min_input_to_output_spacing: float = 200.0,
    optical_routing_type: int = 1,
    optical_port_labels: Optional[Tuple[str, ...]] = None,
    excluded_ports: Optional[Tuple[str, ...]] = None,
    auto_widen: bool = False,
    component_name: Optional[str] = None,
    select_ports: Callable = select_ports_optical,
    cross_section: CrossSectionFactory = strip,
    **kwargs,
) -> Tuple[List[Union[ComponentReference, Label]], List[ComponentReference]]:
    """Returns route Tuple(references, grating couplers) for single fiber input/output.

    Args:
        component: to add grating couplers
        fiber_spacing: between grating couplers
        grating_coupler:
        min_input_to_output_spacing: so opposite fibers do not touch
        optical_routing_type: 0 (basic), 1 (standard), 2 (looks at ports)
        optical_port_labels: port labels that need connection
        excluded_ports: ports excluded from routing
        auto_widen: for long routes
        component_name:
        select_ports:
        cross_section:
        **kwargs: cross_section settings

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
    N = len(optical_ports)

    if isinstance(grating_coupler, list):
        grating_couplers = [gf.call_if_func(g) for g in grating_coupler]
        grating_coupler = grating_couplers[0]
    else:
        grating_coupler = gf.call_if_func(grating_coupler)
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
        elements_south, gratings_south, _ = route_fiber_array(
            component=component_west_ports,
            with_loopback=False,
            fiber_spacing=fiber_spacing,
            fanout_length=fanout_length,
            grating_coupler=grating_couplers[0],
            optical_routing_type=optical_routing_type,
            auto_widen=auto_widen,
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

    elements_north, gratings_north, _ = route_fiber_array(
        component=component,
        with_loopback=False,
        fiber_spacing=fiber_spacing,
        fanout_length=fanout_length,
        grating_coupler=grating_couplers[1:],
        optical_routing_type=optical_routing_type,
        auto_widen=auto_widen,
        component_name=component_name,
        cross_section=cross_section,
        select_ports=select_ports,
        **kwargs,
    )
    for e in elements_north:
        if isinstance(e, list):
            for ei in e:
                elements_south.append(ei.rotate(180))
        else:
            elements_south.append(e.rotate(180))

    if len(gratings_north) > 0:
        for io in gratings_north[0]:
            gratings_south.append(io.rotate(180))

    return elements_south, gratings_south


if __name__ == "__main__":
    gcte = gf.components.grating_coupler_te
    gctm = gf.components.grating_coupler_tm

    c = gf.components.cross(length=500)
    c = gf.components.ring_double()
    c = gf.components.crossing()
    c = gf.components.rectangle()

    # elements, gc = route_fiber_single(
    #     c, grating_coupler=[gcte, gctm, gcte, gctm], auto_widen=False
    # )

    layer = (2, 0)
    c = gf.components.straight(width=2, length=500)
    c = gf.components.mmi1x2(length_mmi=167)
    c = gf.components.ring_single(length_x=167)
    c = gf.components.mmi2x2()
    c = gf.components.spiral(direction="NORTH")
    c = gf.c.spiral_inner_io_fiber_single()

    gc = gf.components.grating_coupler_elliptical_te(layer=layer)
    elements, gc = route_fiber_single(
        c,
        grating_coupler=[gc, gc, gc, gc],
        auto_widen=False,
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
    cc.show()

    # layer = (31, 0)
    # c = gf.components.mmi2x2()
    # c = gf.components.straight(width=2, length=500)
    # gc = gf.components.grating_coupler_elliptical_te(layer=layer)
    # elements, gc = route_fiber_single(
    #     c,
    #     grating_coupler=[gc, gc, gc, gc],
    #     auto_widen=False,
    #     radius=10,
    #     layer=layer,
    # )
    # cc = gf.Component("sample_route_fiber_single")
    # cr = cc << c.rotate(90)
    # for e in elements:
    #     cc.add(e)
    # for e in gc:
    #     cc.add(e)
    # cc.show()

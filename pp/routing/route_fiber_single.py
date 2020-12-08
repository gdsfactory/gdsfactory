from typing import List, Optional, Tuple, Union

from phidl.device_layout import Label

import pp
from pp.component import Component, ComponentReference
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_te
from pp.routing.route_fiber_array import route_fiber_array


def route_fiber_single(
    component: Component,
    optical_io_spacing: int = 50,
    grating_coupler: Component = grating_coupler_te,
    min_input2output_spacing: int = 230,
    optical_routing_type: int = 1,
    optical_port_labels: Optional[List[str]] = None,
    excluded_ports: Optional[List[str]] = None,
    **kwargs
) -> Tuple[List[Union[ComponentReference, Label]], List[ComponentReference]]:
    """Returns routes with grating couplers for single fiber input/output.

    Args:
        component: to add grating couplers
        optical_io_spacing: between grating couplers
        grating_coupler:
        min_input2output_spacing: so opposite fibers do not touch
        optical_routing_type: 0, 1, 2
        optical_port_labels: port labels that need connection
        excluded_ports: ports excluded from routing

    Returns:
        elements: list of routes ComponentReference
        grating_couplers: list of grating_couplers ComponentReferences

    """
    component = component.copy()
    component_copy = component.copy()

    if optical_port_labels is None:
        optical_ports = component.get_ports_list(port_type="optical")
    else:
        optical_ports = [component.ports[lbl] for lbl in optical_port_labels]

    excluded_ports = excluded_ports or []
    optical_ports = [p for p in optical_ports if p.name not in excluded_ports]
    N = len(optical_ports)

    if isinstance(grating_coupler, list):
        grating_couplers = [pp.call_if_func(g) for g in grating_coupler]
        grating_coupler = grating_couplers[0]
    else:
        grating_coupler = pp.call_if_func(grating_coupler)
        grating_couplers = [grating_coupler] * N

    gc_port2center = getattr(grating_coupler, "port2center", grating_coupler.xsize / 2)
    if component.xsize + 2 * gc_port2center < min_input2output_spacing:
        fanout_length = (
            pp.drc.snap_to_grid(
                min_input2output_spacing - component.xsize - 2 * gc_port2center, 10
            )
            / 2
        )
    else:
        fanout_length = None

    """
         _________
        |         |_E1
     W0_|         |
        |         |_E0
        |_________|

    rotate +90 deg and route West ports to South

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
            W0

    """
    # route west ports to south
    component = component.rotate(90)
    west_ports = component.get_ports_dict(prefix="W")
    north_ports = {
        p.name: p for p in component.ports.values() if not p.name.startswith("W")
    }
    component.ports = west_ports

    elements_south, gratings_south, _ = route_fiber_array(
        component=component,
        with_align_ports=False,
        optical_io_spacing=optical_io_spacing,
        fanout_length=fanout_length,
        grating_coupler=grating_couplers[0],
        optical_routing_type=optical_routing_type,
        **kwargs
    )

    # route north ports
    component = component_copy.rotate(-90)
    north_ports = {
        p.name: p for p in component.ports.values() if not p.name.startswith("W")
    }
    component.ports = north_ports

    elements_north, gratings_north, _ = route_fiber_array(
        component=component,
        with_align_ports=False,
        optical_io_spacing=optical_io_spacing,
        fanout_length=fanout_length,
        grating_coupler=grating_couplers[1:],
        **kwargs
    )
    for e in elements_north:
        elements_south.append(e.rotate(180))

    if len(gratings_north) > 0:
        for io in gratings_north[0]:
            gratings_south.append(io.rotate(180))

    return elements_south, gratings_south


if __name__ == "__main__":
    gcte = pp.c.grating_coupler_te
    gctm = pp.c.grating_coupler_tm

    # c = pp.c.crossing()
    # c = pp.c.mmi2x2()
    # c = pp.c.ring_double()  # FIXME
    c = pp.c.cross(length=500)
    c = pp.c.waveguide(width=2, length=500)

    elements, gc, _ = route_fiber_single(c, grating_coupler=[gcte, gctm, gcte, gctm])

    cc = pp.Component()
    cr = cc << c.rotate(90)

    for e in elements:
        cc.add(e)
    for e in gc:
        cc.add(e)
    pp.show(cc)

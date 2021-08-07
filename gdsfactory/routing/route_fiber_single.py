from typing import Callable, List, Optional, Tuple, Union

from phidl.device_layout import Label

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.components.grating_coupler.elliptical_trenches import grating_coupler_te
from gdsfactory.routing.route_fiber_array import route_fiber_array
from gdsfactory.types import StrOrDict


def route_fiber_single(
    component: Component,
    fiber_spacing: float = 50.0,
    grating_coupler: Callable = grating_coupler_te,
    min_input_to_output_spacing: float = 200.0,
    optical_routing_type: int = 1,
    optical_port_labels: Optional[List[str]] = None,
    excluded_ports: Optional[List[str]] = None,
    auto_widen: bool = False,
    component_name: Optional[str] = None,
    waveguide: StrOrDict = "strip",
    **waveguide_settings,
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

    Returns:
        elements: list of routes ComponentReference
        grating_couplers: list of grating_couplers ComponentReferences

    """
    # print(waveguide_settings)
    if not component.get_ports_list(port_type="optical"):
        raise ValueError(f"No ports for {component.name}")

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
        with_loopback=False,
        fiber_spacing=fiber_spacing,
        fanout_length=fanout_length,
        grating_coupler=grating_couplers[0],
        optical_routing_type=optical_routing_type,
        auto_widen=auto_widen,
        component_name=component_name,
        waveguide=waveguide,
        **waveguide_settings,
    )

    # route north ports
    component = component_copy.rotate(-90)
    north_ports = {
        p.name: p for p in component.ports.values() if not p.name.startswith("W")
    }
    component.ports = north_ports

    elements_north, gratings_north, _ = route_fiber_array(
        component=component,
        with_loopback=False,
        fiber_spacing=fiber_spacing,
        fanout_length=fanout_length,
        grating_coupler=grating_couplers[1:],
        optical_routing_type=optical_routing_type,
        auto_widen=auto_widen,
        component_name=component_name,
        waveguide=waveguide,
        **waveguide_settings,
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
    c = gf.components.mmi2x2()
    c = gf.components.crossing()
    c = gf.components.rectangle()
    c = gf.components.ring_single()

    # elements, gc = route_fiber_single(
    #     c, grating_coupler=[gcte, gctm, gcte, gctm], auto_widen=False
    # )

    c = gf.components.mmi2x2(waveguide="nitride")
    c = gf.components.straight(width=2, length=500)
    gc = gf.components.grating_coupler_elliptical_te(
        layer=gf.TECH.waveguide.nitride.layer
    )
    elements, gc = route_fiber_single(
        c,
        grating_coupler=[gc, gc, gc, gc],
        auto_widen=False,
        layer=(2, 0),
        radius=10,
        waveguide="nitride",
    )

    cc = gf.Component("sample_route_fiber_single")
    cr = cc << c.rotate(90)

    for e in elements:
        cc.add(e)
    for e in gc:
        cc.add(e)
    cc.show()

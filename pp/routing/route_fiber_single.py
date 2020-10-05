from typing import Any, List, Tuple, Union
from phidl.device_layout import Label
import pp
from pp.rotate import rotate
from pp.routing.connect_component import route_fiber_array
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_te
from pp.component import Component, ComponentReference


def route_fiber_single(
    component: Component,
    optical_io_spacing: int = 50,
    grating_coupler: Component = grating_coupler_te,
    min_input2output_spacing: int = 230,
    optical_routing_type: int = 2,
    optical_port_labels: None = None,
    excluded_ports: List[Any] = [],
    **kwargs
) -> Union[
    Tuple[List[Union[ComponentReference, Label]], List[List[ComponentReference]], None],
    Tuple[
        List[Union[ComponentReference, Label]],
        List[Union[List[ComponentReference], ComponentReference]],
        None,
    ],
]:
    """Returns component I/O for optical testing with single input and oputput fibers (no fiber array)

    Args:
        component: to add grating couplers
        optical_io_spacing: between grating couplers
        grating_coupler:
        straight_factory
        min_input2output_spacing: so opposite fibers do not touch
        optical_routing_type: 0, 1, 2

    .. plot::
      :include-source:

       import pp
       from pp.routing import add_io_optical
       from pp.routing import route_fiber_single

       c = pp.c.mmi1x2()
       cc = add_io_optical(c, get_route_factory=route_fiber_single)
       pp.plotgds(cc)
    """
    component_name = component.name
    component = component.copy()

    if optical_port_labels is None:
        optical_ports = component.get_optical_ports()
    else:
        optical_ports = [component.ports[lbl] for lbl in optical_port_labels]
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

    west_ports = [p for p in component.get_optical_ports() if p.name.startswith("W")]
    east_ports = [
        p for p in component.get_optical_ports() if not p.name.startswith("W")
    ]

    # add west input grating couplers
    component.ports = {p.name: p for p in west_ports}
    component = component.rotate(90)

    elements_east, io_grating_lines_east, _ = route_fiber_array(
        component=component,
        with_align_ports=False,
        optical_io_spacing=optical_io_spacing,
        fanout_length=fanout_length,
        grating_coupler=grating_couplers[0],
        optical_routing_type=optical_routing_type,
        **kwargs
    )
    component = rotate(component, angle=-90)

    # add EAST input grating couplers
    component.ports = {p.name: p for p in east_ports}
    component = rotate(component, angle=-90)

    component.name = component_name
    elements_west, io_grating_lines_west, _ = route_fiber_array(
        component=component,
        with_align_ports=False,
        optical_io_spacing=optical_io_spacing,
        fanout_length=fanout_length,
        grating_coupler=grating_couplers[1:],
        **kwargs
    )
    for e in elements_west:
        elements_east.append(e.rotate(180))

    if len(io_grating_lines_west) > 0:
        for io in io_grating_lines_west[0]:
            io_grating_lines_east.append(io.rotate(180))

    return elements_east, io_grating_lines_east, None


if __name__ == "__main__":
    gcte = pp.c.grating_coupler_te
    gctm = pp.c.grating_coupler_tm

    c = pp.c.crossing()
    # c = pp.c.mmi2x2()
    # c = pp.c.waveguide()
    # c = pp.c.ring_double()  # FIXME

    elements, gc, _ = route_fiber_single(c, grating_coupler=[gcte, gctm, gcte, gctm])
    for e in elements:
        c.add(e)
    for e in gc:
        c.add(e)
    pp.show(c)

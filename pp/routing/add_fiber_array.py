from typing import Callable, Optional

import pp
from pp.add_tapers import add_tapers
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.components.grating_coupler.elliptical_trenches import (
    grating_coupler_te,
    grating_coupler_tm,
)
from pp.components.taper import taper
from pp.container import container
from pp.routing.get_input_labels import get_input_labels
from pp.routing.route_fiber_array import route_fiber_array
from pp.tech import TECH_SILICON_C, Tech
from pp.types import ComponentFactory


def add_fiber_array_te(*args, **kwargs):
    return add_fiber_array(*args, **kwargs)


def add_fiber_array_tm(
    *args, grating_coupler=grating_coupler_tm, **kwargs
) -> Component:
    return add_fiber_array(*args, grating_coupler=grating_coupler, **kwargs)


@container
def add_fiber_array(
    component: Component,
    grating_coupler: Component = grating_coupler_te,
    bend_factory: ComponentFactory = bend_euler,
    gc_port_name: str = "W0",
    component_name: Optional[str] = None,
    taper_factory: Callable = taper,
    taper_length: Optional[float] = None,
    get_input_labels_function: Callable = get_input_labels,
    tech: Tech = TECH_SILICON_C,
    **kwargs,
) -> Component:
    """Returns component with optical IO (tapers, south routes and grating_couplers).

    Args:
        component: to connect
        grating_coupler: grating coupler instance, function or list of functions
        bend_factory: bend_circular
        gc_port_name: grating coupler input port name 'W0'
        component_name: for the label
        taper_factory: taper function
        taper_length: length of the taper
        get_input_labels_function: function to get input labels for grating couplers
        straight_factory: waveguide
        fanout_length: None  # if None, automatic calculation of fanout length
        max_y0_optical: None
        with_align_ports: True, adds loopback structures
        waveguide_separation: 4.0
        bend_radius: optional bend_radius (defaults to tech.bend_radius)
        list_port_labels: None, adds TM labels to port indices in this list
        connected_port_list_ids: None # only for type 0 optical routing
        nb_optical_ports_lines: 1
        force_manhattan: False
        excluded_ports:
        grating_indices: None
        routing_waveguide: None
        routing_method: get_route
        optical_routing_type: None: auto, 0: no extension, 1: standard, 2: check
        gc_rotation: -90
        layer_label: LAYER.LABEL
        input_port_indexes: [0]

    .. plot::
      :include-source:

       import pp

       c = pp.c.crossing()
       cc = pp.routing.add_fiber_array(c)
       cc.plot()

    """
    c = component
    if not c.ports:
        return c

    if isinstance(grating_coupler, list):
        gc = grating_coupler[0]
    else:
        gc = grating_coupler
    gc = pp.call_if_func(gc)

    gc_polarization = gc.polarization

    component_name = component_name or c.name
    name = f"{component_name}_{gc_polarization}"
    cc = pp.Component(name=name)

    port_width_gc = gc.ports[gc_port_name].width

    optical_ports = c.get_ports_list(port_type="optical")
    port_width_component = optical_ports[0].width

    taper_length = taper_length or tech.taper_length

    if port_width_component != port_width_gc:
        c = add_tapers(
            c,
            taper_factory(
                length=taper_length, width1=port_width_gc, width2=port_width_component
            ),
        )

    # for pn, p in c.ports.items():
    #     print(p.name, p.port_type, p.layer)

    elements, io_gratings_lines, _ = route_fiber_array(
        component=c,
        grating_coupler=grating_coupler,
        bend_factory=bend_factory,
        gc_port_name=gc_port_name,
        component_name=component_name,
        get_input_labels_function=get_input_labels_function,
        tech=tech,
        **kwargs,
    )
    if len(elements) == 0:
        return c

    for e in elements:
        cc.add(e)
    for io_gratings in io_gratings_lines:
        cc.add(io_gratings)
    cc.add(c.ref())
    cc.move(origin=io_gratings_lines[0][0].ports[gc_port_name], destination=(0, 0))

    for pname, p in c.ports.items():
        if p.port_type != "optical":
            cc.add_port(pname, port=p)

    return cc


def test_type0() -> Component:
    component = pp.c.coupler(gap=0.244, length=5.67)
    cc = add_fiber_array(component, optical_routing_type=0)
    pp.write_gds(cc)
    return cc


def test_type1() -> Component:
    component = pp.c.coupler(gap=0.2, length=5.0)
    cc = add_fiber_array(component, optical_routing_type=1)
    pp.write_gds(cc)
    return cc


def test_type2() -> Component:
    c = pp.c.coupler(gap=0.244, length=5.67)
    c.polarization = "tm"
    cc = add_fiber_array(c, optical_routing_type=2)
    pp.write_gds(cc)
    return cc


def demo_tapers():
    c = pp.c.waveguide(width=2)
    cc = add_fiber_array(c, optical_routing_type=2)
    return cc


def demo_te_and_tm():
    c = pp.Component()
    w = pp.c.waveguide()
    wte = add_fiber_array(w, grating_coupler=pp.c.grating_coupler_elliptical_te)
    wtm = add_fiber_array(w, grating_coupler=pp.c.grating_coupler_elliptical_tm)
    c.add_ref(wte)
    wtm_ref = c.add_ref(wtm)
    wtm_ref.movey(wte.size_info.height)
    return c


if __name__ == "__main__":
    test_type0()
    gcte = pp.c.grating_coupler_te
    gctm = pp.c.grating_coupler_tm

    # from pprint import pprint

    layer_label = pp.LAYER.TEXT
    layer_label = (66, 5)

    # cc = demo_tapers()
    # cc = test_type1()
    # pprint(cc.get_json())

    # c = pp.c.coupler(gap=0.2, length=5.6)

    c = pp.c.waveguide()
    c = pp.c.waveguide(length=1, width=2)
    c = pp.c.mmi2x2()
    c = pp.c.ring_single()
    c = pp.c.mzi2x2()

    c.y = 0
    cc = add_fiber_array(
        c,
        # optical_routing_type=0,  # needs fix for mzi2x2
        # optical_routing_type=1,
        # optical_routing_type=2,
        # layer_label=layer_label,
        # get_route_factory=route_fiber_single,
        # get_route_factory=route_fiber_array,
        grating_coupler=[gcte, gctm, gcte, gctm],
        bend_radius=20,
    )
    # cc = demo_te_and_tm()
    # print(cc.ports.keys())
    cc.show()

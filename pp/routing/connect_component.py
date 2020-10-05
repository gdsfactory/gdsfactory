from typing import Callable
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_te
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_tm

import pp
from pp.add_tapers import add_tapers
from pp.components.taper import taper
from pp.container import container

from pp.routing.route_fiber_array import route_fiber_array
from pp.routing.get_input_labels import get_input_labels
from pp.component import Component


def add_io_optical_te(*args, **kwargs):
    return add_io_optical(*args, **kwargs)


def add_io_optical_tm(*args, grating_coupler=grating_coupler_tm, **kwargs):
    return add_io_optical(*args, grating_coupler=grating_coupler, **kwargs)


@container
def add_io_optical(
    component: Component,
    grating_coupler: Component = grating_coupler_te,
    gc_port_name: str = "W0",
    component_name: None = None,
    taper_factory: Callable = taper,
    get_route_factory: Callable = route_fiber_array,
    get_input_labels_function: Callable = get_input_labels,
    **kwargs,
) -> Component:
    """returns component with optical IO (tapers, south routes and grating_couplers)

    Args:
        component: to connect
        optical_io_spacing: SPACING_GC
        grating_coupler: grating coupler instance, function or list of functions
        bend_factory: bend_circular
        straight_factory: waveguide
        fanout_length: None  # if None, automatic calculation of fanout length
        max_y0_optical: None
        with_align_ports: True, adds loopback structures
        waveguide_separation: 4.0
        bend_radius: BEND_RADIUS
        list_port_labels: None, adds TM labels to port indices in this list
        connected_port_list_ids: None # only for type 0 optical routing
        nb_optical_ports_lines: 1
        force_manhattan: False
        excluded_ports:
        grating_indices: None
        routing_waveguide: None
        routing_method: connect_strip
        gc_port_name: W0
        optical_routing_type: None: autoselection, 0: no extension
        gc_rotation: -90
        layer_label: LAYER.LABEL
        input_port_indexes: [0]
        component_name: for the label
        taper_factory: taper

    .. plot::
      :include-source:

       import pp
       from pp.routing import add_io_optical

       c = pp.c.mmi1x2()
       cc = add_io_optical(c)
       pp.plotgds(cc)

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
    cc.function_name = "add_io_optical"

    port_width_gc = list(gc.ports.values())[0].width

    optical_ports = c.get_optical_ports()
    port_width_component = optical_ports[0].width

    if port_width_component != port_width_gc:
        c = add_tapers(
            c,
            taper_factory(length=10, width1=port_width_gc, width2=port_width_component),
        )

    # for pn, p in c.ports.items():
    #     print(p.name, p.port_type, p.layer)

    elements, io_gratings_lines, _ = get_route_factory(
        component=c,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        component_name=component_name,
        get_input_labels_function=get_input_labels_function,
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


def test_type0():
    component = pp.c.coupler(gap=0.244, length=5.67)
    cc = add_io_optical(component, optical_routing_type=0)
    pp.write_gds(cc)
    return cc


def test_type1():
    component = pp.c.coupler(gap=0.2, length=5.0)
    cc = add_io_optical(component, optical_routing_type=1)
    pp.write_gds(cc)
    return cc


def test_type2():
    c = pp.c.coupler(gap=0.244, length=5.67)
    c.polarization = "tm"
    cc = add_io_optical(c, optical_routing_type=2)
    pp.write_gds(cc)
    return cc


def demo_tapers():
    c = pp.c.waveguide(width=2)
    cc = add_io_optical(c, optical_routing_type=2)
    return cc


def demo_te_and_tm():
    c = pp.Component()
    w = pp.c.waveguide()
    wte = add_io_optical(w, grating_coupler=pp.c.grating_coupler_elliptical_te)
    wtm = add_io_optical(w, grating_coupler=pp.c.grating_coupler_elliptical_tm)
    c.add_ref(wte)
    wtm_ref = c.add_ref(wtm)
    wtm_ref.movey(wte.size_info.height)
    return c


if __name__ == "__main__":
    gcte = pp.c.grating_coupler_te
    gctm = pp.c.grating_coupler_tm

    # from pprint import pprint

    layer_label = pp.LAYER.TEXT
    layer_label = (66, 5)

    # cc = demo_tapers()
    # cc = test_type1()
    # pprint(cc.get_json())

    # c = pp.c.coupler(gap=0.2, length=5.6)

    c = pp.c.mmi2x2()
    # c = pp.c.waveguide()

    c.y = 0
    cc = add_io_optical(
        c,
        # optical_routing_type=0,
        # optical_routing_type=1,
        # optical_routing_type=2,
        # layer_label=layer_label,
        # get_route_factory=route_fiber_single,
        # get_route_factory=route_fiber_array,
        grating_coupler=[gcte, gctm, gcte, gctm],
    )
    # cc = demo_te_and_tm()
    # print(cc.ports.keys())
    pp.show(cc)
    print(cc.get_settings()["component"])

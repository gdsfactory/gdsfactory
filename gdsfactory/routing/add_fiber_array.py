from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.grating_coupler.elliptical_trenches import grating_coupler_te
from gdsfactory.components.straight import straight
from gdsfactory.routing.route_fiber_array import route_fiber_array
from gdsfactory.types import ComponentFactory, StrOrDict


@gf.cell_without_validator
def add_fiber_array(
    component: Component,
    grating_coupler: Component = grating_coupler_te,
    straight_factory: ComponentFactory = straight,
    bend_factory: ComponentFactory = bend_euler,
    gc_port_name: str = "W0",
    component_name: Optional[str] = None,
    taper_length: float = 10.0,
    waveguide: StrOrDict = "strip",
    **kwargs,
) -> Component:
    """Returns component with optical IO (tapers, south routes and grating_couplers).

    Args:
        component: to connect
        grating_coupler: grating coupler instance, function or list of functions
        bend_factory: bend_circular
        gc_port_name: grating coupler input port name 'W0'
        component_name: for the label
        taper: taper function name or dict
        get_input_labels_function: function to get input labels for grating couplers
        get_input_label_text_loopback_function: function to get input label test
        get_input_label_text_function
        straight_factory: straight
        fanout_length: None  # if None, automatic calculation of fanout length
        max_y0_optical: None
        with_loopback: True, adds loopback structures
        straight_separation: 4.0
        list_port_labels: None, adds TM labels to port indices in this list
        connected_port_list_ids: None # only for type 0 optical routing
        nb_optical_ports_lines: 1
        force_manhattan: False
        excluded_ports:
        grating_indices: None
        routing_straight: None
        routing_method: get_route
        optical_routing_type: None: auto, 0: no extension, 1: standard, 2: check
        gc_rotation: -90
        layer_label: LAYER.LABEL
        input_port_indexes: [0]

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.crossing()
        cc = gf.routing.add_fiber_array(
            component=c,
            optical_routing_type=2,
            grating_coupler=gf.components.grating_coupler_elliptical_te,
        )
        cc.plot()

    """
    component = gf.call_if_func(component)
    grating_coupler = (
        grating_coupler() if callable(grating_coupler) else grating_coupler
    )
    c = component
    if not c.ports:
        return c

    if isinstance(grating_coupler, list):
        gc = grating_coupler[0]
    else:
        gc = grating_coupler
    gc = gf.call_if_func(gc)

    if gc_port_name not in gc.ports:
        raise ValueError(f"gc_port_name={gc_port_name} not in {gc.ports.keys()}")

    component_name = component_name or c.name
    cc = Component()

    optical_ports = c.get_ports_list(port_type="optical")
    if not optical_ports:
        return c

    elements, io_gratings_lines, _ = route_fiber_array(
        component=c,
        grating_coupler=grating_coupler,
        bend_factory=bend_factory,
        straight_factory=straight_factory,
        gc_port_name=gc_port_name,
        component_name=component_name,
        waveguide=waveguide,
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

    for i, io_row in enumerate(io_gratings_lines):
        for j, io in enumerate(io_row):
            ports = io.get_ports_list(prefix="vertical")
            if ports:
                port = ports[0]
                cc.add_port(f"{port.name}_{i}{j}", port=port)

    return cc


def demo_te_and_tm():
    c = gf.Component()
    w = gf.components.straight()
    wte = add_fiber_array(
        component=w, grating_coupler=gf.components.grating_coupler_elliptical_te
    )
    wtm = add_fiber_array(
        component=w, grating_coupler=gf.components.grating_coupler_elliptical_tm
    )
    c.add_ref(wte)
    wtm_ref = c.add_ref(wtm)
    wtm_ref.movey(wte.size_info.height)
    return c


if __name__ == "__main__":
    # test_type0()
    gcte = gf.components.grating_coupler_te
    gctm = gf.components.grating_coupler_tm

    # from pprint import pprint
    layer_label = gf.LAYER.TEXT
    layer_label = (66, 5)

    # cc = demo_tapers()
    # cc = test_type1()
    # pprint(cc.get_json())
    # c = gf.components.coupler(gap=0.2, length=5.6)
    # c = gf.components.straight()
    # c = gf.components.straight(length=1, width=2)
    # c = gf.components.mmi2x2()
    c = gf.components.ring_single()

    c.y = 0
    cc = add_fiber_array(
        component=c,
        # optical_routing_type=0,
        # optical_routing_type=1,
        # optical_routing_type=2,
        # layer_label=layer_label,
        # get_route_factory=route_fiber_single,
        # get_route_factory=route_fiber_array,
        grating_coupler=[gcte, gctm, gcte, gctm],
        auto_widen=False,
    )
    # cc = demo_te_and_tm()
    # print(cc.ports.keys())
    # print(cc.ports.keys())
    print(cc.name)
    cc.show()
    # cc.pprint()

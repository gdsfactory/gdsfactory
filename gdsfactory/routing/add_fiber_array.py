import warnings
from typing import Callable, Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.get_input_labels import get_input_labels
from gdsfactory.routing.route_fiber_array import route_fiber_array
from gdsfactory.routing.sort_ports import sort_ports_x
from gdsfactory.types import (
    ComponentFactory,
    ComponentOrFactory,
    ComponentOrFactoryOrList,
    CrossSectionFactory,
)


@gf.cell
def add_fiber_array(
    component: ComponentOrFactory,
    grating_coupler: ComponentOrFactoryOrList = grating_coupler_te,
    straight: ComponentFactory = straight_function,
    bend: ComponentFactory = bend_euler,
    gc_port_name: str = "o1",
    gc_port_labels: Optional[Tuple[str, ...]] = None,
    component_name: Optional[str] = None,
    select_ports: Callable = select_ports_optical,
    cross_section: CrossSectionFactory = strip,
    get_input_labels_function: Optional[Callable] = get_input_labels,
    layer_label: Optional[Tuple[int, int]] = (66, 0),
    **kwargs,
) -> Component:
    """Returns component with optical IO (tapers, south routes and grating_couplers).

    Args:
        component: to connect
        grating_coupler: grating coupler instance, function or list of functions
        straight: factory
        bend: bend_circular
        gc_port_name: grating coupler input port name 'W0'
        gc_port_labels: grating coupler list of labels
        component_name: for the label
        select_ports: function to select ports
        cross_section: cross_section function
        taper: taper function name or dict
        get_input_labels_function: function to get input labels for grating couplers
        get_input_label_text_loopback_function: function to get input label test
        get_input_label_text_function
        straight: straight
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
        gf.config.set_plot_options(show_subports=False)

        c = gf.components.crossing()
        cc = gf.routing.add_fiber_array(
            component=c,
            optical_routing_type=2,
            grating_coupler=gf.components.grating_coupler_elliptical_te,
            with_loopback=False
        )
        cc.plot()

    """
    get_input_labels_function = None if gc_port_labels else get_input_labels_function
    component = gf.call_if_func(component)
    grating_coupler = (
        grating_coupler() if callable(grating_coupler) else grating_coupler
    )
    if not component.ports:
        return component

    if isinstance(grating_coupler, list):
        gc = grating_coupler[0]
    else:
        gc = grating_coupler
    gc = gf.call_if_func(gc)

    orientation = int(gc.ports[gc_port_name].orientation)

    if orientation != 180:
        warnings.warn(
            "add_fiber_array requires a grating coupler port facing west "
            f"(orientation = 180). "
            f"Got orientation = {orientation} degrees for port {gc_port_name!r}"
        )

    if gc_port_name not in gc.ports:
        raise ValueError(f"gc_port_name={gc_port_name} not in {gc.ports.keys()}")

    component_name = component_name or component.info_child.get("name", component.name)
    component_new = Component()
    component_new.component = component

    optical_ports = select_ports(component.ports)
    optical_ports_names = list(optical_ports.keys())
    if not optical_ports:
        return component

    elements, io_gratings_lines, ports = route_fiber_array(
        component=component,
        grating_coupler=grating_coupler,
        bend=bend,
        straight=straight,
        gc_port_name=gc_port_name,
        component_name=component_name,
        cross_section=cross_section,
        select_ports=select_ports,
        get_input_labels_function=get_input_labels_function,
        layer_label=layer_label,
        **kwargs,
    )
    if len(elements) == 0:
        return component

    for e in elements:
        component_new.add(e)
    for io_gratings in io_gratings_lines:
        component_new.add(io_gratings)

    component_new.add_ref(component)

    for port in component.ports.values():
        if port.name not in optical_ports_names:
            component_new.add_port(port.name, port=port)

    ports = sort_ports_x(ports)

    if gc_port_labels:
        for gc_port_label, port in zip(gc_port_labels, ports):
            component_new.add_label(
                text=gc_port_label, layer=layer_label, position=port.midpoint
            )

    for i, io_row in enumerate(io_gratings_lines):
        for j, io in enumerate(io_row):
            ports = io.get_ports_list(prefix="vertical")
            if ports:
                port = ports[0]
                component_new.add_port(f"{port.name}_{i}{j}", port=port)

    component_new.copy_child_info(component)
    return component_new


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
    # layer_label = gf.LAYER.TEXT
    # layer_label = (66, 5)

    # cc = demo_tapers()
    # cc = test_type1()
    # pprint(cc.get_json())
    # c = gf.components.coupler(gap=0.2, length=5.6)
    # c = gf.components.straight()
    # c = gf.components.mmi2x2()
    # c = gf.components.ring_single()
    # c = gf.components.straight_heater_metal()
    c = gf.components.spiral(direction="NORTH")
    cc = add_fiber_array(
        component=c,
        # optical_routing_type=0,
        # optical_routing_type=1,
        # optical_routing_type=2,
        # layer_label=layer_label,
        # get_route_factory=route_fiber_single,
        # get_route_factory=route_fiber_array,
        grating_coupler=[gcte, gctm, gcte, gctm],
        # grating_coupler=gf.functions.rotate(gcte, angle=180),
        auto_widen=True,
        # layer=(2, 0),
        gc_port_labels=["loop_in", "in", "out", "loop_out"],
    )
    cc.show()

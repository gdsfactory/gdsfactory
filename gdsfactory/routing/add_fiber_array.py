import warnings
from typing import Callable, Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.get_input_labels import get_input_labels
from gdsfactory.routing.route_fiber_array import route_fiber_array
from gdsfactory.routing.sort_ports import sort_ports_x
from gdsfactory.types import (
    ComponentSpec,
    ComponentSpecOrList,
    CrossSectionSpec,
    LayerSpec,
)


@gf.cell
def add_fiber_array(
    component: ComponentSpec = straight_function,
    grating_coupler: ComponentSpecOrList = grating_coupler_te,
    gc_port_name: str = "o1",
    gc_port_labels: Optional[Tuple[str, ...]] = None,
    component_name: Optional[str] = None,
    select_ports: Callable = select_ports_optical,
    cross_section: CrossSectionSpec = "strip",
    get_input_labels_function: Optional[Callable] = get_input_labels,
    layer_label: LayerSpec = "TEXT",
    **kwargs,
) -> Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        gc_port_name: grating coupler input port name.
        gc_port_labels: grating coupler list of labels.
        component_name: for the label.
        select_ports: function to select ports.
        cross_section: cross_section function.
        get_input_labels_function: function to get input labels. None skips labels.
        layer_label: optional layer for grating coupler label.

    Keyword Args:
        bend: bend spec.
        straight: straight spec.
        taper: taper spec.
        get_input_label_text_loopback_function: function to get input label test.
        get_input_label_text_function: for labels.
        fanout_length: if None, automatic calculation of fanout length.
        max_y0_optical: in um.
        with_loopback: True, adds loopback structures.
        straight_separation: from edge to edge.
        list_port_labels: None, adds TM labels to port indices in this list.
        connected_port_list_ids: names of ports only for type 0 optical routing.
        nb_optical_ports_lines: number of grating coupler lines.
        force_manhattan: False
        excluded_ports: list of port names to exclude when adding gratings.
        grating_indices: list of grating coupler indices.
        routing_straight: function to route.
        routing_method: get_route.
        optical_routing_type: None: auto, 0: no extension, 1: standard, 2: check.
        gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
        input_port_indexes: to connect.

    .. plot::
        :include-source:

        import gdsfactory as gf

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
    component = gf.get_component(component)

    if isinstance(grating_coupler, list):
        gc = grating_coupler[0]
    else:
        gc = grating_coupler
    gc = gf.get_component(gc)

    if gc_port_name not in gc.ports:
        gc_ports = list(gc.ports.keys())
        raise ValueError(f"gc_port_name = {gc_port_name!r} not in {gc_ports}")

    orientation = gc.ports[gc_port_name].orientation

    grating_coupler = (
        [gf.get_component(i) for i in grating_coupler]
        if isinstance(grating_coupler, list)
        else gf.get_component(grating_coupler)
    )

    if int(orientation) != 180:
        warnings.warn(
            "add_fiber_array requires a grating coupler port facing west "
            f"(orientation = 180). "
            f"Got orientation = {orientation} degrees for port {gc_port_name!r}"
        )

    if gc_port_name not in gc.ports:
        raise ValueError(f"gc_port_name={gc_port_name} not in {gc.ports.keys()}")

    component_name = component_name or component.metadata_child.get(
        "name", component.name
    )
    component_new = Component()
    component_new.component = component

    optical_ports = select_ports(component.ports)
    optical_ports_names = list(optical_ports.keys())
    if not optical_ports:
        raise ValueError(f"No optical ports found in {component.name!r}")

    elements, io_gratings_lines, ports = route_fiber_array(
        component=component,
        grating_coupler=grating_coupler,
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
            if layer_label:
                component_new.add_label(
                    text=gc_port_label, layer=layer_label, position=port.center
                )

    for i, io_row in enumerate(io_gratings_lines):
        for j, io in enumerate(io_row):
            ports = io.get_ports_list(prefix="vertical") or io.get_ports_list()
            if ports:
                port = ports[0]
                component_new.add_port(f"{port.name}_{i}{j}", port=port)

    component_new.copy_child_info(component)
    component_new.info["grating_coupler"] = gc.info
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
    strip = gf.partial(
        gf.cross_section.cross_section,
        width=1,
        layer=(2, 0),
        bbox_layers=((61, 0), (62, 0)),
        bbox_offsets=(3, 3)
        # cladding_layers=((61, 0), (62, 0)),
        # cladding_offsets=(3, 3)
    )

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
    # c = gf.components.spiral(direction="NORTH")

    c = gf.components.bend_euler(info=dict(doe="bends"))
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
        cross_section=strip,
        info=dict(a=1),
    )
    cc.show(show_ports=True)

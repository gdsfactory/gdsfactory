from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.route_fiber_array import route_fiber_array
from gdsfactory.typings import (
    ComponentSpec,
    ComponentSpecOrList,
    CrossSectionSpec,
    PortsFactory,
)


def add_fiber_array(
    component: ComponentSpec = "straight",
    grating_coupler: ComponentSpecOrList = "grating_coupler_te",
    gc_port_name: str = "o1",
    select_ports: PortsFactory = select_ports_optical,
    cross_section: CrossSectionSpec = "strip",
    start_straight_length: float = 0,
    end_straight_length: float = 0,
    **kwargs: Any,
) -> Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        gc_port_name: grating coupler input port name.
        select_ports: function to select ports.
        cross_section: cross_section function.
        kwargs: additional arguments.

    Keyword Args:
        bend: bend spec.
        straight: straight spec.
        fanout_length: if None, automatic calculation of fanout length.
        max_y0_optical: in um.
        with_loopback: True, adds loopback structures.
        with_loopback_inside: True, adds loopback structures inside the component.
        straight_separation: from edge to edge.
        list_port_labels: None, adds TM labels to port indices in this list.
        nb_optical_ports_lines: number of grating coupler lines.
        force_manhattan: False
        excluded_ports: list of port names to exclude when adding gratings.
        grating_indices: list of grating coupler indices.
        routing_straight: function to route.
        routing_method: route_single.
        gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
        input_port_indexes: to connect.
        pitch: in um.
        radius: optional radius of the bend. Defaults to the cross_section.
        radius_loopback: optional radius of the loopback bend. Defaults to the cross_section.
        start_straight_length: length of the start straight.
        end_straight_length: length of the end straight.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.crossing()
        cc = gf.routing.add_fiber_array(
            component=c,
            grating_coupler=gf.components.grating_coupler_elliptical_te,
            with_loopback=False
        )
        cc.plot()

    """
    component = gf.get_component(component)

    if isinstance(grating_coupler, list):
        gc = grating_coupler[0]
    else:
        gc = grating_coupler
    gc = gf.get_component(gc)

    gc_port_names = [port.name for port in gc.ports]
    if gc_port_name not in gc_port_names:
        raise ValueError(f"gc_port_name = {gc_port_name!r} not in {gc_port_names}")

    orientation = gc.ports[gc_port_name].orientation

    grating_coupler = (
        [gf.get_component(i) for i in grating_coupler]
        if isinstance(grating_coupler, list)
        else gf.get_component(grating_coupler)
    )

    if int(orientation) != 180:
        raise ValueError(
            "add_fiber_array requires a grating coupler port facing west "
            f"(orientation = 180). "
            f"Got orientation = {orientation} degrees for port {gc_port_name!r}"
        )

    if gc_port_name not in gc.ports:
        raise ValueError(f"gc_port_name={gc_port_name!r} not in {list(gc.ports)}")

    component_new = Component()

    optical_ports = select_ports(component.ports)
    if not optical_ports:
        raise ValueError(f"No optical ports found in {component.name!r}")

    ref = component_new.add_ref(component)
    route_fiber_array(
        component_new,
        ref,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        cross_section=cross_section,
        select_ports=select_ports,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        **kwargs,
    )

    component_new.copy_child_info(component)
    return component_new


if __name__ == "__main__":
    # from gdsfactory.samples.big_device import big_device

    component = gf.c.mmi2x2()
    # component = big_device(nports=2)
    # radius = 5.0
    c = add_fiber_array(component=component)
    # test_type0()
    # gcte = gf.components.grating_coupler_te
    # gctm = gf.components.grating_coupler_tm
    # strip = partial(
    #     gf.cross_section.cross_section,
    #     width=1,
    #     layer=(2, 0),
    #     # bbox_layers=((61, 0), (62, 0)),
    #     # bbox_offsets=(3, 3)
    #     # cladding_layers=((61, 0), (62, 0)),
    #     # cladding_offsets=(3, 3)
    # )

    # from pprint import pprint

    # cc = demo_tapers()
    # cc = test_type1()
    # pprint(cc.get_json())
    # c = gf.components.coupler(gap=0.2, length=5.6)
    # c = gf.components.straight()
    # c = gf.components.mmi2x2()
    # c = gf.components.nxn(north=2, south=2)
    # c = gf.components.ring_single()
    # c = gf.components.straight_heater_metal()
    # c = gf.components.spiral(direction="NORTH")

    # c = gf.components.mzi_phase_shifter()
    # c = add_fiber_array(c, radius=20, with_loopback=False)

    # c1 = partial(add_fiber_array, component=gf.c.mmi1x2)
    # c2 = partial(add_fiber_array, component=gf.c.nxn)
    # c = gf.pack((c1,c2))[0]
    # c = c2()
    c.pprint_ports()
    c.show()

    # cc = add_fiber_array(
    #     component=c,
    #     # layer_label=layer_label,
    #     # route_single_factory=route_fiber_single,
    #     # route_single_factory=route_fiber_array,
    #     grating_coupler=gctm,
    #     # grating_coupler=[gcte, gctm, gcte, gctm],
    #     # grating_coupler=gf.functions.drotate(gcte, angle=180),
    #     auto_widen=True,
    #     # layer=(2, 0),
    #     # gc_port_labels=["loop_in", "in", "out", "loop_out"],
    #     cross_section=strip,
    #     info=dict(a=1),
    # )
    # cc.pprint_ports()
    # cc.show()

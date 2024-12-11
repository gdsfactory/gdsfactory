from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.route_fiber_array import route_fiber_array
from gdsfactory.typings import (
    ComponentSpec,
    ComponentSpecOrList,
    CrossSectionSpec,
    SelectPorts,
)


def add_fiber_single(
    component: ComponentSpec = "straight",
    grating_coupler: ComponentSpecOrList = "grating_coupler_te",
    gc_port_name: str = "o1",
    gc_port_name_fiber: str = "o2",
    select_ports: SelectPorts = select_ports_optical,
    cross_section: CrossSectionSpec = "strip",
    input_port_names: Sequence[str] | None = None,
    pitch: float = 70,
    with_loopback: bool = True,
    loopback_spacing: float = 100.0,
    straight: ComponentSpec = "straight",
    **kwargs: Any,
) -> Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        gc_port_name: grating coupler input port name.
        gc_port_name_fiber: grating coupler output port name.
        select_ports: function to select ports.
        cross_section: cross_section function.
        input_port_names: list of input port names to connect to grating couplers.
        pitch: spacing between fibers.
        with_loopback: adds loopback structures.
        loopback_spacing: spacing between loopback and test structure.
        straight: straight spec.
        kwargs: additional arguments.

    Keyword Args:
        bend: bend spec.
        straight: straight spec.
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
        routing_method: route_single.
        gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
        input_port_indexes: to connect.

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
    optical_ports = select_ports(component.ports)
    if not optical_ports:
        raise ValueError(f"No optical ports found in {component.name!r}")

    if isinstance(grating_coupler, list):
        gc = grating_coupler[0]
    else:
        gc = grating_coupler
    gc = gf.get_component(gc)
    if gc_port_name not in gc.ports:
        raise ValueError(f"gc_port_name={gc_port_name!r} not in {list(gc.ports)}")

    gc_port_names = [port.name for port in gc.ports]
    if gc_port_name_fiber not in gc_port_names:
        gc_port_name_fiber_option = next(
            (name for name in gc_port_names if name is not None), None
        )
        if gc_port_name_fiber_option is None:
            raise ValueError("No valid grating coupler port names found.")
        gc_port_name_fiber = gc_port_name_fiber_option

    if gc_port_name not in gc_port_names:
        gc_port_name_option = next(
            (name for name in gc_port_names if name is not None), None
        )
        if gc_port_name_option is None:
            raise ValueError("No valid grating coupler port names found.")
        gc_port_name = gc_port_name_option

    orientation = gc.ports[gc_port_name].orientation
    if int(orientation) != 180:
        raise ValueError(
            "add_fiber_array requires a grating coupler port facing west "
            f"(orientation = 180). "
            f"Got orientation = {orientation} degrees for port {gc_port_name!r}"
        )

    grating_coupler = (
        [gf.get_component(i) for i in grating_coupler]
        if isinstance(grating_coupler, list)
        else gf.get_component(grating_coupler)
    )

    c1 = Component()
    ref = c1.add_ref(component)

    input_port_names = list(
        input_port_names
        or [p.name for p in ref.ports.filter(orientation=180) if p.name is not None]
    )
    output_port_names = [
        port.name
        for port in ref.ports
        if port.name not in input_port_names
        if port.name is not None
    ]
    ref.drotate(+90)

    route_fiber_array(
        c1,
        ref,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        cross_section=cross_section,
        select_ports=select_ports,
        with_loopback=False,
        port_names=input_port_names,
        pitch=pitch,
        **kwargs,
    )

    c2 = Component()
    ref = c2 << c1
    ref.drotate(-180)
    route_fiber_array(
        c2,
        ref,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        cross_section=cross_section,
        select_ports=select_ports,
        with_loopback=False,
        port_names=output_port_names,
        pitch=pitch,
        **kwargs,
    )
    c2.copy_child_info(component)

    if with_loopback:
        straight_component = c2 << gf.get_component(
            straight, cross_section=cross_section, length=c2.dysize - 2 * gc.dxsize
        )
        gc1 = c2 << gc
        gc2 = c2 << gc

        straight_component.drotate(90)
        straight_component.dxmin = c2.dxmax + loopback_spacing
        straight_component.dymin = c2.dymin + gc1.dxsize

        gc1.connect(gc_port_name, straight_component.ports[0])
        gc2.connect(gc_port_name, straight_component.ports[1])

        c2.add_port(name="loopback1", port=gc1.ports[gc_port_name_fiber])
        c2.add_port(name="loopback2", port=gc2.ports[gc_port_name_fiber])

    return c2


if __name__ == "__main__":
    from gdsfactory.samples.big_device import big_device

    c = big_device(nports=1)
    c.info["polarization"] = "te"
    # c = gf.c.mmi2x2()
    c = add_fiber_single(c, gc_port_name_fiber="o3")
    c.pprint_ports()
    c.show()

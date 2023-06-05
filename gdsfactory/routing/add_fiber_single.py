from __future__ import annotations

from typing import Callable, Optional

import gdsfactory as gf
from gdsfactory.add_labels import (
    get_input_label_text_dash,
    get_input_label_text_dash_loopback,
    get_input_label_text_loopback,
)
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.functions import move_port_to_zero
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.get_input_labels import get_input_labels
from gdsfactory.routing.get_route import get_route_from_waypoints
from gdsfactory.routing.route_fiber_single import route_fiber_single
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec


@cell
def add_fiber_single(
    component: ComponentSpec = straight_function,
    grating_coupler=grating_coupler_te,
    layer_label: LayerSpec = "LABEL",
    fiber_spacing: float = 50,
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight_function,
    route_filter: Callable = get_route_from_waypoints,
    min_input_to_output_spacing: float = 200.0,
    optical_routing_type: int = 2,
    with_loopback: bool = True,
    loopback_xspacing: float = 50.0,
    component_name: Optional[str] = None,
    gc_port_name: str = "o1",
    zero_port: Optional[str] = "o1",
    get_input_label_text_loopback_function: Optional[
        Callable
    ] = get_input_label_text_dash_loopback,
    get_input_label_text_function: Optional[Callable] = get_input_label_text_dash,
    select_ports: Callable = select_ports_optical,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Returns component with grating couplers and labels on each port.

    It returns grating couplers in north-south orientation.
    First routes input port gc_port_name south, and other ports north.
    You can always rotate it for East-West orientation.

    Args:
        component: component or component function to connect to grating couplers.
        grating_coupler: grating coupler instance, function or list of functions.
        layer_label: for test and measurement label.
        fiber_spacing: between outputs.
        bend: bend spec.
        straight: straight sepc.
        route_filter: function to get route waypoints.
        min_input_to_output_spacing: spacing from input to output fiber (um).
        optical_routing_type: None: autoselection, 0: no extension.
        with_loopback: True, adds loopback reference straight waveguide.
        loopback_xspacing: spacing from loopback xmin to component.xmin.
        component_name: optional name of component.
        gc_port_name: grating coupler waveguide port name.
        zero_port: name of the port to move to (0, 0) for the routing to work correctly.
        get_input_label_text_loopback_function: for the loopbacks input label.
        get_input_label_text_function: for the grating couplers input label.
        get_input_labels_function: function to get input labels for grating couplers.
        select_ports: function to select ports.
        cross_section: cross_section spec.

    Keyword Args:
        max_y0_optical: in um.
        straight_separation: spacing between waveguides.
        list_port_labels: None, add labels to port indices in this list.
        connected_port_list_ids: None # only for type 0 optical routing.
        nb_optical_ports_lines: 1.
        force_manhattan: False.
        excluded_ports: list of ports to exclude.
        grating_indices: None.
        routing_method: function to ge the route.
        gc_rotation: grating_coupler rotation (deg).
        kwargs: cross_section settings.

    .. code::

        assumes grating coupler has o1 input port facing west at xmin = 0
             ___________
            /| | | | | |
           / | | | | | |
        o1|  | | | | | |
           \ | | | | | |
          | \|_|_|_|_|_|

          |
         xmin = 0


    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.crossing()
        cc = gf.routing.add_fiber_single(
            component=c,
            optical_routing_type=0,
            grating_coupler=gf.components.grating_coupler_elliptical_te,
        )
        cc.plot()

    """
    component = gf.get_component(component)

    optical_ports = select_ports(component.ports)
    optical_ports = list(optical_ports.values())
    optical_port_names = [p.name for p in optical_ports]

    zero_port = zero_port or optical_port_names[0]

    if not optical_ports:
        raise ValueError(f"No optical ports found in {component.name!r}")

    if zero_port not in optical_port_names:
        raise ValueError(f"zero_port = {zero_port!r} not in {optical_port_names}")

    component = move_port_to_zero(component, zero_port) if zero_port else component

    optical_ports = select_ports(component.ports)
    optical_ports = list(optical_ports.values())
    optical_port_names = [p.name for p in optical_ports]

    if not optical_ports:
        raise ValueError(f"No ports for {component.name}")

    component_name = component_name or component.metadata_child.get(
        "name", component.name
    )

    gc = (
        grating_coupler[0]
        if isinstance(grating_coupler, (list, tuple))
        else grating_coupler
    )
    gc = gf.get_component(gc)

    if gc_port_name not in gc.ports:
        raise ValueError(f"{gc_port_name!r} not in {list(gc.ports.keys())}")

    gc_port_orientation = int(gc.ports[gc_port_name].orientation)

    if gc_port_orientation != 180:
        raise ValueError(
            f"{gc_port_name!r} orientation {gc_port_orientation} needs to be 180 deg."
        )

    gc_port_to_edge = abs(gc.xmax - gc.ports[gc_port_name].center[0])

    c = Component()

    c.component = component
    cr = c << component
    cr.rotate(90)

    elements = []

    for port in cr.ports.values():
        if port.name not in optical_port_names:
            c.add_port(name=port.name, port=port)

    if (
        len(optical_ports) == 2
        and abs(optical_ports[0].x - optical_ports[1].x) > min_input_to_output_spacing
    ):
        grating_couplers = []
        for port in cr.ports.values():
            if port.name in optical_port_names:
                gc_ref = gc.ref()
                gc_ref.connect(gc_port_name, port)
                grating_couplers.append(gc_ref)

        if get_input_label_text_function and layer_label:
            elements = get_input_labels(
                io_gratings=grating_couplers,
                ordered_ports=list(cr.ports.values()),
                component_name=component_name,
                layer_label=layer_label,
                gc_port_name=gc_port_name,
                get_input_label_text_function=get_input_label_text_function,
            )

    else:
        elements, grating_couplers, ports_grating, ports_component = route_fiber_single(
            component,
            fiber_spacing=fiber_spacing,
            bend=bend,
            straight=straight,
            route_filter=route_filter,
            grating_coupler=gc,
            layer_label=layer_label,
            optical_routing_type=optical_routing_type,
            min_input_to_output_spacing=min_input_to_output_spacing,
            gc_port_name=gc_port_name,
            component_name=component_name,
            cross_section=cross_section,
            select_ports=select_ports,
            get_input_label_text_function=get_input_label_text_function,
            get_input_label_text_loopback_function=get_input_label_text_loopback,
            **kwargs,
        )

    for e in elements:
        c.add(e)
    for gc_ref in grating_couplers:
        c.add(gc_ref)

    for i, io_row in enumerate(grating_couplers):
        if isinstance(io_row, list):
            for j, io in enumerate(io_row):
                ports = io.get_ports_list(prefix="opt") or io.get_ports_list()

                if ports:
                    port = ports[0]
                    c.add_port(f"{port.name}-{component_name}-{i}-{j}", port=port)
        else:
            ports = io_row.get_ports_list(prefix="opt")
            if ports:
                port = ports[0]
                c.add_port(f"{port.name}-{component_name}-{i}-0", port=port)

    if with_loopback:
        length = c.ysize - 2 * gc_port_to_edge
        wg = c << gf.get_component(
            straight, length=length, cross_section=cross_section, **kwargs
        )
        wg.rotate(90)
        wg.xmax = c.xmin - loopback_xspacing
        wg.ymin = c.ymin + gc_port_to_edge

        gci = c << gc
        gco = c << gc
        gci.connect(gc_port_name, wg.ports["o1"])
        gco.connect(gc_port_name, wg.ports["o2"])

        port = wg.ports["o2"]
        ports = gc.get_ports_list(prefix="opt") or gc.get_ports_list()
        pname = ports[0].name

        c.add_port(name=f"{pname}-{component_name}-loopback1", port=gci.ports[pname])
        c.add_port(name=f"{pname}-{component_name}-loopback2", port=gco.ports[pname])
        if (
            layer_label
            and get_input_label_text_function
            and get_input_label_text_loopback_function
        ):
            text = get_input_label_text_loopback_function(
                port=port, gc=gc, gc_index=0, component_name=component_name
            )

            c.add_label(
                text=text,
                position=port.center,
                anchor="o",
                layer=layer_label,
            )

            port = wg.ports["o1"]
            text = get_input_label_text_loopback_function(
                port=port, gc=gc, gc_index=1, component_name=component_name
            )
            c.add_label(
                text=text,
                position=port.center,
                anchor="o",
                layer=layer_label,
            )

    c.copy_child_info(component)
    c.info["grating_coupler"] = gc.info
    return c


if __name__ == "__main__":
    # from gdsfactory.samples.big_device import big_device
    # w = h = 18 * 50
    # c = big_device(spacing=50.0, size=(w, h))
    # gc = gf.functions.rotate90(gf.components.grating_coupler_elliptical_arbitrary)

    gc = gf.components.grating_coupler_elliptical_arbitrary
    c = gf.c.mmi2x2()
    cc = gf.routing.add_fiber_single(
        component=c,
        grating_coupler=gc,
        # layer_label="TEXT"
        # layer_label=None,
    )
    cc.show(show_ports=False)

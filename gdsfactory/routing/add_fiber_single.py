from typing import Callable, Optional, Tuple

from gdsfactory.add_labels import get_input_label_text, get_input_label_text_loopback
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.config import TECH, call_if_func
from gdsfactory.cross_section import strip
from gdsfactory.functions import move_port_to_zero
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.get_input_labels import get_input_labels
from gdsfactory.routing.get_route import get_route_from_waypoints
from gdsfactory.routing.route_fiber_single import route_fiber_single
from gdsfactory.types import (
    ComponentFactory,
    ComponentOrFactory,
    ComponentOrFactoryOrList,
    CrossSectionFactory,
)


@cell
def add_fiber_single(
    component: ComponentOrFactory,
    grating_coupler: ComponentOrFactoryOrList = grating_coupler_te,
    layer_label: Tuple[int, int] = TECH.layer_label,
    fiber_spacing: float = TECH.fiber_spacing,
    bend: ComponentFactory = bend_circular,
    straight: ComponentFactory = straight_function,
    route_filter: Callable = get_route_from_waypoints,
    min_input_to_output_spacing: float = 200.0,
    optical_routing_type: int = 2,
    with_loopback: bool = True,
    component_name: Optional[str] = None,
    gc_port_name: str = "o1",
    zero_port: Optional[str] = "o1",
    get_input_label_text_loopback_function: Callable = get_input_label_text_loopback,
    get_input_label_text_function: Callable = get_input_label_text,
    select_ports: Callable = select_ports_optical,
    cross_section: CrossSectionFactory = strip,
    **kwargs,
) -> Component:
    r"""Returns component with grating ports and labels on each port.

    Can add loopback reference structure next to it.

    Args:
        component: to connect
        grating_coupler: grating coupler instance, function or list of functions
        layer_label: for test and measurement label
        fiber_spacing: between outputs
        bend: bend_circular
        straight: straight
        route_filter:
        min_input_to_output_spacing: spacing from input to output fiber
        max_y0_optical: None
        with_loopback: True, adds loopback structures
        straight_separation: 4.0
        list_port_labels: None, add labels to port indices in this list
        connected_port_list_ids: None # only for type 0 optical routing
        nb_optical_ports_lines: 1
        force_manhattan: False
        excluded_ports: list of ports to exclude
        grating_indices: None
        routing_method: function to ge the route
        gc_port_name: grating coupler name
        zero_port: name of the port to move to (0, 0) for the routing to work correctly
        get_input_labels_function: function to get input labels for grating couplers
        optical_routing_type: None: autoselection, 0: no extension
        gc_rotation: grating_coupler rotation (deg)
        component_name: name of component
        cross_section:
        get_input_label_text_function: for the grating couplers input label
        get_input_label_text_loopback_function: for the loopacks input label
        kwargs: cross_section settings

    .. code::

        asumes grating coupler has o1 input port facing west at xmin = 0
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
    component = component() if callable(component) else component
    component = move_port_to_zero(component, zero_port) if zero_port else component

    optical_ports = select_ports(component.ports)
    optical_ports = list(optical_ports.values())
    optical_port_names = [p.name for p in optical_ports]

    if not optical_ports:
        raise ValueError(f"No ports for {component.name}")

    component = component() if callable(component) else component

    component_name = component_name or component.info_child.get("name", component.name)

    gc = (
        grating_coupler[0]
        if isinstance(grating_coupler, (list, tuple))
        else grating_coupler
    )
    gc = gc() if callable(gc) else gc

    if gc_port_name not in gc.ports:
        raise ValueError(f"{gc_port_name} not in {list(gc.ports.keys())}")

    gc_port_to_edge = abs(gc.xmax - gc.ports[gc_port_name].midpoint[0])

    c = Component()

    c.component = component
    cr = c << component
    cr.rotate(90)

    for port in cr.ports.values():
        if port.name not in optical_port_names:
            c.add_port(name=port.name, port=port)

    if (
        len(optical_ports) == 2
        and abs(optical_ports[0].x - optical_ports[1].x) > min_input_to_output_spacing
    ):

        grating_coupler = call_if_func(grating_coupler)
        grating_couplers = []
        for port in cr.ports.values():
            if port.name in optical_port_names:
                gc_ref = grating_coupler.ref()
                gc_ref.connect(gc_port_name, port)
                grating_couplers.append(gc_ref)

        elements = get_input_labels(
            io_gratings=grating_couplers,
            ordered_ports=list(cr.ports.values()),
            component_name=component_name,
            layer_label=layer_label,
            gc_port_name=gc_port_name,
            get_input_label_text_function=get_input_label_text_function,
        )

    else:
        elements, grating_couplers = route_fiber_single(
            component,
            fiber_spacing=fiber_spacing,
            bend=bend,
            straight=straight,
            route_filter=route_filter,
            grating_coupler=grating_coupler,
            layer_label=layer_label,
            optical_routing_type=optical_routing_type,
            min_input_to_output_spacing=min_input_to_output_spacing,
            gc_port_name=gc_port_name,
            component_name=component_name,
            cross_section=cross_section,
            select_ports=select_ports,
            **kwargs,
        )

    for e in elements:
        c.add(e)
    for gc in grating_couplers:
        c.add(gc)

    for i, io_row in enumerate(grating_couplers):
        if isinstance(io_row, list):
            for j, io in enumerate(io_row):
                ports = io.get_ports_list(prefix="vertical")
                if ports:
                    port = ports[0]
                    c.add_port(f"{port.name}_{i}{j}", port=port)
        else:
            ports = io_row.get_ports_list(prefix="vertical")
            if ports:
                port = ports[0]
                c.add_port(f"{port.name}_{i}", port=port)

    if isinstance(grating_coupler, (list, tuple)):
        grating_coupler = grating_coupler[0]

    grating_coupler = call_if_func(grating_coupler)
    if with_loopback:
        length = c.ysize - 2 * gc_port_to_edge
        wg = c << straight(length=length, cross_section=cross_section, **kwargs)
        wg.rotate(90)
        wg.xmax = (
            c.xmin - fiber_spacing
            if abs(c.xmin) > abs(fiber_spacing)
            else c.xmin - fiber_spacing
        )
        wg.ymin = c.ymin + gc_port_to_edge

        gci = c << grating_coupler
        gco = c << grating_coupler
        gci.connect(gc_port_name, wg.ports["o1"])
        gco.connect(gc_port_name, wg.ports["o2"])

        port = wg.ports["o2"]
        text = get_input_label_text_loopback_function(
            port=port, gc=grating_coupler, gc_index=0, component_name=component_name
        )

        c.add_label(
            text=text,
            position=port.midpoint,
            anchor="o",
            layer=layer_label,
        )

        port = wg.ports["o1"]
        text = get_input_label_text_loopback_function(
            port=port, gc=grating_coupler, gc_index=1, component_name=component_name
        )
        c.add_label(
            text=text,
            position=port.midpoint,
            anchor="o",
            layer=layer_label,
        )

    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.components.crossing()
    # c = gf.components.mmi1x2()
    # c = gf.components.rectangle()
    # c = gf.components.mzi()
    # c = gf.components.straight(length=500)
    # gc = gf.components.grating_coupler_elliptical_te
    # gc = gf.components.grating_coupler_circular
    # gc = gf.components.grating_coupler_te
    # gc = gf.components.grating_coupler_rectangular

    @gf.cell
    def component_with_offset(**kwargs):
        c = gf.Component()
        ref = c << gf.components.mmi1x2(**kwargs)
        ref.movey(1)
        c.add_ports(ref.ports)
        return c

    # c = gf.components.ring_single(length_x=167)
    # c = gf.components.spiral(direction="NORTH")
    c = gf.c.spiral_inner_io_fiber_single()
    cc = add_fiber_single(
        # component=gf.c.straight_heater_metal(width=2),
        component=c,
        auto_widen=False,
        with_loopback=True,
        layer=(2, 0),
        zero_port="o2",
        # grating_coupler=[gf.c.grating_coupler_te, gf.c.grating_coupler_tm],
    )
    cc.show()

    # c = gf.components.straight(length=20)
    # gc = gf.components.grating_coupler_elliptical_te(layer=gf.TECH.layer.WGN)
    # cc = add_fiber_single(component=c, grating_coupler=gc, with_loopback=True, )
    # cc.show()

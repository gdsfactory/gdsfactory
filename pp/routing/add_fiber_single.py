from typing import Callable, Optional, Tuple

import phidl.device_layout as pd

from pp.add_labels import get_optical_text
from pp.add_tapers import add_tapers
from pp.cell import cell
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_te
from pp.components.straight import straight
from pp.components.taper import taper
from pp.config import call_if_func
from pp.layers import LAYER
from pp.routing.get_input_labels import get_input_labels
from pp.routing.get_route import get_route_from_waypoints
from pp.routing.route_fiber_single import route_fiber_single
from pp.types import ComponentFactory


@cell
def add_fiber_single(
    component: Component,
    grating_coupler: ComponentFactory = grating_coupler_te,
    layer_label: Tuple[int, int] = LAYER.LABEL,
    optical_io_spacing: float = 50.0,
    bend_factory: ComponentFactory = bend_circular,
    straight_factory: ComponentFactory = straight,
    taper_factory: ComponentFactory = taper,
    taper_length: float = 10.0,
    route_filter: Callable = get_route_from_waypoints,
    min_input2output_spacing: float = 200.0,
    optical_routing_type: int = 2,
    with_align_ports: bool = True,
    component_name: Optional[str] = None,
    gc_port_name: str = "W0",
    get_input_labels_function: Callable = get_input_labels,
    auto_widen: bool = False,
    **kwargs,
) -> Component:
    r"""Returns component with grating ports and labels on each port.

    Can add align_ports reference structure next to it.

    Args:
        component: to connect
        grating_coupler: grating coupler instance, function or list of functions
        layer_label: LAYER.LABEL
        optical_io_spacing: SPACING_GC
        bend_factory: bend_circular
        straight_factory: straight
        taper_factory: taper
        fanout_length: None  # if None, automatic calculation of fanout length
        max_y0_optical: None
        with_align_ports: True, adds loopback structures
        straight_separation: 4.0
        bend_radius: BEND_RADIUS
        list_port_labels: None, adds TM labels to port indices in this list
        connected_port_list_ids: None # only for type 0 optical routing
        nb_optical_ports_lines: 1
        force_manhattan: False
        excluded_ports:
        grating_indices: None
        routing_method: get_route
        gc_port_name: W0
        get_input_labels_function: function to get input labels for grating couplers
        optical_routing_type: None: autoselection, 0: no extension
        gc_rotation: -90
        component_name: name of component
        auto_widen: for lower loss
        auto_widen: widen straight waveguides for lower loss in long routes

    .. code::

              fiber
             ______
            /| | |
           / | | |
        W0|  | | |
           \ | | |
          | \|_|_|_

          |
         xmin = 0

    .. plot::
        :include-source:

        import pp

        c = pp.components.crossing()
        cc = pp.routing.add_fiber_single(
            component=c,
            optical_routing_type=0,
            grating_coupler=pp.components.grating_coupler_elliptical_te,
            bend_radius=5
        )
        cc.plot()

    """
    if not component.get_ports_list(port_type="optical"):
        raise ValueError(f"No ports for {component.name}")

    component = component() if callable(component) else component
    gc = grating_coupler = (
        grating_coupler() if callable(grating_coupler) else grating_coupler
    )
    gc_port_to_edge = abs(gc.xmax - gc.ports[gc_port_name].midpoint[0])
    port_width_gc = grating_coupler.ports[gc_port_name].width
    optical_ports = component.get_ports_list(port_type="optical")
    port_width_component = optical_ports[0].width

    if port_width_component != port_width_gc:
        taper = (
            taper_factory(
                length=taper_length, width1=port_width_gc, width2=port_width_component
            )
            if callable(taper_factory)
            else taper_factory
        )
        component = add_tapers(component=component, taper=taper)

    component_name = component_name or component.name
    name = f"{component_name}_{grating_coupler.name}"

    c = Component(name=name)
    cr = c << component
    cr.rotate(90)

    if (
        len(optical_ports) == 2
        and abs(optical_ports[0].x - optical_ports[1].x) > min_input2output_spacing
    ):

        grating_coupler = call_if_func(grating_coupler)
        grating_couplers = []
        for port in cr.ports.values():
            gc_ref = grating_coupler.ref()
            gc_ref.connect(list(gc_ref.ports.values())[0], port)
            grating_couplers.append(gc_ref)

        elements = get_input_labels(
            grating_couplers,
            list(cr.ports.values()),
            component_name=component.name,
            layer_label=layer_label,
            gc_port_name=gc_port_name,
        )

    else:
        elements, grating_couplers = route_fiber_single(
            component,
            component_name=component_name,
            optical_io_spacing=optical_io_spacing,
            bend_factory=bend_factory,
            straight_factory=straight_factory,
            route_filter=route_filter,
            grating_coupler=grating_coupler,
            layer_label=layer_label,
            optical_routing_type=optical_routing_type,
            min_input2output_spacing=min_input2output_spacing,
            gc_port_name=gc_port_name,
            auto_widen=auto_widen,
            **kwargs,
        )

    for e in elements:
        c.add(e)
    for gc in grating_couplers:
        c.add(gc)

    for pname, p in component.ports.items():
        if p.port_type != "optical":
            c.add_port(pname, port=p)

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

    if isinstance(grating_coupler, list):
        grating_couplers = [call_if_func(g) for g in grating_coupler]
        grating_coupler = grating_couplers[0]
    else:
        grating_coupler = call_if_func(grating_coupler)
        grating_couplers = [grating_coupler]

    if with_align_ports:
        length = c.ysize - 2 * gc_port_to_edge
        wg = c << straight_factory(length=length)
        wg.rotate(90)
        wg.xmax = (
            c.xmin - optical_io_spacing
            if abs(c.xmin) > abs(optical_io_spacing)
            else c.xmin - optical_io_spacing
        )
        wg.ymin = c.ymin + gc_port_to_edge

        gci = c << grating_coupler
        gco = c << grating_coupler
        gci.connect(gc_port_name, wg.ports["W0"])
        gco.connect(gc_port_name, wg.ports["E0"])

        gds_layer_label, gds_datatype_label = pd._parse_layer(layer_label)

        port = wg.ports["E0"]
        text = get_optical_text(
            port, grating_coupler, 0, component_name=f"loopback_{component.name}"
        )

        label = pd.Label(
            text=text,
            position=port.midpoint,
            anchor="o",
            layer=gds_layer_label,
            texttype=gds_datatype_label,
        )
        c.add(label)

        port = wg.ports["W0"]
        text = get_optical_text(
            port, grating_coupler, 1, component_name=f"loopback_{component.name}"
        )
        label = pd.Label(
            text=text,
            position=port.midpoint,
            anchor="o",
            layer=gds_layer_label,
            texttype=gds_datatype_label,
        )
        c.add(label)

    return c


if __name__ == "__main__":
    import pp

    c = pp.components.crossing()
    c = pp.components.ring_double(length_y=3)  # FIXME
    c = pp.components.straight(length=500)
    c = pp.components.mmi1x2()
    c = pp.components.rectangle()
    c = pp.components.mzi(length_x=50)
    c = pp.components.ring_single()
    c = pp.components.mzi2x2(with_elec_connections=True)

    gc = pp.components.grating_coupler_elliptical_te
    # gc = pp.components.grating_coupler_elliptical2
    # gc = pp.components.grating_coupler_te
    # gc = pp.components.grating_coupler_uniform

    cc = add_fiber_single(component=c, grating_coupler=gc, with_align_ports=True)
    cc = add_fiber_single(component=c, auto_widen=False)

    # print(cc.get_settings()["component"])
    print(cc.ports.keys())
    cc.show()

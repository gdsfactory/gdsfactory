from typing import Callable, Tuple
import phidl.device_layout as pd
from pp.config import call_if_func
from pp.layers import LAYER
from pp.component import Component
from pp.add_labels import get_optical_text
from pp.routing.route_fiber_single import route_fiber_single
from pp.routing.connect import connect_strip_way_points

from pp.components.bend_circular import bend_circular
from pp.components import waveguide
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_te
from pp.components.taper import taper
from pp.container import container


@container
def add_fiber_single(
    component: Component,
    grating_coupler: Callable = grating_coupler_te,
    layer_label: Tuple[int, int] = LAYER.LABEL,
    optical_io_spacing: int = 50,
    bend_factory: Callable = bend_circular,
    straight_factory: Callable = waveguide,
    taper_factory: Callable = taper,
    route_filter: Callable = connect_strip_way_points,
    min_input2output_spacing: int = 127,
    optical_routing_type: int = 2,
    with_align_ports: bool = True,
    component_name=None,
    **kwargs,
) -> Component:
    """returns component with grating ports and labels on each port
    can add align_ports reference structure

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
        component_name: name of component
        taper_factory: taper

    """
    component = component() if callable(component) else component
    grating_coupler = (
        grating_coupler() if callable(grating_coupler) else grating_coupler
    )

    elements, grating_couplers, _ = route_fiber_single(
        component,
        optical_io_spacing=optical_io_spacing,
        bend_factory=bend_factory,
        straight_factory=straight_factory,
        route_filter=route_filter,
        grating_coupler=grating_coupler,
        layer_label=layer_label,
        taper_factory=taper_factory,
        optical_routing_type=optical_routing_type,
        min_input2output_spacing=min_input2output_spacing,
        **kwargs,
    )

    component_name = component_name or component.name
    name = f"{component_name}_{grating_coupler.polarization}"
    c = Component(name=name)
    cr = c << component
    cr.rotate(90)
    c.function_name = "add_io_optical"

    for e in elements:
        c.add(e)
    for gc in grating_couplers:
        c.add(gc)

    if isinstance(grating_coupler, list):
        grating_couplers = [call_if_func(g) for g in grating_coupler]
        grating_coupler = grating_couplers[0]
    else:
        grating_coupler = call_if_func(grating_coupler)
        grating_couplers = [grating_coupler]

    if with_align_ports:
        gc_port_name = list(grating_coupler.ports.keys())[0]

        length = c.ysize - 2 * grating_coupler.xsize
        wg = c << straight_factory(length=length)
        wg.rotate(90)
        wg.xmax = (
            c.xmin - optical_io_spacing
            if abs(c.xmin) > abs(optical_io_spacing)
            else c.xmin - optical_io_spacing
        )
        wg.ymin = c.ymin + grating_coupler.xsize

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

    c = pp.c.crossing()
    c = pp.c.ring_double()  # FIXME
    c = pp.c.mmi2x2()
    cc = add_fiber_single(c)
    pp.show(cc)

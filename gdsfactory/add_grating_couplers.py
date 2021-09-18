"""Add grating_couplers to a component."""
from typing import Callable, List, Optional, Tuple

import numpy as np
from phidl.device_layout import Label

from gdsfactory.add_labels import get_input_label_text_loopback
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.get_input_labels import get_input_labels
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.routing.utils import (
    check_ports_have_equal_spacing,
    direction_ports_from_list_ports,
)
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@cell
def add_grating_couplers(
    component: Component,
    grating_coupler: ComponentFactory = grating_coupler_te,
    layer_label: Tuple[int, int] = (200, 0),
    gc_port_name: str = "o1",
    get_input_labels_function: Callable[..., List[Label]] = get_input_labels,
    select_ports: Callable = select_ports_optical,
) -> Component:
    """Returns new component with grating couplers and labels.

    Args:
        component: to add grating_couplers
        grating_coupler: grating_coupler function
        layer_label: for label
        gc_port_name: where to add label
        get_input_labels_function: function to get label
        select_ports: for selecting optical_ports

    """

    c = Component()
    c.component = component
    c.add_ref(component)
    grating_coupler = (
        grating_coupler() if callable(grating_coupler) else grating_coupler
    )

    io_gratings = []
    optical_ports = select_ports(component.ports)
    optical_ports = list(optical_ports.values())
    for port in optical_ports:
        gc_ref = grating_coupler.ref()
        gc_port = gc_ref.ports[gc_port_name]
        gc_ref.connect(gc_port, port)
        io_gratings.append(gc_ref)
        c.add(gc_ref)

    labels = get_input_labels_function(
        io_gratings,
        list(component.ports.values()),
        component_name=component.name,
        layer_label=layer_label,
        gc_port_name=gc_port_name,
    )
    c.add(labels)
    return c


@cell
def add_grating_couplers_with_loopback_fiber_single(
    component: Component,
    grating_coupler: ComponentFactory = grating_coupler_te,
    layer_label: Tuple[int, int] = (200, 0),
    gc_port_name: str = "o1",
    get_input_labels_function: Callable[..., List[Label]] = get_input_labels,
    get_input_label_text_loopback_function: Callable = get_input_label_text_loopback,
    select_ports: Callable = select_ports_optical,
    with_loopback: bool = True,
    cross_section: CrossSectionFactory = strip,
    component_name: Optional[str] = None,
    fiber_spacing: float = 50.0,
    loopback_xspacing: float = 5.0,
    straight: ComponentFactory = straight_function,
    rotation: int = 90,
) -> Component:
    """
    Returns component with all ports terminated with grating couplers

    Args:
        component:
        grating_coupler:
        layer_label:
        gc_port_name:
        get_input_label_text_loopback
        with_loopback: adds a reference loopback
        rotation: 90 for North South devices, 0 for East-West

    """

    c = Component()
    c.component = component
    c.add_ref(component)
    grating_coupler = (
        grating_coupler() if callable(grating_coupler) else grating_coupler
    )

    component_name = component_name or component.name

    io_gratings = []
    optical_ports = select_ports(component.ports)
    optical_ports = list(optical_ports.values())
    for port in optical_ports:
        gc_ref = grating_coupler.ref()
        gc_port = gc_ref.ports[gc_port_name]
        gc_ref.connect(gc_port, port)
        io_gratings.append(gc_ref)
        c.add(gc_ref)

    labels = get_input_labels_function(
        io_gratings,
        list(component.ports.values()),
        component_name=component_name,
        layer_label=layer_label,
        gc_port_name=gc_port_name,
    )
    c.add(labels)

    p2 = optical_ports[0]
    p1 = optical_ports[-1]

    if with_loopback:

        if rotation in [0, 180]:
            length = abs(p2.x - p1.x)
            wg = c << straight(length=length, cross_section=cross_section)
            wg.rotate(rotation)
            wg.xmin = p2.x
            wg.ymin = c.ymax + grating_coupler.ysize / 2 + loopback_xspacing
        else:
            length = abs(p2.y - p1.y)
            wg = c << straight(length=length, cross_section=cross_section)
            wg.rotate(rotation)
            wg.ymin = p1.y
            wg.xmin = c.xmax + grating_coupler.ysize / 2 + loopback_xspacing

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

    return c


@cell
def add_grating_couplers_with_loopback_fiber_array(
    component: Component,
    grating_coupler: ComponentFactory = grating_coupler_te,
    excluded_ports: None = None,
    grating_separation: float = 127.0,
    bend_radius_loopback: Optional[float] = None,
    gc_port_name: str = "o1",
    gc_rotation: int = -90,
    straight_separation: float = 5.0,
    bend_factory: ComponentFactory = bend_euler,
    straight_factory: ComponentFactory = straight_function,
    layer_label: Tuple[int, int] = (200, 0),
    layer_label_loopback: Optional[Tuple[int, int]] = None,
    component_name: None = None,
    with_loopback: bool = True,
    nlabels_loopback: int = 2,
    get_input_labels_function: Callable = get_input_labels,
    cross_section: CrossSectionFactory = strip,
    select_ports: Callable = select_ports_optical,
    **kwargs,
) -> Component:
    """Returns a component with grating_couplers and loopback.

    Args:
        component:
        grating_coupler:
        excluded_ports:
        grating_separation:
        bend_radius_loopback:
        gc_port_name:
        gc_rotation:
        straight_separation:
        bend_factory:
        straight_factory:
        layer_label:
        component_name:
        with_loopback: If True, add compact loopback alignment ports
        nlabels_loopback: number of ports to label (0: no labels, 1: first port, 2: both ports)
        cross_section:
        **kwargs: cross_section settings
    """
    x = cross_section(**kwargs)
    bend_radius_loopback = bend_radius_loopback or x.info["radius"]
    excluded_ports = excluded_ports or []
    gc = grating_coupler() if callable(grating_coupler) else grating_coupler

    direction = "S"
    component_name = component_name or component.name
    c = Component()
    c.component = component

    c.add_ref(component)

    # Find grating port name if not specified
    if gc_port_name is None:
        gc_port_name = list(gc.ports.values())[0].name

    # List the optical ports to connect
    optical_ports = select_ports(component.ports)
    optical_ports = list(optical_ports.values())
    optical_ports = [p for p in optical_ports if p.name not in excluded_ports]
    optical_ports = direction_ports_from_list_ports(optical_ports)[direction]

    # Check if the ports are equally spaced
    grating_separation_extracted = check_ports_have_equal_spacing(optical_ports)
    if grating_separation_extracted != grating_separation:
        raise ValueError(
            "Grating separation must be {}. Got {}".format(
                grating_separation, grating_separation_extracted
            )
        )

    # Add grating references
    references = []
    for port in optical_ports:
        gc_ref = c.add_ref(gc)
        gc_ref.connect(gc.ports[gc_port_name].name, port)
        references += [gc_ref]

    labels = get_input_labels_function(
        io_gratings=references,
        ordered_ports=optical_ports,
        component_name=component_name,
        layer_label=layer_label,
        gc_port_name=gc_port_name,
    )
    c.add(labels)

    if with_loopback:
        y0 = references[0].ports[gc_port_name].y
        xs = [p.x for p in optical_ports]
        x0 = min(xs) - grating_separation
        x1 = max(xs) + grating_separation

        gca1, gca2 = [
            gc.ref(position=(x, y0), rotation=gc_rotation, port_id=gc_port_name)
            for x in [x0, x1]
        ]

        gsi = gc.size_info
        port0 = gca1.ports[gc_port_name]
        port1 = gca2.ports[gc_port_name]
        p0 = port0.position
        p1 = port1.position
        a = bend_radius_loopback + 0.5
        b = max(2 * a, grating_separation / 2)
        y_bot_align_route = -gsi.width - straight_separation

        points = np.array(
            [
                p0,
                p0 + (0, a),
                p0 + (b, a),
                p0 + (b, y_bot_align_route),
                p1 + (-b, y_bot_align_route),
                p1 + (-b, a),
                p1 + (0, a),
                p1,
            ]
        )
        bend90 = bend_factory(
            radius=bend_radius_loopback, cross_section=cross_section, **kwargs
        )
        loopback_route = round_corners(
            points=points,
            bend_factory=bend90,
            straight_factory=straight_factory,
            cross_section=cross_section,
            **kwargs,
        )
        c.add([gca1, gca2])
        c.add(loopback_route.references)

        component_name_loopback = f"loopback_{component_name}"
        if nlabels_loopback == 1:
            io_gratings_loopback = [gca1]
            ordered_ports_loopback = [port0]
        if nlabels_loopback == 2:
            io_gratings_loopback = [gca1, gca2]
            ordered_ports_loopback = [port0, port1]
        if nlabels_loopback == 0:
            pass
        elif 0 < nlabels_loopback <= 2:
            c.add(
                get_input_labels_function(
                    io_gratings=io_gratings_loopback,
                    ordered_ports=ordered_ports_loopback,
                    component_name=component_name_loopback,
                    layer_label=layer_label_loopback or layer_label,
                    gc_port_name=gc_port_name,
                )
            )
        else:
            raise ValueError(
                f"Invalid nlabels_loopback = {nlabels_loopback}, "
                "valid (0: no labels, 1: first port, 2: both ports2)"
            )
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    # from gdsfactory.add_labels import get_optical_text
    # c = gf.components.grating_coupler_elliptical_te()
    # print(c.wavelength)
    # print(c.get_property('wavelength'))
    # c = gf.components.straight()
    c = gf.components.mzi_phase_shifter()
    c = add_grating_couplers_with_loopback_fiber_single(component=c, rotation=0)
    # c = gf.c.spiral_inner_io()
    # c = add_grating_couplers_with_loopback_fiber_array(component=c)
    # c = add_grating_couplers(c)

    c.show()

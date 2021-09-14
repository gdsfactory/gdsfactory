"""Add grating_couplers to a component."""
from typing import Callable, List, Optional, Tuple

from phidl.device_layout import Label

import gdsfactory as gf
from gdsfactory.add_labels import get_input_label_text_loopback
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.components.straight import straight
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.get_input_labels import get_input_labels
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@cell
def add_grating_couplers(
    component: Component,
    grating_coupler: ComponentFactory = grating_coupler_te,
    layer_label: Tuple[int, int] = gf.LAYER.LABEL,
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
    grating_coupler = gf.call_if_func(grating_coupler)

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


def add_grating_couplers_and_loopback_fiber_single(
    component: Component,
    grating_coupler: ComponentFactory = grating_coupler_te,
    layer_label: Tuple[int, int] = gf.LAYER.LABEL,
    gc_port_name: str = "o1",
    get_input_labels_function: Callable[..., List[Label]] = get_input_labels,
    get_input_label_text_loopback_function: Callable = get_input_label_text_loopback,
    select_ports: Callable = select_ports_optical,
    with_loopback: bool = True,
    cross_section: CrossSectionFactory = strip,
    component_name: Optional[str] = None,
    fiber_spacing: float = 50.0,
    loopback_xspacing: float = 5.0,
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

    """

    c = Component()
    c.component = component
    c.add_ref(component)
    grating_coupler = gf.call_if_func(grating_coupler)

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
        length = p2.y - p1.y
        wg = c << straight(length=length, cross_section=cross_section)
        wg.rotate(90)
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


if __name__ == "__main__":
    # from gdsfactory.add_labels import get_optical_text
    # c = gf.components.grating_coupler_elliptical_te()
    # print(c.wavelength)

    # print(c.get_property('wavelength'))

    c = gf.components.spiral_inner_io_fiber_single(width=2)
    cc = add_grating_couplers_and_loopback_fiber_single(component=c)
    cc.show()

"""Add grating_couplers to a component."""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.add_labels import get_input_label_text_loopback
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.components.spiral_inner_io import (
    spiral_inner_io,
    spiral_inner_io_fiber_single,
)
from gdsfactory.components.straight import straight
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.get_input_labels import get_input_labels
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.routing.utils import (
    check_ports_have_equal_spacing,
    direction_ports_from_list_ports,
)
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Label, PortsDict


@cell
def add_grating_couplers(
    component: ComponentSpec = straight,
    grating_coupler: ComponentSpec = grating_coupler_te,
    layer_label: Tuple[int, int] = (200, 0),
    gc_port_name: str = "o1",
    get_input_labels_function: Callable[..., List[Label]] = get_input_labels,
    select_ports: Callable[..., PortsDict] = select_ports_optical,
    component_name: Optional[str] = None,
) -> Component:
    """Returns new component with grating couplers and labels.

    Args:
        component: to add grating_couplers.
        grating_coupler: grating_coupler spec.
        layer_label: for label.
        gc_port_name: where to add label.
        get_input_labels_function: function to get label.
        select_ports: for selecting optical_ports.
        component_name: optional component name.
    """
    c = Component()
    component = gf.get_component(component)

    c.component = component
    component_name = component_name or component.metadata_child.get("name")
    c.add_ref(component)
    grating_coupler = gf.get_component(grating_coupler)

    c.info["polarization"] = grating_coupler.info["polarization"]
    c.info["wavelength"] = grating_coupler.info["wavelength"]

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
    c.copy_child_info(component)
    return c


@cell
def add_grating_couplers_with_loopback_fiber_single(
    component: ComponentSpec = spiral_inner_io_fiber_single,
    grating_coupler: ComponentSpec = grating_coupler_te,
    layer_label: Optional[Tuple[int, int]] = (200, 0),
    gc_port_name: str = "o1",
    get_input_labels_function: Callable[..., List[Label]] = get_input_labels,
    get_input_label_text_loopback_function: Callable = get_input_label_text_loopback,
    select_ports: Callable[..., PortsDict] = select_ports_optical,
    with_loopback: bool = True,
    cross_section: CrossSectionSpec = strip,
    component_name: Optional[str] = None,
    loopback_xspacing: float = 5.0,
    rotation: int = 90,
) -> Component:
    """Returns new component with all ports terminated with grating couplers.

    Args:
        component: to add grating_couplers.
        grating_coupler: grating_coupler spec function, string or dict.
        layer_label: optional layer_label for the ports.
        gc_port_name: grating_coupler port name.
        get_input_labels_function: function to get grating_coupler labels.
        get_input_label_text_loopback_function:
        select_ports: function to select ports.
        with_loopback: adds a reference loopback.
        cross_section: for routing.
        component_name: optional component name.
        loopback_xspacing: in um.
        rotation: in degrees, 90 for North South devices, 0 for East-West.
    """
    c = Component()
    component = gf.get_component(component)

    c.component = component
    c.add_ref(component)
    grating_coupler = gf.get_component(grating_coupler)

    c.info["polarization"] = grating_coupler.info["polarization"]
    c.info["wavelength"] = grating_coupler.info["wavelength"]
    component_name = component_name or component.metadata_child.get("name")

    io_gratings = []
    optical_ports = select_ports(component.ports)
    optical_ports = list(optical_ports.values())

    for port in optical_ports:
        gc_ref = grating_coupler.ref()
        gc_port = gc_ref.ports[gc_port_name]
        gc_ref.connect(gc_port, port)
        io_gratings.append(gc_ref)
        c.add(gc_ref)
        c.add_port(name=port.name, port=port)

    labels = get_input_labels_function(
        io_gratings,
        optical_ports,
        component_name=component_name,
        layer_label=layer_label,
        gc_port_name=gc_port_name,
    )
    c.add(labels)

    p2 = optical_ports[0]
    p1 = optical_ports[-1]

    if with_loopback:
        if rotation in {0, 180}:
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

        p1 = c.add_port(name="loopback1", port=wg.ports["o1"])
        p2 = c.add_port(name="loopback2", port=wg.ports["o2"])
        p1.port_type = "loopback"
        p2.port_type = "loopback"

        if layer_label and get_input_labels_function:
            port = wg.ports["o2"]
            text = get_input_label_text_loopback_function(
                port=port, gc=grating_coupler, gc_index=0, component_name=component_name
            )

            c.add_label(
                text=text,
                position=port.center,
                anchor="o",
                layer=layer_label,
            )

            port = wg.ports["o1"]
            text = get_input_label_text_loopback_function(
                port=port, gc=grating_coupler, gc_index=1, component_name=component_name
            )
            c.add_label(
                text=text,
                position=port.center,
                anchor="o",
                layer=layer_label,
            )

    c.copy_child_info(component)
    return c


@cell
def add_grating_couplers_with_loopback_fiber_array(
    component: ComponentSpec = spiral_inner_io,
    grating_coupler: ComponentSpec = grating_coupler_te,
    excluded_ports: Optional[List[str]] = None,
    grating_separation: float = 127.0,
    bend_radius_loopback: Optional[float] = None,
    gc_port_name: str = "o1",
    gc_rotation: int = -90,
    straight_separation: float = 5.0,
    bend: ComponentSpec = bend_euler,
    layer_label: Tuple[int, int] = (200, 0),
    layer_label_loopback: Optional[Tuple[int, int]] = None,
    component_name: Optional[str] = None,
    with_loopback: bool = False,
    nlabels_loopback: int = 2,
    get_input_labels_function: Callable = get_input_labels,
    cross_section: CrossSectionSpec = strip,
    select_ports: Callable = select_ports_optical,
    loopback_yspacing: float = 4.0,
    **kwargs,
) -> Component:
    """Returns a component with grating_couplers and loopback.

    Args:
        component: to add grating_couplers.
        grating_coupler: grating_coupler.
        excluded_ports: list of ports to exclude.
        grating_separation: in um.
        bend_radius_loopback: um.
        gc_port_name: optional grating coupler name.
        gc_rotation: grating coupler rotation in degrees.
        straight_separation: in um.
        bend: bend spec.
        layer_label: for testing label.
        layer_label_loopback: for testing label alignment loopback.
        component_name: optional component name.
        with_loopback: If True, add compact loopback alignment ports.
        nlabels_loopback: number of ports to label
            (0: no labels, 1: first port, 2: both ports).
        get_input_labels_function: for getting test labels.
        cross_section: CrossSectionSpec.
        select_ports: function to select ports.
        loopback_yspacing: in um.
        kwargs: cross_section settings.
    """
    component = gf.get_component(component)
    x = gf.get_cross_section(cross_section, **kwargs)
    bend_radius_loopback = bend_radius_loopback or x.radius
    excluded_ports = excluded_ports or []
    gc = gf.get_component(grating_coupler)

    direction = "S"
    component_name = component_name or component.metadata_child.get("name")

    c = Component()
    c.component = component
    c.info["polarization"] = gc.info["polarization"]
    c.info["wavelength"] = gc.info["wavelength"]

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
            f"Grating separation must be {grating_separation}. "
            f"Got {grating_separation_extracted}"
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
        y0 = references[0].ports[gc_port_name].y - loopback_yspacing
        xs = [p.x for p in optical_ports]
        x0 = min(xs) - grating_separation
        x1 = max(xs) + grating_separation

        gca1, gca2 = (
            gc.ref(position=(x, y0), rotation=gc_rotation, port_id=gc_port_name)
            for x in [x0, x1]
        )

        gsi = gc.size_info
        port0 = gca1.ports[gc_port_name]
        port1 = gca2.ports[gc_port_name]
        p0 = port0.center
        p1 = port1.center
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
        bend90 = gf.get_component(
            bend, radius=bend_radius_loopback, cross_section=cross_section, **kwargs
        )
        loopback_route = round_corners(
            points=points,
            bend=bend90,
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
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    # from gdsfactory.add_labels import get_optical_text
    # c = gf.components.grating_coupler_elliptical_te()
    # print(c.wavelength)
    # print(c.get_property('wavelength'))
    # c = gf.components.straight()
    # c = gf.components.mzi_phase_shifter()
    # c = add_grating_couplers_with_loopback_fiber_single(component=c, rotation=0)
    # c = gf.components.spiral_inner_io()
    # c = add_grating_couplers_with_loopback_fiber_array(component=c)
    # c = add_grating_couplers(c)

    # c = add_grating_couplers_with_loopback_fiber_array()
    c = add_grating_couplers_with_loopback_fiber_single()
    c.show(show_ports=True)

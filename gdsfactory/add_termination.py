from typing import Callable, Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.grating_coupler.elliptical_trenches import (
    grating_coupler_te,
    grating_coupler_tm,
)
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import get_waveguide_settings
from gdsfactory.routing.get_input_labels import get_input_labels
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.routing.utils import (
    check_ports_have_equal_spacing,
    direction_ports_from_list_ports,
)
from gdsfactory.types import ComponentFactory, StrOrDict


@cell
def add_termination(
    component: Component, terminator: ComponentFactory = taper_function
) -> Component:
    """returns component containing a comonent with all ports terminated"""
    terminator = gf.call_if_func(terminator)
    c = gf.Component(name=component.name + "_t")
    c.add_ref(component)

    for port in component.ports.values():
        t_ref = c.add_ref(terminator)
        t_ref.connect(list(t_ref.ports.values())[0].name, port)

    return c


def add_gratings_and_loopback_te(*args, **kwargs):
    return add_gratings_and_loopback(*args, **kwargs)


def add_gratings_and_loopback_tm(*args, grating_coupler=grating_coupler_tm, **kwargs):
    return add_gratings_and_loopback(*args, grating_coupler=grating_coupler, **kwargs)


@gf.cell_without_validator
def add_gratings_and_loopback(
    component: Component,
    grating_coupler: ComponentFactory = grating_coupler_te,
    excluded_ports: None = None,
    grating_separation: float = 127.0,
    bend_radius_loopback: Optional[float] = None,
    gc_port_name: str = "W0",
    gc_rotation: int = -90,
    straight_separation: float = 5.0,
    bend_factory: ComponentFactory = bend_euler,
    straight_factory: ComponentFactory = straight_function,
    layer_label: Tuple[int, int] = gf.LAYER.LABEL,
    layer_label_loopback: Optional[Tuple[int, int]] = None,
    component_name: None = None,
    with_loopback: bool = True,
    nlabels_loopback: int = 2,
    waveguide: StrOrDict = "strip",
    get_input_labels_function: Callable = get_input_labels,
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
        nlabels_loopback: number of labels of align ports (0: no labels, 1: first port, 2: both ports2)
        waveguide: waveguide definition from TECH.waveguide
        **kwargs: waveguide_settings
    """
    waveguide_settings = get_waveguide_settings(waveguide, **kwargs)
    bend_radius_loopback = bend_radius_loopback or waveguide_settings["radius"]
    excluded_ports = excluded_ports or []
    gc = gf.call_if_func(grating_coupler)

    direction = "S"
    component_name = component_name or component.name
    c = gf.Component()
    c.add_ref(component)

    # Find grating port name if not specified
    if gc_port_name is None:
        gc_port_name = list(gc.ports.values())[0].name

    # List the optical ports to connect
    optical_ports = component.get_ports_list(port_type="optical")
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
            radius=bend_radius_loopback, waveguide=waveguide, **kwargs
        )
        loopback_route = round_corners(
            points=points,
            bend_factory=bend90,
            straight_factory=straight_factory,
            waveguide=waveguide,
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
    # gc = gf.components.grating_coupler_elliptical_te()
    # cc = add_termination(c, gc)
    # import gdsfactory as gf
    # c = gf.components.straight()
    from gdsfactory.components.spiral_inner_io import spiral_inner_io

    c = spiral_inner_io()
    cc = add_gratings_and_loopback(component=c, with_loopback=True)

    # cc = add_termination(component=c)
    print(cc.get_settings()["settings"]["component"])
    cc.show()

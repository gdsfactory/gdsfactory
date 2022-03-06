"""FabC example """

import pathlib
from typing import Callable

import pydantic.dataclasses as dataclasses

import gdsfactory as gf
import gdsfactory.simulation as sim
from gdsfactory.add_pins import add_pin_rectangle_inside
from gdsfactory.component import Component
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports
from gdsfactory.simulation import lumerical
from gdsfactory.tech import LayerLevel, LayerStack
from gdsfactory.types import Layer


@dataclasses.dataclass(frozen=True)
class LayerMap:
    WG: Layer = (10, 1)
    WG_CLAD: Layer = (10, 2)
    WGN: Layer = (34, 0)
    WGN_CLAD: Layer = (36, 0)
    PIN: Layer = (100, 0)


LAYER = LayerMap()
WIDTH_NITRIDE_OBAND = 0.9
WIDTH_NITRIDE_CBAND = 1.0

select_ports_optical = gf.partial(select_ports, layers_excluded=((100, 0),))


def get_layer_stack_fab_c(thickness: float = 350.0) -> LayerStack:
    """Returns generic LayerStack"""
    return LayerStack(
        layers=[
            LayerLevel(
                layer=LAYER.WG,
                zmin=0.0,
                thickness=0.22,
            ),
            LayerLevel(
                layer=LAYER.WGN,
                zmin=0.22 + 0.1,
                thickness=0.4,
            ),
        ]
    )


def add_pins(
    component: Component,
    function: Callable = add_pin_rectangle_inside,
    pin_length: float = 0.5,
    port_layer: Layer = LAYER.PIN,
    **kwargs,
) -> None:
    """Add Pin port markers.

    Args:
        component: to add ports
        function:
        pin_length:
        port_layer:
        function: kwargs

    """
    for p in component.ports.values():
        function(
            component=component,
            port=p,
            layer=port_layer,
            layer_label=port_layer,
            pin_length=pin_length,
            **kwargs,
        )


# cross_sections

xs_nitridec = gf.partial(
    strip, width=WIDTH_NITRIDE_CBAND, layer=LAYER.WGN, layers_cladding=(LAYER.WGN_CLAD,)
)
xs_nitrideo = gf.partial(
    strip, width=WIDTH_NITRIDE_OBAND, layer=LAYER.WGN, layers_cladding=(LAYER.WGN_CLAD,)
)


# LEAF COMPONENTS have pins

mmi1x2_nitride_c = gf.partial(
    gf.components.mmi1x2,
    width=WIDTH_NITRIDE_CBAND,
    width_mmi=3,
    cross_section=xs_nitridec,
    decorator=add_pins,
)
mmi1x2_nitride_o = gf.partial(
    gf.components.mmi1x2,
    width=WIDTH_NITRIDE_OBAND,
    cross_section=xs_nitrideo,
    decorator=add_pins,
)
bend_euler_c = gf.partial(
    gf.components.bend_euler, cross_section=xs_nitridec, decorator=add_pins
)
straight_c = gf.partial(
    gf.components.straight, cross_section=xs_nitridec, decorator=add_pins
)
bend_euler_o = gf.partial(
    gf.components.bend_euler, cross_section=xs_nitrideo, decorator=add_pins
)
straight_o = gf.partial(
    gf.components.straight, cross_section=xs_nitrideo, decorator=add_pins
)

gc_nitride_c = gf.partial(
    gf.components.grating_coupler_elliptical,
    grating_line_width=0.6,
    wg_width=WIDTH_NITRIDE_CBAND,
    layer=LAYER.WGN,
    decorator=add_pins,
    layer_slab=None,
)

# HIERARCHICAL COMPONENTS made of leaf components

mzi_nitride_c = gf.partial(
    gf.components.mzi,
    cross_section=xs_nitridec,
    splitter=mmi1x2_nitride_c,
    decorator=add_pins,
    straight=straight_c,
    bend=bend_euler_c,
)
mzi_nitride_o = gf.partial(
    gf.components.mzi,
    cross_section=xs_nitrideo,
    splitter=mmi1x2_nitride_o,
    decorator=add_pins,
    straight=straight_o,
    bend=bend_euler_o,
)


# for testing
factory = dict(
    mmi1x2_nitride_c=mmi1x2_nitride_c,
    mmi1x2_nitride_o=mmi1x2_nitride_o,
    bend_euler_c=bend_euler_c,
    straight_c=straight_c,
    mzi_nitride_c=mzi_nitride_c,
    mzi_nitride_o=mzi_nitride_o,
    gc_nitride_c=gc_nitride_c,
)


LAYER_STACK = get_layer_stack_fab_c()
SPARAMETERS_PATH = pathlib.Path.home() / "fabc"

write_sparameters_lumerical = gf.partial(
    lumerical.write_sparameters_lumerical,
    layer_stack=LAYER_STACK,
    dirpath=SPARAMETERS_PATH,
)

plot_sparameters = gf.partial(
    sim.plot.plot_sparameters,
    dirpath=SPARAMETERS_PATH,
    write_sparameters_function=write_sparameters_lumerical,
)

get_sparameters_path_lumerical = gf.partial(
    sim.get_sparameters_data_lumerical,
    layer_stack=LAYER_STACK,
    dirpath=SPARAMETERS_PATH,
)


if __name__ == "__main__":

    mzi = mzi_nitride_c()
    mzi_gc = gf.routing.add_fiber_single(
        component=mzi,
        grating_coupler=gc_nitride_c,
        cross_section=xs_nitridec,
        optical_routing_type=1,
        straight=straight_c,
        bend=bend_euler_c,
        select_ports=select_ports_optical,
    )
    mzi_gc.show()

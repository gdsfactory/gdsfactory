"""
# Fab C

Lets assume that fab C has both Silicon and Silicon Nitride components, and you need different waveguide widths for C and O band.

Lets asume that O band nitride waveguide width is 0.9 and Cband Nitride waveguide width is 1um, and for 0.4um for Silicon O band and 0.5um for silicon Cband.

Lets also that this foundry has an LVS flow where all components have optical pins defined in layer (100, 0)


"""


from typing import Callable, Dict, Optional, Tuple

import pydantic
import pydantic.dataclasses as dataclasses

import pp
from pp.add_pins import add_pin_square_inside
from pp.component import Component, ComponentReference
from pp.tech import TECH, LayerLevel, LayerStack, Library, Tech, Waveguide
from pp.types import Layer


@dataclasses.dataclass
class LayerMap:
    WG: Layer = (10, 1)
    WG_CLAD: Layer = (10, 2)
    WGN: Layer = (34, 0)
    WGN_CLAD: Layer = (36, 0)
    PIN: Layer = (100, 0)


LAYER = LayerMap()
WIDTH_NITRIDE_OBAND = 0.9
WIDTH_NITRIDE_CBAND = 1.0
PORT_TYPE_TO_LAYER = dict(optical=(100, 0))


pp.tech.LAYER_SET.clear()


@pydantic.dataclasses.dataclass
class LayerStackFabc(LayerStack):
    WGN = LayerLevel(
        name="core",
        gds_layer=34,
        gds_datatype=0,
        thickness_nm=350.0,
        zmin_nm=220.0 + 100.0,
        material="sin",
        color="orange",
        alpha=1.0,
    )
    WGN_CLAD = LayerLevel(name="clad", gds_layer=36, gds_datatype=0)


@dataclasses.dataclass
class StripNitrideCband(Waveguide):
    width: float = WIDTH_NITRIDE_CBAND
    layer: Layer = LAYER.WGN
    auto_widen: bool = False
    radius: float = 10.0
    layers_cladding: Tuple[Layer, ...] = (LAYER.WGN_CLAD,)


@dataclasses.dataclass
class StripNitrideOband(StripNitrideCband):
    width: float = WIDTH_NITRIDE_OBAND


NITRIDE_CBAND = StripNitrideCband()
NITRIDE_OBAND = StripNitrideOband()

TECH.waveguide.fabc_nitride_cband = NITRIDE_CBAND
TECH.waveguide.fabc_nitride_oband = NITRIDE_OBAND


def add_pins(
    component: Component,
    reference: Optional[ComponentReference] = None,
    function: Callable = add_pin_square_inside,
    port_type_to_layer: Dict[str, Tuple[int, int]] = PORT_TYPE_TO_LAYER,
    pin_length: float = 0.5,
    **kwargs,
) -> None:
    """Add Pin port markers.

    Args:
        component: to add ports
        function:
        port_type_to_layer: dict mapping port types to marker layers for ports

    """
    reference = reference or component
    for p in reference.ports.values():
        if p.port_type in port_type_to_layer:
            layer = port_type_to_layer[p.port_type]
            function(
                component=component,
                port=p,
                layer=layer,
                label_layer=layer,
                pin_length=pin_length,
                **kwargs,
            )


mmi1x2_nitride_c = pp.partial(
    pp.c.mmi1x2,
    width=WIDTH_NITRIDE_CBAND,
    waveguide="fabc_nitride_cband",
    post_init=add_pins,
)
mmi1x2_nitride_o = pp.partial(
    pp.c.mmi1x2,
    width=WIDTH_NITRIDE_OBAND,
    waveguide="fabc_nitride_oband",
    post_init=add_pins,
)
bend_euler_c = pp.partial(
    pp.c.bend_euler, waveguide="fabc_nitride_cband", post_init=add_pins
)
straight_c = pp.partial(
    pp.c.straight, waveguide="fabc_nitride_cband", post_init=add_pins
)
bend_euler_o = pp.partial(
    pp.c.bend_euler, waveguide="fabc_nitride_oband", post_init=add_pins
)
straight_o = pp.partial(
    pp.c.straight, waveguide="fabc_nitride_oband", post_init=add_pins
)

mzi_nitride_c = pp.partial(
    pp.c.mzi,
    waveguide="fabc_nitride_cband",
    splitter=mmi1x2_nitride_c,
    post_init=add_pins,
    straight=straight_c,
    bend=bend_euler_c,
)
mzi_nitride_o = pp.partial(
    pp.c.mzi,
    waveguide="fabc_nitride_oband",
    splitter=mmi1x2_nitride_c,
    post_init=add_pins,
    straight=straight_o,
    bend=bend_euler_o,
)


gc_nitride_c = pp.partial(
    pp.components.grating_coupler_elliptical_te,
    grating_line_width=0.6,
    wg_width=WIDTH_NITRIDE_CBAND,
    layer=LAYER.WGN,
    post_init=add_pins,
)


TECH_FABC = Tech(name="fab_c")
LIBRARY = Library(name="fab_c")
LIBRARY.register(
    [
        mmi1x2_nitride_c,
        mmi1x2_nitride_o,
        bend_euler_c,
        straight_c,
        mzi_nitride_c,
        mzi_nitride_o,
    ]
)


if __name__ == "__main__":
    # c = mmi1x2_nitride_o()
    # c.show()

    mzi = mzi_nitride_c()
    mzi_gc = pp.routing.add_fiber_single(
        component=mzi,
        grating_coupler=gc_nitride_c,
        waveguide="fabc_nitride_cband",
        optical_routing_type=1,
        straight_factory=straight_c,
        bend_factory=bend_euler_c,
    )
    mzi_gc.show()
    mzi_gc.plot()

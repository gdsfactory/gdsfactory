"""
# Fab C

Lets assume that fab C has both Silicon and Silicon Nitride components, and you need different waveguide widths for C and O band.

Lets asume that O band nitride waveguide width is 0.9 and Cband Nitride waveguide width is 1um, and for 0.4um for Silicon O band and 0.5um for silicon Cband.

Lets also that this foundry has an LVS flow where all components have optical pins defined in layer (100, 0)


"""


from typing import Callable, Dict, Optional, Tuple

import pydantic.dataclasses as dataclasses

import pp
from pp.add_pins import add_pin_square_inside
from pp.cell import cell
from pp.component import Component, ComponentReference
from pp.tech import TECH, Factory, Tech, Waveguide
from pp.types import Layer, StrOrDict


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

TECH.waveguide.nitride_cband = NITRIDE_CBAND
TECH.waveguide.nitride_oband = NITRIDE_OBAND


def add_pins_custom(
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
        layer = port_type_to_layer[p.port_type]
        function(
            component=component,
            port=p,
            layer=layer,
            label_layer=layer,
            pin_length=pin_length,
            **kwargs,
        )


@cell
def mmi1x2_nitride_cband(
    width: float = WIDTH_NITRIDE_CBAND,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.5,
    width_mmi: float = 2.5,
    gap_mmi: float = 0.25,
    with_cladding_box: bool = True,
    waveguide: StrOrDict = "nitride_cband",
    **kwargs,
) -> Component:
    c = pp.components.mmi1x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        with_cladding_box=with_cladding_box,
        waveguide=waveguide,
        **kwargs,
    )
    add_pins_custom(c)
    return c


@cell
def mmi1x2_nitride_oband(
    width: float = WIDTH_NITRIDE_OBAND,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.5,
    width_mmi: float = 2.5,
    gap_mmi: float = 0.25,
    with_cladding_box: bool = True,
    waveguide: StrOrDict = "nitride_oband",
    **kwargs,
) -> Component:
    c = pp.components.mmi1x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        with_cladding_box=with_cladding_box,
        waveguide=waveguide,
        **kwargs,
    )
    add_pins_custom(c)
    return c


@cell
def bend_euler(waveguide: StrOrDict = "nitride_cband", **kwargs) -> Component:
    c = pp.c.bend_euler(waveguide=waveguide, **kwargs)
    add_pins_custom(c)
    return c


@cell
def bend_euler_cband(waveguide: StrOrDict = "nitride_cband", **kwargs) -> Component:
    c = pp.c.bend_euler(waveguide=waveguide, **kwargs)
    add_pins_custom(c)
    return c


@cell
def straight_cband(waveguide: StrOrDict = "nitride_cband", **kwargs) -> Component:
    c = pp.c.straight(waveguide=waveguide, **kwargs)
    add_pins_custom(c)
    return c


FACTORY = Factory(name="fab_c", post_init=add_pins_custom)
# FACTORY = TECH.factory

TECH_FABC = Tech(name="fab_c")
TECH_FABC.factory = FACTORY
FACTORY.register(
    [mmi1x2_nitride_cband, mmi1x2_nitride_oband, bend_euler_cband, straight_cband]
)


@cell
def mzi_nitride_cband(delta_length: float = 10.0) -> Component:
    c = pp.c.mzi(
        delta_length=delta_length,
        splitter="mmi1x2_nitride_cband",
        waveguide="nitride_cband",
        width=WIDTH_NITRIDE_CBAND,
        bend="bend_euler_cband",
        straight="straight_cband",
        factory=FACTORY,
    )
    add_pins_custom(c)
    return c


@cell
def mzi_nitride_oband(delta_length: float = 10.0) -> Component:
    c = pp.c.mzi(
        delta_length=delta_length,
        splitter="mmi1x2_nitride_cband",
        waveguide="nitride_oband",
        width=WIDTH_NITRIDE_CBAND,
        bend="bend_euler_cband",
        straight="straight_cband",
        factory=FACTORY,
    )
    add_pins_custom(c)
    return c


FACTORY.register([mzi_nitride_cband, mzi_nitride_oband])


if __name__ == "__main__":
    mzi = mzi_nitride_cband()
    gc = pp.c.grating_coupler_elliptical_te(
        wg_width=WIDTH_NITRIDE_CBAND, layer=LAYER.WGN
    )
    mzi_gc = pp.routing.add_fiber_single(
        component=mzi,
        grating_coupler=gc,
        waveguide="nitride_cband",
        straight_factory=straight_cband,
        optical_routing_type=1,
        factory=FACTORY,
    )
    mzi_gc.show()

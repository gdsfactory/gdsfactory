"""
Raises strange error for a transition

"""

import pydantic

import gdsfactory as gf
from gdsfactory.types import Layer


@pydantic.dataclasses.dataclass
class LayerMap:
    WG: Layer = (2, 0)
    SLAB: Layer = (41, 0)
    PPP: Layer = (4, 0)
    NPP: Layer = (5, 0)
    HEATER: Layer = (6, 0)


LAYER = LayerMap()


xs_strip_heater_metal = gf.partial(
    gf.cross_section.strip_heater_metal, layer=LAYER.WG, width=1
)
xs_rib_heater_doped = gf.partial(
    gf.cross_section.rib_heater_doped, layer=LAYER.WG, width=1, layer_slab=LAYER.SLAB
)
xs_rib_pin = gf.partial(
    gf.cross_section.pin, layer=LAYER.WG, width=1, layer_slab=LAYER.SLAB
)


ps_heater_metal = gf.partial(
    gf.c.straight_heater_metal,
    cross_section_heater=xs_strip_heater_metal,
)
ps_heater_doped = gf.partial(
    gf.c.straight_heater_doped_rib,
    cross_section_heater=xs_rib_heater_doped,
)
ps_pin = gf.partial(
    gf.c.straight_pin,
    cross_section=xs_rib_pin,
)


component_factory = dict(
    ps_heater_metal=ps_heater_metal,
    ps_heater_doped=ps_heater_doped,
)


def write_library(component_factory, dirpath):
    for function in component_factory.values():
        component = function()
        component.write_gds_with_metadata(gdsdir=dirpath)


if __name__ == "__main__":
    # import pathlib
    # write_library(component_factory=component_factory, dirpath=pathlib.Path.cwd())

    # c = ps_heater_metal()
    # c = ps_pin()
    c = ps_heater_doped()
    c.show()

from __future__ import annotations

from functools import partial
from pathlib import Path

import pydantic

import gdsfactory as gf
from gdsfactory.cross_section import strip_heater_metal
from gdsfactory.typings import ComponentFactory, Layer


@pydantic.dataclasses.dataclass
class LayerMap:
    WG: Layer = (2, 0)
    SLAB: Layer = (41, 0)
    PPP: Layer = (4, 0)
    NPP: Layer = (5, 0)
    HEATER: Layer = (6, 0)


LAYER = LayerMap


xs_strip = partial(gf.cross_section.strip, layer=(1, 0), width=1)
xs_strip_heater_metal = partial(strip_heater_metal, layer=(1, 0), width=1)
rib_heater_doped = partial(
    gf.cross_section.rib_heater_doped, layer=(1, 0), width=1, layer_slab=LAYER.SLAB
)
xs_strip_heater_doped = partial(
    gf.cross_section.strip_heater_doped,
    layer=(1, 0),
    width=1,
    layers_heater=((1, 0), LAYER.HEATER),
    bbox_offsets_heater=(0, 0.1),
)
rib_pin = partial(gf.cross_section.pin, layer=(1, 0), width=1, layer_slab=LAYER.SLAB)


ps_heater_metal = partial(
    gf.components.straight_heater_metal,
    cross_section_heater=xs_strip_heater_metal,
)
ps_heater_doped = partial(
    gf.components.straight_heater_doped_strip,
    cross_section=xs_strip,
    cross_section_heater=xs_strip_heater_doped,
)
ps_pin = partial(
    gf.components.straight_pin,
    cross_section=rib_pin,
)

component_factory = dict(
    ps_heater_metal=ps_heater_metal,
    ps_heater_doped=ps_heater_doped,
)


def write_library(
    component_factory: dict[str, ComponentFactory], dirpath: Path
) -> None:
    for function in component_factory.values():
        component = function()
        component.write_gds(gdsdir=dirpath, with_metadata=True)


if __name__ == "__main__":
    # import pathlib
    # write_library(component_factory=component_factory, dirpath=pathlib.Path.cwd())

    c = ps_heater_doped()
    # c = ps_heater_metal()
    # c = ps_pin()
    c.show()

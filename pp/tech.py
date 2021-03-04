from dataclasses import dataclass
from typing import Optional, Tuple

from pp.layers import LAYER
from pp.types import Layer


@dataclass(frozen=True)
class Tech:
    name: str
    wg_width: float
    bend_radius: float
    cladding_offset: float
    layer_wg: Layer
    layers_cladding: Optional[Tuple[Layer, ...]]
    layer_label: Layer
    taper_length: float
    taper_width: float
    fiber_single_spacing: float = 50.0
    fiber_array_spacing: float = 127.0
    fiber_input_to_output_spacing: float = 120.0
    snap_to_grid_nm: Optional[int] = None


@dataclass(frozen=True)
class TechSiliconCband(Tech):
    name: str = "silicon_cband"
    wg_width: float = 0.5
    bend_radius: float = 10.0
    cladding_offset: float = 3.0
    layer_wg: Layer = LAYER.WG
    layers_cladding: Tuple[Layer, ...] = (LAYER.WGCLAD,)
    layer_label: Layer = LAYER.LABEL
    layer_heater: Layer = LAYER.HEATER
    taper_length: float = 15.0
    taper_width: float = 2.0  # taper to wider waveguides for lower loss


@dataclass(frozen=True)
class TechNitrideCband(Tech):
    name: str = "nitride_cband"
    wg_width: float = 1.0
    bend_radius: float = 20.0
    cladding_offset: float = 3.0
    layer_wg: Layer = LAYER.WGN
    layers_cladding: Tuple[Layer, ...] = (LAYER.NO_TILE_SI,)
    layer_label: Layer = LAYER.LABEL
    taper_length: float = 20.0
    taper_width: float = 1.0


@dataclass(frozen=True)
class TechMetal1(Tech):
    name: str = "metal1"
    wg_width: float = 2.0
    bend_radius: float = 10.0
    cladding_offset: float = 3.0
    layer_wg: Layer = LAYER.M1
    layers_cladding: Tuple[Layer, ...] = (LAYER.WGCLAD,)
    layer_label: Layer = LAYER.LABEL
    taper_length: float = 20.0
    taper_width: float = 10.0
    snap_to_grid_nm: int = 10


TECH_SILICON_C = TechSiliconCband()
TECH_NITRIDE_C = TechNitrideCband()
TECH_METAL1 = TechMetal1()


if __name__ == "__main__":
    import json

    from pydantic.json import pydantic_encoder

    import pp

    c = pp.c.waveguide(tech=TECH_METAL1)
    print(c.name)

    tech = TECH_METAL1
    # print(tech.dict())
    print(json.dumps(tech, indent=4, default=pydantic_encoder))

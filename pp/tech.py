import dataclasses
from typing import Optional, Tuple

from pp.cross_section import CrossSection, CrossSectionFactory, strip
from pp.layers import LAYER
from pp.types import Layer


@dataclasses.dataclass(frozen=True)
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
    cross_section: CrossSectionFactory = strip

    def get_cross_section(self, width=Optional[float]) -> CrossSection:
        return self.cross_section(
            width=width or self.wg_width,
            layer=self.layer_wg,
            layers_cladding=self.layers_cladding,
            cladding_offset=self.cladding_offset,
        )


@dataclasses.dataclass(frozen=True)
class TechSiliconCband(Tech):
    name: str = "si_c"
    wg_width: float = 0.5
    bend_radius: float = 5.0
    cladding_offset: float = 3.0
    layer_wg: Layer = LAYER.WG
    layers_cladding: Tuple[Layer, ...] = (LAYER.WGCLAD,)
    layer_label: Layer = LAYER.LABEL
    taper_length: float = 15.0
    taper_width: float = 2.0  # taper to wider waveguides for lower loss


@dataclasses.dataclass(frozen=True)
class TechNitrideCband(Tech):
    name: str = "sin_c"
    wg_width: float = 1.0
    bend_radius: float = 10.0
    cladding_offset: float = 3.0
    layer_wg: Layer = LAYER.WGN
    layers_cladding: Tuple[Layer, ...] = (LAYER.WGCLAD,)
    layer_label: Layer = LAYER.LABEL
    taper_length: float = 20.0
    taper_width: float = 10.0


@dataclasses.dataclass(frozen=True)
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


TECH_SILICON_C = TechSiliconCband()
TECH_NITRIDE_C = TechNitrideCband()
TECH_METAL1 = TechMetal1()

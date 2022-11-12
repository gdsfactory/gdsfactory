"""Technology settings."""
import pathlib
from typing import Tuple, Union

from pydantic import BaseModel

from gdsfactory.types import LayerLevel, LayerStack

module_path = pathlib.Path(__file__).parent.absolute()
Layer = Tuple[int, int]
LayerSpec = Union[int, Layer, str, None]
nm = 1e-3


class LayerMap(BaseModel):
    """Generic layermap based on book.

    Lukas Chrostowski, Michael Hochberg, "Silicon Photonics Design",
    Cambridge University Press 2015, page 353

    You will need to create a new LayerMap with your specific foundry layers.
    """

    WG: Layer = (1, 0)
    WGCLAD: Layer = (111, 0)
    SLAB150: Layer = (2, 0)
    SLAB90: Layer = (3, 0)
    DEEPTRENCH: Layer = (4, 0)
    GE: Layer = (5, 0)
    WGN: Layer = (34, 0)
    WGN_CLAD: Layer = (36, 0)

    N: Layer = (20, 0)
    NP: Layer = (22, 0)
    NPP: Layer = (24, 0)
    P: Layer = (21, 0)
    PP: Layer = (23, 0)
    PPP: Layer = (25, 0)
    GEN: Layer = (26, 0)
    GEP: Layer = (27, 0)

    HEATER: Layer = (47, 0)
    M1: Layer = (41, 0)
    M2: Layer = (45, 0)
    M3: Layer = (49, 0)
    VIAC: Layer = (40, 0)
    VIA1: Layer = (44, 0)
    VIA2: Layer = (43, 0)
    PADOPEN: Layer = (46, 0)

    DICING: Layer = (100, 0)
    NO_TILE_SI: Layer = (71, 0)
    PADDING: Layer = (67, 0)
    DEVREC: Layer = (68, 0)
    FLOORPLAN: Layer = (64, 0)
    TEXT: Layer = (66, 0)
    PORT: Layer = (1, 10)
    PORTE: Layer = (1, 11)
    PORTH: Layer = (70, 0)
    SHOW_PORTS: Layer = (1, 12)
    LABEL: Layer = (201, 0)
    LABEL_SETTINGS: Layer = (202, 0)
    TE: Layer = (203, 0)
    TM: Layer = (204, 0)
    DRC_MARKER: Layer = (205, 0)
    LABEL_INSTANCE: Layer = (206, 0)
    ERROR_MARKER: Layer = (207, 0)
    ERROR_PATH: Layer = (208, 0)


LAYER = LayerMap()


def get_layer_stack_generic(
    thickness_wg: float = 220 * nm,
    thickness_clad: float = 3.0,
    thickness_nitride: float = 350 * nm,
    gap_silicon_to_nitride: float = 100 * nm,
) -> LayerStack:
    """Returns generic LayerStack.

    based on paper https://www.degruyter.com/document/doi/10.1515/nanoph-2013-0034/html

    Args:
        thickness_wg: waveguide thickness.
        thickness_clad: cladding.
        thickness_nitride: for nitride.
        gap_silicon_to_nitride: in um.
    """
    return LayerStack(
        layers=dict(
            core=LayerLevel(
                layer=LAYER.WG,
                thickness=thickness_wg,
                zmin=0.0,
                material="si",
            ),
            clad=LayerLevel(
                layer=LAYER.WGCLAD,
                zmin=0.0,
                material="sio2",
                thickness=thickness_clad,
            ),
            slab150=LayerLevel(
                layer=LAYER.SLAB150,
                thickness=150e-3,
                zmin=0,
                material="si",
            ),
            slab90=LayerLevel(
                layer=LAYER.SLAB90,
                thickness=90e-3,
                zmin=0.0,
                material="si",
            ),
            nitride=LayerLevel(
                layer=LAYER.WGN,
                thickness=thickness_nitride,
                zmin=thickness_wg + gap_silicon_to_nitride,
                material="sin",
            ),
            ge=LayerLevel(
                layer=LAYER.GE,
                thickness=500e-3,
                zmin=thickness_wg,
                material="ge",
            ),
            via_contact=LayerLevel(
                layer=LAYER.VIAC,
                thickness=1100e-3,
                zmin=90e-3,
                material="Aluminum",
            ),
            metal1=LayerLevel(
                layer=LAYER.M1,
                thickness=750e-3,
                zmin=thickness_wg + 1100e-3,
                material="Aluminum",
            ),
            heater=LayerLevel(
                layer=LAYER.HEATER,
                thickness=750e-3,
                zmin=thickness_wg + 1100e-3,
                material="TiN",
            ),
            viac=LayerLevel(
                layer=LAYER.VIA1,
                thickness=1500e-3,
                zmin=thickness_wg + 1100e-3 + 750e-3,
                material="Aluminum",
            ),
            metal2=LayerLevel(
                layer=LAYER.M2,
                thickness=2000e-3,
                zmin=thickness_wg + 1100e-3 + 750e-3 + 1.5,
                material="Aluminum",
            ),
        )
    )


LAYER_STACK = get_layer_stack_generic()


if __name__ == "__main__":
    ls = LAYER_STACK
    ls.get_klayout_3d_script()

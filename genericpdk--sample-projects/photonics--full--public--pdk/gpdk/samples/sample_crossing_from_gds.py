"""Sample implementation for importing crossing components from GDS files."""

import pathlib

import gdsfactory as gf

from gpdk import LAYER
from gpdk.config import PATH

module = pathlib.Path(__file__).parent.absolute()


@gf.cell
def sample_crossing_from_gds() -> gf.Component:
    """Sample of a crossing imported from GDS. Make sure you use the right layer, and ports (width, orientation and location)."""
    layer = LAYER.WG

    c = gf.import_gds(PATH.gds / "SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing.gds")
    c.remap_layers({(3, 0): layer})

    c.add_port("o1", orientation=180, center=(485.0, 0.0), width=0.4, layer=layer)
    c.add_port("o2", orientation=90, center=(489.235, 4.235), width=0.4, layer=layer)
    c.add_port("o3", orientation=0, center=(493.47, 0.0), width=0.4, layer=layer)
    c.add_port("o4", orientation=270, center=(489.235, -4.235), width=0.4, layer=layer)
    return c

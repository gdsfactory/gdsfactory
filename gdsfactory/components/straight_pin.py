"""Straight Doped PIN waveguide."""
from typing import Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import pin
from gdsfactory.path import extrude, straight
from gdsfactory.snap import snap_to_grid
from gdsfactory.tech import LAYER, TECH
from gdsfactory.types import Layer


@gf.cell
def straight_pin(
    length: float = 10.0,
    npoints: int = 2,
    width: float = TECH.waveguide.strip.width,
    layer: Layer = TECH.waveguide.strip.layer,
    layer_slab: Layer = LAYER.SLAB90,
    width_i: float = 0.0,
    width_p: float = 1.0,
    width_n: float = 1.0,
    width_pp: float = 1.0,
    width_np: float = 1.0,
    width_ppp: float = 1.0,
    width_npp: float = 1.0,
    layer_p: Tuple[int, int] = LAYER.P,
    layer_n: Tuple[int, int] = LAYER.N,
    layer_pp: Tuple[int, int] = LAYER.Pp,
    layer_np: Tuple[int, int] = LAYER.Np,
    layer_ppp: Tuple[int, int] = LAYER.Ppp,
    layer_npp: Tuple[int, int] = LAYER.Npp,
    snap_to_grid_nm: int = TECH.snap_to_grid_nm,
) -> Component:
    """Returns a Doped PIN waveguide.

    Args:
        length: of straight
        npoints: number of points
        width: straight width
        layer: layer for
        layers_cladding: for cladding
        cladding_offset: offset from straight to cladding edge
        cross_section_factory: function that returns a cross_section

    .. code::


                           |<------width------>|
                            ____________________
                           |     |       |     |
        ___________________|     |       |     |__________________________|
                                 |       |                                |
            P++     P+     P     |   I   |     N        N+         N++    |
        __________________________________________________________________|
                                                                          |
                                 |width_i| width_n | width_np | width_npp |
                                    0    oi        on        onp         onpp

    """

    p = straight(length=length, npoints=npoints)
    cross_section = pin(
        width=width,
        layer=layer,
        layer_slab=layer_slab,
        width_i=width_i,
        width_p=width_p,
        width_n=width_n,
        width_pp=width_pp,
        width_np=width_np,
        width_ppp=width_ppp,
        width_npp=width_npp,
        layer_p=layer_p,
        layer_n=layer_n,
        layer_pp=layer_pp,
        layer_np=layer_np,
        layer_ppp=layer_ppp,
        layer_npp=layer_npp,
    )
    c = extrude(p, cross_section, snap_to_grid_nm=snap_to_grid_nm)
    c.width = width
    c.length = snap_to_grid(length)
    return c


if __name__ == "__main__":

    c = straight_pin(width_i=1)
    print(c.ports.keys())
    c.show()

from typing import Callable

import numpy as np
from pydantic import BaseModel

import gdsfactory as gf
from gdsfactory.typings import Layer


class DopingLayerLevel(BaseModel):
    """Level for doping layer.

    Parameters:
        layer: (GDSII Layer number, GDSII datatype).
        type: str: "Acceptor" or "Donor".
        z_profile: callable evaluating doping versus depth.
        xy_profile: callable modulating z_profile at the edge of the layer.

    """

    layer: Layer
    type: str
    z_profile: Callable
    # xy_profile: Optional[Callable] = None # not implemented yet

    class Config:
        """pydantic config."""

        frozen = True
        extra = "forbid"


cm3_to_um3 = 1e-12


def get_doping_info_generic(
    n_conc: float = 1e17,  # * cm3_to_um3,
    p_conc: float = 1e17,  # * cm3_to_um3,
    npp_conc: float = 1e18,  # * cm3_to_um3,
    ppp_conc: float = 1e18,  # * cm3_to_um3,
):
    layermap = gf.generic_tech.LayerMap()

    return {
        "N": DopingLayerLevel(
            layer=layermap.N,
            type="Donor",
            z_profile=step(n_conc),
        ),
        "P": DopingLayerLevel(
            layer=layermap.P,
            type="Acceptor",
            z_profile=step(p_conc),
        ),
        "NPP": DopingLayerLevel(
            layer=layermap.NPP,
            type="Donor",
            z_profile=step(npp_conc),
        ),
        "PPP": DopingLayerLevel(
            layer=layermap.PPP,
            type="Acceptor",
            z_profile=step(ppp_conc),
        ),
    }


# def get_doping_xyz():
#     """Returns acceptor  vs x,y,z from DopingLayerLevel and a component."""

# def get_doping_density():
#     """Returns acceptor  vs x,y,z from DopingLayerLevel and a component."""


# def get_net_doping(component: Component, doping_info: Dict):
#     """Returns net doping from DopingLayerLevel and a component."""

#     for name, dopinglevel in doping_info.items():
#         print("ok")


def step(c, zmin=-np.inf, zmax=np.inf):
    """Step function doping of value c, between zmin and zmax."""
    return lambda y: c


# def step(c, zmin=-np.inf, zmax=np.inf):
#     """Step function doping of value c, between zmin and zmax."""
#     return lambda x, y, z: np.heaviside(c - zmin) - np.heaviside(c - zmax)

# def gaussian(n0, range, straggle):
#     """Gaussian function doping of value c."""
#     return n0*np.exp(-(z - range)**2/(2*straggle**2))

# def pearsonIV(n0, range, straggle):
#     """Gaussian function doping of value c."""
#     return n0*np.exp(-(z - range)**2/(2*straggle**2))

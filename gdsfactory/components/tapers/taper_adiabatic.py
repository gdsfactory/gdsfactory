from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory.path import transition_adiabatic
from gdsfactory.typings import CrossSectionSpec


def neff_TE1550SOI_220nm(w: float) -> float:
    """Returns the effective index of the fundamental TE mode for a 220nm-thick core with 3.45 index, fully clad with 1.44 index.

    Args:
        w: width in um.

    Returns:
        effective index.
    """
    adiabatic_polyfit_TE1550SOI_220nm = np.array(
        [
            1.02478963e-09,
            -8.65556534e-08,
            3.32415694e-06,
            -7.68408985e-05,
            1.19282177e-03,
            -1.31366332e-02,
            1.05721429e-01,
            -6.31057637e-01,
            2.80689677e00,
            -9.26867694e00,
            2.24535191e01,
            -3.90664800e01,
            4.71899278e01,
            -3.74726005e01,
            1.77381560e01,
            -1.12666286e00,
        ]
    )
    return float(np.poly1d(adiabatic_polyfit_TE1550SOI_220nm)(w))


@gf.cell_with_module_name
def taper_adiabatic(
    width1: float = 0.5,
    width2: float = 5.0,
    length: float = 0,
    neff_w: Callable[[float], float] = neff_TE1550SOI_220nm,
    alpha: float = 1,
    wavelength: float = 1.55,
    npoints: int = 200,
    cross_section: CrossSectionSpec = "strip",
    max_length: float = 200,
) -> gf.Component:
    """Returns a straight adiabatic_taper from an effective index callable.

    Args:
        width1: initial width.
        width2: final width.
        length: 0 uses the optimized length, and otherwise the optimal shape is compressed/stretched to the specified length.
        neff_w: a callable that returns the effective index as a function of width
                - By default, will use a compact model of neff(y) for fundamental 1550 nm TE mode of 220nm-thick core with 3.45 index, fully clad with 1.44 index. Many coefficients are needed to capture the behaviour.
        alpha: parameter that scales the rate of width change.
                - closer to 0 means longer and more adiabatic;
                - 1 is the intuitive limit beyond which higher order modes are excited;
                - [2] reports good performance up to 1.4 for fundamental TE in SOI (for multiple core thicknesses)
        wavelength: wavelength in um.
        npoints: number of points for sampling.
        cross_section: cross_section specification.
        max_length: maximum length for the taper.

    References:
        [1] Burns, W. K., et al. "Optical waveguide parabolic coupling horns." Appl. Phys. Lett., vol. 30, no. 1, 1 Jan. 1977, pp. 28-30, doi:10.1063/1.89199.
        [2] Fu, Yunfei, et al. "Efficient adiabatic silicon-on-insulator waveguide taper." Photonics Res., vol. 2, no. 3, 1 June 2014, pp. A41-A44, doi:10.1364/PRJ.2.000A41.
        npoints: number of points for sampling
    """
    xs = gf.get_cross_section(cross_section)
    layer = xs.layer
    assert layer is not None

    # Obtain optimal curve
    x_opt, w_opt = transition_adiabatic(
        width1,
        width2,
        neff_w=neff_w,
        wavelength=wavelength,
        alpha=alpha,
        max_length=max_length,
    )

    # Resample the points
    from scipy import interpolate

    w_opt_interp = interpolate.interp1d(x_opt, w_opt)

    if not length:
        length = x_opt[-1]
    x = np.linspace(0, length, npoints)
    w: npt.NDArray[np.floating[Any]] = w_opt_interp(x)

    assert isinstance(w, np.ndarray)

    # Stretch/compress x
    x_array: npt.NDArray[np.float64] = np.linspace(0, length, npoints) * (
        1 + length - x_opt[-1]
    )
    assert isinstance(x_array, np.ndarray)
    y_array = w / 2

    c = gf.Component()
    c.add_polygon(
        list(zip(x_array, y_array)) + list(zip(x_array, -y_array))[::-1],
        layer=layer,
    )

    # Define ports
    c.add_port(
        name="o1",
        center=(0, 0),
        width=width1,
        orientation=180,
        cross_section=cross_section,
        layer=layer,
    )
    c.add_port(
        name="o2",
        center=(length, 0),
        width=width2,
        orientation=0,
        cross_section=cross_section,
        layer=layer,
    )
    xs.add_bbox(c)
    return c


if __name__ == "__main__":
    c = taper_adiabatic(width1=0.5, width2=5, cross_section="rib_bbox")
    c.show()

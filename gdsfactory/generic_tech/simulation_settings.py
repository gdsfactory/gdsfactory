from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from scipy import interpolate

if TYPE_CHECKING:
    pass

material_name_to_lumerical_default = {
    "si": "Si (Silicon) - Palik",
    "sio2": "SiO2 (Glass) - Palik",
    "sin": "Si3N4 (Silicon Nitride) - Phillip",
}


class SimulationSettingsLumericalFdtd(BaseModel):
    """Lumerical FDTD simulation_settings.

    Parameters:
        background_material: for the background.
        port_margin: on both sides of the port width (um).
        port_height: port height (um).
        port_extension: port extension (um).
        mesh_accuracy: 2 (1: coarse, 2: fine, 3: superfine).
        zmargin: for the FDTD region (um).
        ymargin: for the FDTD region (um).
        xmargin: for the FDTD region (um).
        wavelength_start: 1.2 (um).
        wavelength_stop: 1.6 (um).
        wavelength_points: 500.
        simulation_time: (s) related to max path length
            3e8/2.4*10e-12*1e6 = 1.25mm.
        simulation_temperature: in kelvin (default = 300).
        frequency_dependent_profile: compute mode profiles for each wavelength.
        field_profile_samples: number of wavelengths to compute field profile.
    """

    background_material: str = "sio2"
    port_margin: float = 1.5
    port_extension: float = 5.0
    mesh_accuracy: int = 2
    wavelength_start: float = 1.2
    wavelength_stop: float = 1.6
    wavelength_points: int = 500
    simulation_time: float = 10e-12
    simulation_temperature: float = 300
    frequency_dependent_profile: bool = True
    field_profile_samples: int = 15
    distance_monitors_to_pml: float = 0.5
    material_name_to_lumerical: dict[str, str] = material_name_to_lumerical_default

    class Config:
        """pydantic basemodel config."""

        arbitrary_types_allowed = True


SIMULATION_SETTINGS_LUMERICAL_FDTD = SimulationSettingsLumericalFdtd()

wavelengths = [
    0.600,
    0.700,
    0.800,
    0.900,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
]
refractive_indices_silicon = [
    3.90700641,
    3.75348253,
    3.66385374,
    3.6063478,
    3.56700863,
    3.53880654,
    3.5178488,
    3.50182331,
    3.48928031,
    3.47927041,
    3.47114944,
    3.46446703,
    3.45890031,
    3.45421262,
    3.45022722,
]

refractive_indices_nitride = [
    2.04088838,
    2.0261673,
    2.01736617,
    2.01164108,
    2.00769004,
    2.00484054,
    2.00271388,
    2.00108253,
    1.99980257,
    1.99877915,
    1.99794759,
    1.99726249,
    1.99669118,
    1.99620968,
    1.99580002,
]

refractive_indices_oxide = [
    1.4677275,
    1.46456204,
    1.46253104,
    1.46114913,
    1.46016586,
    1.45944112,
    1.45889147,
    1.45846465,
    1.45812656,
    1.45785418,
    1.45763151,
    1.45744713,
    1.45729274,
    1.45716216,
    1.45705073,
]


def _interpolate_material(wav: np.ndarray, wavelengths, refractive_index) -> np.ndarray:
    """Returns Interpolated refractive index of material for given wavelength.

    Args:
        wav: wavelength (um) to interpolate.
        wavelengths: list of reference wavelengths (um).
        refractive_index: list of reference refractive indices.
    """
    f = interpolate.interp1d(wavelengths, refractive_index)
    return f(wav)


si = partial(
    _interpolate_material,
    wavelengths=wavelengths,
    refractive_index=refractive_indices_silicon,
)
sio2 = partial(
    _interpolate_material,
    wavelengths=wavelengths,
    refractive_index=refractive_indices_oxide,
)
sin = partial(
    _interpolate_material,
    wavelengths=wavelengths,
    refractive_index=refractive_indices_nitride,
)


materials_index = {"si": si, "sio2": sio2, "sin": sin}

if __name__ == "__main__":
    print(sio2(1.55))

from __future__ import annotations

from functools import partial

from pydantic import BaseModel, ConfigDict

from gdsfactory.gpdk.material_cards import (
    MATERIAL_ALIASES,
    MATERIAL_CARDS,
    get_refractive_index,
)
from gdsfactory.gpdk.material_indices import (
    _interpolate_material,
    refractive_indices_nitride,
    refractive_indices_oxide,
    refractive_indices_silicon,
    si,
    sin,
    sio2,
    wavelengths,
)
from gdsfactory.typings import MaterialSpec

material_name_to_lumerical_default = {
    "si": "Si (Silicon) - Palik",
    "sio2": "SiO2 (Glass) - Palik",
    "sin": "Si3N4 (Silicon Nitride) - Phillip",
}

material_name_to_lumerical_default.update(
    {
        name: material_name_to_lumerical_default[alias]
        for alias, name in MATERIAL_ALIASES.items()
        if alias in material_name_to_lumerical_default
    }
)


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
    model_config = ConfigDict(arbitrary_types_allowed=True)


SIMULATION_SETTINGS_LUMERICAL_FDTD = SimulationSettingsLumericalFdtd()

materials_index: dict[str, MaterialSpec] = {
    name: partial(get_refractive_index, name) for name in MATERIAL_CARDS
}
materials_index.update(
    {alias: materials_index[name] for alias, name in MATERIAL_ALIASES.items()}
)


__all__ = [
    "SIMULATION_SETTINGS_LUMERICAL_FDTD",
    "SimulationSettingsLumericalFdtd",
    "_interpolate_material",
    "material_name_to_lumerical_default",
    "materials_index",
    "refractive_indices_nitride",
    "refractive_indices_oxide",
    "refractive_indices_silicon",
    "si",
    "sin",
    "sio2",
    "wavelengths",
]

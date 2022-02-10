from typing import Dict, Optional, Union

import meep as mp
import meep.materials as mat
import numpy as np

# Dictionary for materials with different names in Meep
MATERIAL_NAME_TO_MEEP = {
    "si": "Si",
    "sin": "Si3N4_NIR",
}

MATERIALS = [m for m in dir(mat) if not m.startswith("_")]


def get_material(
    name: str = "si",
    wavelength: float = 1.55,
    dispersive: bool = False,
    material_name_to_meep: Optional[Dict[str, Union[str, float]]] = None,
) -> mp.Medium:
    """Returns Meep Medium from database

    Args:
        name: material name
        wavelength: wavelength (um)
        dispersive: True for built-in Meep index model,
            False for simple, non-dispersive model

    Note:
        Using the built-in models can be problematic at low resolution.
        If fields are NaN or Inf, increase resolution or use a non-dispersive model
    """
    material_name_to_meep_new = material_name_to_meep or {}
    material_name_to_meep = MATERIAL_NAME_TO_MEEP.copy()
    material_name_to_meep.update(**material_name_to_meep_new)

    materials_lower = [material.lower() for material in MATERIALS]

    if name in MATERIAL_NAME_TO_MEEP.keys():
        name = MATERIAL_NAME_TO_MEEP[name]
    if name.lower() not in materials_lower:
        raise ValueError(f"material, name = {name!r} not in {MATERIALS}")

    name = MATERIALS[materials_lower.index(name.lower())]
    medium = getattr(mat, name)

    if dispersive:
        return medium
    else:
        return mp.Medium(epsilon=medium.epsilon(1 / wavelength)[0][0])


def get_index(
    wavelength: float = 1.55, name: str = "Si", dispersive: bool = False
) -> float:
    """Returns refractive index from Meep's material database.

    Args:
        name: material name
        wavelength: wavelength (um)
        dispersive: True for built-in Meep index model,
            False for simple, non-dispersive model

    Note:
        Using the built-in models can be problematic at low resolution.
        If fields are NaN or Inf, increase resolution or use a non-dispersive model

    """
    medium = get_material(name=name, wavelength=wavelength, dispersive=dispersive)

    epsilon_matrix = medium.epsilon(1 / wavelength)
    epsilon11 = epsilon_matrix[0][0]
    return float(epsilon11.real ** 0.5)


def test_index():
    n = get_index(name="sin")
    assert np.isclose(n, 1.9962797317138816)


# si = partial(get_index, name="Si")
# sio2 = partial(get_index, name="SiO2")


if __name__ == "__main__":
    n = get_index(name="sio2")
    print(n, type(n))

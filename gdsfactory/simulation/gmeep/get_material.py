from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Union

import meep as mp
import meep.materials as mat
import numpy as np

from gdsfactory.materials import material_name_to_meep as material_name_to_meep_default
from gdsfactory.pdk import get_material_index

MATERIALS = [m for m in dir(mat) if not m.startswith("_")]


def get_material(
    name: str = "si",
    wavelength: float = 1.55,
    dispersive: bool = False,
    material_name_to_meep: Optional[Dict[str, Union[str, float]]] = None,
) -> mp.Medium:
    """Returns Meep Medium from database.

    Args:
        name: material name.
        wavelength: wavelength (um).
        dispersive: True for built-in Meep index model,
            False for simple, non-dispersive model.
        material_name_to_meep: dispersive materials have a wavelength
            dependent index. Maps layer_stack names with meep material database names.

    Note:
        Using the built-in models can be problematic at low resolution.

    """
    material_name_to_meep_new = material_name_to_meep or {}
    material_name_to_meep = material_name_to_meep_default.copy()
    material_name_to_meep.update(**material_name_to_meep_new)

    materials = [material.lower() for material in MATERIALS]
    name = name.lower()

    if dispersive:
        if name in material_name_to_meep.keys():
            name_or_index = material_name_to_meep[name]
        else:
            valid_materials = list(material_name_to_meep.keys())
            raise ValueError(
                f"name = {name!r} not in material_name_to_meep {valid_materials}"
            )
    else:
        material_index = get_material_index(name, wavelength)
        return mp.Medium(epsilon=material_index**2)

    if not isinstance(name_or_index, str):
        return mp.Medium(epsilon=name_or_index**2)

    name = name_or_index.lower()
    if name not in materials:
        raise ValueError(f"material, name = {name!r} not in {MATERIALS}")
    name = MATERIALS[materials.index(name)]
    medium = getattr(mat, name)
    if dispersive:
        return medium
    try:
        return mp.Medium(epsilon=medium.epsilon(1 / wavelength)[0][0])
    except ValueError as e:
        print(f"material = {name!r} wavelength={wavelength}")
        raise e


def get_index(
    wavelength: float = 1.55,
    name: str = "Si",
    dispersive: bool = False,
) -> float:
    """Returns refractive index from Meep's material database.

    Args:
        name: material name.
        wavelength: wavelength (um).
        dispersive: True for built-in Meep index model,
            False for simple, non-dispersive model.

    Note:
        Using the built-in models can be problematic at low resolution.
        If fields are NaN or Inf, increase resolution or use a non-dispersive model.

    """
    medium = get_material(
        name=name,
        wavelength=wavelength,
        dispersive=dispersive,
    )

    epsilon_matrix = medium.epsilon(1 / wavelength)
    epsilon11 = epsilon_matrix[0][0]
    return float(epsilon11.real**0.5)


def test_index() -> None:
    n = get_index(name="sin")
    n_reference = 1.9983425877199599
    assert np.isclose(n, n_reference), n


si = partial(get_index, name="Si")
sio2 = partial(get_index, name="SiO2")


if __name__ == "__main__":
    test_index()
    # n = get_index(name="Si", wavelength=1.31)
    # print(n, type(n))

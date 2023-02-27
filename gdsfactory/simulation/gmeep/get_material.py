from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Union

import meep as mp
import meep.materials as mat
import numpy as np

from gdsfactory.materials import material_name_to_meep as material_name_to_meep_default

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

    if name not in material_name_to_meep and name not in materials:
        raise KeyError(f"material {name!r} not found in available materials")

    meep_name = material_name_to_meep[name]

    if isinstance(meep_name, (int, float)):
        # if material is only a number, we can return early regardless of dispersion
        return mp.Medium(index=meep_name)

    material = getattr(mat, meep_name)

    if dispersive:
        return material

    # now what's left is the case of having a dispersive meep medium but a simulation
    # without dispersion, so we extract the permittivity at the correct wavelength
    try:
        return mp.Medium(epsilon=material.epsilon(1 / wavelength)[0][0])
    except ValueError as e:
        print(f"material = {name!r} not defined for wavelength={wavelength}")
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
    n_reference = 1.9962797317138816
    assert np.isclose(n, n_reference), n


si = partial(get_index, name="Si")
sio2 = partial(get_index, name="SiO2")


if __name__ == "__main__":
    test_index()
    # n = get_index(name="Si", wavelength=1.31)
    # print(n, type(n))

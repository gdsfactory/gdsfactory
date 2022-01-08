import meep as mp
import meep.materials as mat

# Dictionary for materials with different names in Meep
MATERIAL_NAME_TO_MEEP = {
    "sin": "Si3N4_NIR",
}

MATERIALS = [m for m in dir(mat) if not m.startswith("_")]


def get_material(
    name: str = "Si", wavelength: float = 1.55, dispersive: bool = False
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
    return epsilon11 ** 0.5


# si = partial(get_index, name="Si")
# sio2 = partial(get_index, name="SiO2")


if __name__ == "__main__":
    print(get_index(name="sin"))

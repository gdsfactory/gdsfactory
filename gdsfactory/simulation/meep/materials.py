from functools import partial

import meep as mp
import meep.materials as mat

MATERIALS = [m for m in dir(mat) if not m.startswith("_")]


def get_index(
    wavelength: float = 1.55,
    name: str = "Si",
) -> float:
    """Returns refractive index from Meep's material database.

    Args:
        name: material name
        wavelength: wavelength (um)

    """
    if name not in MATERIALS:
        raise ValueError(f"{name} not in {MATERIALS}")

    medium = getattr(mat, name)
    epsilon_matrix = medium.epsilon(1 / wavelength)
    epsilon11 = epsilon_matrix[0][0]
    return epsilon11 ** 0.5


def get_material(
    wavelength: float = 1.55,
    name: str = "Si",
) -> mp.Medium:
    """Returns Meep Medium from database

    Args:
        name: material name
        wavelength: wavelength (um)

    """

    # FIXME: need to remove this. If I remove this, then I get no fields.
    if name == "SiO2":
        return mp.Medium(epsilon=2.25)
    elif name.lower() == "si":
        return mp.Medium(epsilon=12)
    elif name.lower() == "sin":
        return mp.Medium(epsilon=4)
    else:
        raise ValueError(f"material, name = {name} not in {MATERIALS}")

    return getattr(mat, name)


si = partial(get_index, name="Si")

sio2 = partial(get_index, name="SiO2")


if __name__ == "__main__":
    print(get_index(name="Si"))

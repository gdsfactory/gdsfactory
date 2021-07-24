from functools import partial
import meep.materials as mat
import meep as mp

MATERIALS = [m for m in dir(mat) if not m.startswith("_")]


def get_index(
    wavelength: float = 1.55,
    material: str = "Si",
) -> float:
    """Returns refractive index from Meep's material database.

    Args:
        material: material name
        wavelength: wavelength (um)

    """
    if material not in MATERIALS:
        raise ValueError(f"{material} not in {MATERIALS}")

    medium = getattr(mat, material)
    epsilon_matrix = medium.epsilon(1 / wavelength)
    epsilon11 = epsilon_matrix[0][0]
    return epsilon11 ** 0.5


def get_material(
    wavelength: float = 1.55,
    material: str = "Si",
) -> mp.Medium:
    """Returns Meep Medium from database

    Args:
        material: material name
        wavelength: wavelength (um)

    """
    if material not in MATERIALS:
        raise ValueError(f"{material} not in {MATERIALS}")

    return getattr(mat, material)


si = partial(get_index, material="Si")

sio2 = partial(get_index, material="SiO2")


if __name__ == "__main__":
    print(get_index(material="Si"))

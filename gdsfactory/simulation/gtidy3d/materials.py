from typing import Optional

import tidy3d as td
from tidy3d.material_library import material_library

MATERIAL_NAME_TO_MEDIUM = {
    "si": material_library["cSi"]["Li1993_293K"],
    "csi": material_library["cSi"]["Li1993_293K"],
    "sio2": material_library["SiO2"]["Horiba"],
    "sin": material_library["Si3N4"]["Horiba"],
}


def get_epsilon(
    wavelength: float = 1.55,
    name: str = "cSi",
) -> float:
    """Return permittivity from material database.

    Args:
        wavelength: wavelength (um)
        name: material name
    """
    medium = get_medium(name=name)
    frequency = td.C_0 / wavelength
    return medium.eps_model(frequency)


def get_medium(name: str, n: Optional[float] = None) -> td.Medium:
    """Return Medium from materials database

    Args:
        name: material name
        n: optional index

    """

    name = name.lower()

    if n:
        m = td.Medium(permittivity=n ** 2)

    elif name == "sio2":
        m = td.Medium(permittivity=1.45 ** 2)
    elif name in ["csi", "si"]:
        m = td.Medium(permittivity=3.48 ** 2)

    elif name in ["sin", "si3n4"]:
        m = td.Medium(permittivity=2.0 ** 2)
    elif name in MATERIAL_NAME_TO_MEDIUM:
        m = MATERIAL_NAME_TO_MEDIUM[name]

    else:
        materials = list(MATERIAL_NAME_TO_MEDIUM.keys())

        raise ValueError(f"Material {name!r} not in {materials}")

    return m


if __name__ == "__main__":
    print(get_epsilon(name="cSi") ** 0.5)
    # m = get_medium(name="SiO2")
    # m = td.Medium(permittivity=1.45 ** 2)

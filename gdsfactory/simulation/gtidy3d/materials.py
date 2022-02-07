from typing import Optional

import tidy3d as td
from tidy3d.material_library import material_library

MATERIAL_NAME_TO_MEDIUM = {
    "si": material_library["cSi"]["Li1993_293K"],
    "cSi": material_library["cSi"]["Li1993_293K"],
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
    if name not in material_library:
        materials = list(material_library.keys())
        raise ValueError(f"{name!r} not in {materials}")

    # FIXME: need to remove this.
    if n:
        m = td.Medium(permittivity=n ** 2)

    elif name == "SiO2":
        m = td.Medium(permittivity=1.45 ** 2)
    # elif name in ["cSi", "si"]:
    #     m = td.Medium(permittivity=3.48 ** 2)
    elif name in ["SiN", "Si3N4"]:
        m = td.Medium(permittivity=2.0 ** 2)

    elif name in MATERIAL_NAME_TO_MEDIUM:
        m = MATERIAL_NAME_TO_MEDIUM[name]

    else:
        raise ValueError(
            f"not implemetned material {name!r} in {MATERIAL_NAME_TO_MEDIUM.keys()}"
        )

    return m


if __name__ == "__main__":
    print(get_epsilon(name="cSi") ** 0.5)
    # m = get_medium(name="SiO2")
    # m = td.Medium(permittivity=1.45 ** 2)

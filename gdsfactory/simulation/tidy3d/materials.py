from typing import Optional

import tidy3d as td
from tidy3d.material_library import material_library


def get_epsilon(
    wavelength: float = 1.55,
    name: str = "cSi",
) -> float:
    """Return permittivity from material database.

    Args:
        wavelength: wavelength (um)
        name: material name
    """
    medium = get_medium(wavelength=wavelength, name=name)
    frequency = 3e8 / wavelength
    return medium.eps_model(frequency)


def get_medium(
    wavelength: float = 1.55, name: str = "cSi", n: Optional[float] = None
) -> td.Medium:
    """Return Medium from materials database

    Args:
        wavelength: wavelength (um)
        name: material name
        n: optional index

    """
    if name not in material_library:
        materials = list(material_library.keys())
        raise ValueError(f"{name!r} not in {materials}")

    # FIXME: need to remove this.
    if n:
        return td.Medium(permittivity=n ** 2)
    if name == "SiO2":
        return td.Medium(permittivity=1.45 ** 2)
    elif name in ["cSi", "si"]:
        return td.Medium(permittivity=3.48 ** 2)
    elif name in ["SiN", "Si3N4"]:
        return td.Medium(permittivity=2.0 ** 2)
    else:
        raise ValueError(f"not implemetned material {name!r}")


if __name__ == "__main__":
    print(get_epsilon(name="cSi"))
    # m = get_medium(name="SiO2")
    # m = td.Medium(permittivity=1.45 ** 2)

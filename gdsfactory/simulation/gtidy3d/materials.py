from typing import Dict, Optional

import tidy3d as td
from tidy3d.components.medium import PoleResidue
from tidy3d.components.types import ComplexNumber
from tidy3d.material_library import material_library

MATERIAL_NAME_TO_MEDIUM = {
    "si": material_library["cSi"]["Li1993_293K"],
    "csi": material_library["cSi"]["Li1993_293K"],
    "sio2": material_library["SiO2"]["Horiba"],
    "sin": material_library["Si3N4"]["Horiba"],
    "si3n4": material_library["Si3N4"]["Horiba"],
}


def get_epsilon(
    wavelength: float = 1.55,
    name: str = "cSi",
    material_name_to_medium: Dict[str, PoleResidue] = MATERIAL_NAME_TO_MEDIUM,
) -> ComplexNumber:
    """Return permittivity from material database.

    Args:
        wavelength: wavelength (um)
        name: material name
        material_name_to_medium:
    """
    medium = get_medium(name=name, material_name_to_medium=material_name_to_medium)
    frequency = td.C_0 / wavelength
    return medium.eps_model(frequency)


def get_index(
    wavelength: float = 1.55,
    name: str = "cSi",
    material_name_to_medium: Dict[str, PoleResidue] = MATERIAL_NAME_TO_MEDIUM,
) -> float:
    """Return refractive index from material database.

    Args:
        wavelength: wavelength (um)
        name: material name
        material_name_to_medium:
    """

    eps_complex = get_epsilon(
        wavelength=wavelength,
        name=name,
        material_name_to_medium=material_name_to_medium,
    )
    n, k = td.Medium.eps_complex_to_nk(eps_complex)
    return n


def get_medium(
    name: str,
    n: Optional[float] = None,
    material_name_to_medium: Dict[str, PoleResidue] = MATERIAL_NAME_TO_MEDIUM,
) -> td.Medium:
    """Return Medium from materials database

    Args:
        name: material name from MATERIAL_NAME_TO_MEDIUM
        n: Optional index
        material_name_to_medium:
    """
    name = name.lower()

    if n:
        m = td.Medium(permittivity=n ** 2)
    # elif name == "sio2":
    #     m = td.Medium(permittivity=1.45 ** 2)
    # elif name in ["csi", "si"]:
    #     m = td.Medium(permittivity=3.48 ** 2)
    # elif name in ["sin", "si3n4"]:
    #     m = td.Medium(permittivity=2.0 ** 2)
    elif name in MATERIAL_NAME_TO_MEDIUM:
        m = material_name_to_medium[name]
    else:
        materials = list(material_name_to_medium.keys())

        raise ValueError(f"Material {name!r} not in {materials}")

    return m


if __name__ == "__main__":
    print(get_index(name="cSi"))
    # m = get_medium(name="SiO2")
    # m = td.Medium(permittivity=1.45 ** 2)

import tidy3d as td
import tidy3d.material_library as mat

MATERIALS = [m for m in dir(mat) if not m.startswith("_")]


def get_index(
    wavelength: float = 1.55,
    name: str = "cSi",
) -> float:
    """Returns refractive index from material database.

    Args: wavelength: wavelength (um) name: material name

    """
    if name not in MATERIALS:
        raise ValueError(f"{name} not in {MATERIALS}")

    medium = getattr(mat, name)
    medium.name = name
    frequency = 3e8 / wavelength
    return medium().epsilon(frequency)


def get_material(
    wavelength: float = 1.55,
    name: str = "cSi",
) -> td.Medium:
    """Returns Medium from materials database

    Args:
        wavelength: wavelength (um)
        name: material name

    """
    if name not in MATERIALS:
        raise ValueError(f"{name} not in {MATERIALS}")

    # FIXME: need to remove this.
    if name == "SiO2":
        return td.Medium(epsilon=1.45)
    elif name in ["cSi", "si"]:
        return td.Medium(n=3.48)
    elif name in ["SiN", "Si3N4"]:
        return td.Medium(n=2.0)
    else:
        raise ValueError(f"not implemetned material, name = {name}")

    medium = getattr(mat, name)
    medium.name = name
    return medium


if __name__ == "__main__":
    # print(get_index(name="aSi"))
    # SiO2 = get_material(name="SiO2")
    pass

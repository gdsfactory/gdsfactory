import pathlib
from pathlib import Path
from typing import Dict, Tuple

from gdsfactory.component import Component
from gdsfactory.config import CONFIG
from gdsfactory.name import dict2name
from gdsfactory.tech import LAYER


def get_sparameters_path(
    component: Component,
    layer_to_material: Dict[Tuple[int, int], str],
    layer_to_thickness_nm: Dict[Tuple[int, int], int],
    dirpath: Path = CONFIG["sp"],
) -> Path:
    """Returns Sparameters filepath.

    Args:
        component:
        dirpath
        layer_to_material: GDSlayer to material alias (see aliases in gf.sp.write)
        layer_to_thickness_nm: GDSlayer to thickness (nm)
    """
    dirpath = pathlib.Path(dirpath)
    dirpath = (
        dirpath / component.function_name
        if hasattr(component, "function_name")
        else dirpath
    )
    dirpath.mkdir(exist_ok=True, parents=True)
    material2nm = {
        layer_to_material[layer]: layer_to_thickness_nm[layer]
        for layer in layer_to_thickness_nm.keys()
        if tuple(layer) in component.get_layers()
    }
    suffix = dict2name(**material2nm)
    return dirpath / f"{component.get_name_long()}_{suffix}.dat"


def test_get_sparameters_path() -> None:
    import gdsfactory as gf

    layer_to_thickness_nm_sample = {
        LAYER.WG: 220,
        LAYER.SLAB90: 90,
    }
    layer_to_material_sample = {
        LAYER.WG: "si",
        LAYER.SLAB90: "si",
    }

    c = gf.components.straight()
    p = get_sparameters_path(
        component=c,
        layer_to_thickness_nm=layer_to_thickness_nm_sample,
        layer_to_material=layer_to_material_sample,
    )
    print(p.stem)
    assert p.stem == "straight_S220"

    c = gf.components.straight(layer=LAYER.SLAB90)
    p = get_sparameters_path(
        c,
        layer_to_thickness_nm=layer_to_thickness_nm_sample,
        layer_to_material=layer_to_material_sample,
    )
    print(p.stem)
    assert p.stem == "straight_L3_0_S90"


if __name__ == "__main__":
    # import gdsfactory as gf
    # c = gf.components.straight()
    # p = get_sparameters_path(c)
    # print(p)

    test_get_sparameters_path()

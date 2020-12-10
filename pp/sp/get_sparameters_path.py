import pathlib
from pathlib import PosixPath
from typing import Dict, Tuple

from pp.component import Component
from pp.config import CONFIG
from pp.layers import LAYER, layer2material, layer2nm
from pp.name import dict2name


def get_sparameters_path(
    component: Component,
    dirpath: PosixPath = CONFIG["sp"],
    layer2material: Dict[Tuple[int, int], str] = layer2material,
    layer2nm: Dict[Tuple[int, int], int] = layer2nm,
) -> PosixPath:
    """Returns Sparameters filepath.

    Args:
        component:
        dirpath
        layer2material: GDSlayer to material alias (see aliases in pp.sp.write)
        layer2nm: GDSlayer to thickness (nm)
    """
    dirpath = pathlib.Path(dirpath)
    dirpath = dirpath / component.function_name if component.function_name else dirpath
    dirpath.mkdir(exist_ok=True, parents=True)
    material2nm = {
        layer2material[layer]: layer2nm[layer]
        for layer in layer2nm.keys()
        if layer in component.get_layers()
    }
    suffix = dict2name(**material2nm)
    return dirpath / f"{component.get_name_long()}_{suffix}.dat"


def test_get_sparameters_path():
    import pp

    layer2nm = {
        LAYER.WG: 220,
        LAYER.SLAB90: 90,
    }
    layer2material = {
        LAYER.WG: "si",
        LAYER.SLAB90: "si",
    }

    c = pp.c.waveguide()
    p = get_sparameters_path(c, layer2nm=layer2nm, layer2material=layer2material)
    print(p.stem)
    assert p.stem == "waveguide_S220"

    c = pp.c.waveguide(layer=LAYER.SLAB90)
    p = get_sparameters_path(c, layer2nm=layer2nm, layer2material=layer2material)
    print(p.stem)
    assert p.stem == "waveguide_L3_0_S90"


if __name__ == "__main__":
    # import pp
    # c = pp.c.waveguide()
    # p = get_sparameters_path(c)
    # print(p)

    test_get_sparameters_path()

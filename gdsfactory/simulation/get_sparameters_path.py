import hashlib
import pathlib
from pathlib import Path

from gdsfactory.component import Component
from gdsfactory.config import CONFIG
from gdsfactory.name import clean_value
from gdsfactory.tech import LAYER, LAYER_STACK
from gdsfactory.types import SimulationSuffix


def get_sparameters_path(
    component: Component,
    dirpath: Path = CONFIG["sparameters"],
    suffix: SimulationSuffix = ".dat",
    **kwargs,
) -> Path:
    """Return Sparameters filepath.
    The returned filepath has a hash of all the parameters.

    Args:
        component:
        dirpath:
        suffix: .dat for interconnect, .csv for meep
        kwargs: simulation settings
    """

    dirpath = pathlib.Path(dirpath)
    dirpath = (
        dirpath / component.function_name
        if hasattr(component, "function_name")
        else dirpath
    )
    dirpath.mkdir(exist_ok=True, parents=True)

    settings_list = [
        f"{key}={clean_value(kwargs[key])}" for key in sorted(kwargs.keys())
    ]
    settings_string = "_".join(settings_list)
    settings_hash = hashlib.md5(settings_string.encode()).hexdigest()[:8]
    return dirpath / f"{component.name}_{settings_hash}{suffix}"


def test_get_sparameters_path() -> None:
    import gdsfactory as gf

    layer_to_thickness_sample = {
        LAYER.WG: 220e-3,
        LAYER.SLAB90: 90e-3,
    }
    layer_to_material_sample = {
        LAYER.WG: "si",
        LAYER.SLAB90: "si",
    }

    c = gf.components.straight()
    p = get_sparameters_path(
        component=c,
        layer_to_thickness=layer_to_thickness_sample,
        layer_to_material=layer_to_material_sample,
    )
    name1 = "straight_557fca9f"
    name2 = "straight_cf5c9898_557fca9f"
    name3 = "straight_cf5c9898_b4ea3a4d"

    assert p.stem == name1, p.stem

    c = gf.components.straight(layer=LAYER.SLAB90)
    p = get_sparameters_path(
        c,
        layer_to_thickness=layer_to_thickness_sample,
        layer_to_material=layer_to_material_sample,
    )
    assert p.stem == name2, p.stem

    c = gf.components.straight(layer=LAYER.SLAB90)
    p = get_sparameters_path(c, layer_stack=LAYER_STACK)
    assert p.stem == name3, p.stem


if __name__ == "__main__":
    # import gdsfactory as gf
    # c = gf.components.straight()
    # p = get_sparameters_path(c)
    # print(p)

    test_get_sparameters_path()

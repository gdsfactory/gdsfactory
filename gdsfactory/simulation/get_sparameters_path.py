import hashlib
import pathlib
from functools import partial
from pathlib import Path

import pandas as pd

from gdsfactory.config import CONFIG
from gdsfactory.name import clean_value
from gdsfactory.tech import LAYER, LAYER_STACK
from gdsfactory.types import ComponentOrFactory


def _get_sparameters_path(
    component: ComponentOrFactory,
    dirpath: Path = CONFIG["sparameters"],
    **kwargs,
) -> Path:
    """Return Sparameters CSV filepath.
    hashes of all simulation settings to get a consitent and unique name.

    Args:
        component: component or component factory.
        dirpath: directory path to store sparameters
        kwargs: simulation settings
    """

    component = component() if callable(component) else component

    dirpath = pathlib.Path(dirpath)
    dirpath = (
        dirpath / component.function_name
        if hasattr(component, "function_name")
        else dirpath
    )
    dirpath.mkdir(exist_ok=True, parents=True)

    kwargs_list = [f"{key}={clean_value(kwargs[key])}" for key in sorted(kwargs.keys())]
    kwargs_string = "_".join(kwargs_list)
    kwargs_hash = hashlib.md5(kwargs_string.encode()).hexdigest()[:8]
    return dirpath / f"{component.name}_{kwargs_hash}.csv"


def _get_sparameters_data(**kwargs) -> pd.DataFrame:
    """Returns Sparameters data in a pandas DataFrame.

    Keyword Args:
        component: component
        dirpath: directory path to store sparameters
        kwargs: simulation settings
    """
    filepath = _get_sparameters_path(**kwargs)
    return pd.read_csv(filepath)


get_sparameters_path_meep = partial(_get_sparameters_path, tool="meep")
get_sparameters_path_lumerical = partial(_get_sparameters_path, tool="lumerical")

get_sparameters_data_meep = partial(_get_sparameters_data, tool="meep")
get_sparameters_data_lumerical = partial(_get_sparameters_data, tool="lumerical")


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

    name1 = "straight_713b8220"
    name2 = "straight_cf5c9898_713b8220"
    name3 = "straight_cf5c9898_5f6ae1fd"
    name4 = "straight_2031115e"

    c = gf.components.straight()
    p = get_sparameters_path_lumerical(
        component=c,
        layer_to_thickness=layer_to_thickness_sample,
        layer_to_material=layer_to_material_sample,
    )
    assert p.stem == name1, p.stem

    c = gf.components.straight(layer=LAYER.SLAB90)
    p = get_sparameters_path_lumerical(
        c,
        layer_to_thickness=layer_to_thickness_sample,
        layer_to_material=layer_to_material_sample,
    )
    assert p.stem == name2, p.stem

    c = gf.components.straight(layer=LAYER.SLAB90)
    p = get_sparameters_path_meep(c, layer_stack=LAYER_STACK)
    assert p.stem == name3, p.stem

    c = gf.components.straight()
    p = get_sparameters_path_meep(
        component=c,
        layer_to_thickness=layer_to_thickness_sample,
        layer_to_material=layer_to_material_sample,
    )
    assert p.stem == name4, p.stem


if __name__ == "__main__":
    # import gdsfactory as gf
    # c = gf.components.straight()
    # p = get_sparameters_path(c)
    # print(p)

    test_get_sparameters_path()

import hashlib
import pathlib
from copy import deepcopy
from functools import partial
from pathlib import Path

import pandas as pd

import gdsfactory as gf
from gdsfactory.config import sparameters_path
from gdsfactory.name import clean_value
from gdsfactory.tech import LAYER_STACK
from gdsfactory.types import ComponentSpec


def get_kwargs_hash(**kwargs) -> str:
    """Returns kwargs parameters hash."""
    kwargs_list = [f"{key}={clean_value(kwargs[key])}" for key in sorted(kwargs.keys())]
    kwargs_string = "_".join(kwargs_list)
    return hashlib.md5(kwargs_string.encode()).hexdigest()[:8]


def _get_sparameters_path(
    component: ComponentSpec,
    dirpath: Path = sparameters_path,
    **kwargs,
) -> Path:
    """Return Sparameters CSV filepath.
    hashes of all simulation settings to get a consitent and unique name.

    Args:
        component: component or component factory.
        dirpath: directory path to store sparameters.
        kwargs: simulation settings.
    """

    component = gf.get_component(component)

    dirpath = pathlib.Path(dirpath)
    dirpath = (
        dirpath / component.function_name
        if hasattr(component, "function_name")
        else dirpath
    )
    dirpath.mkdir(exist_ok=True, parents=True)
    return dirpath / f"{component.name}_{get_kwargs_hash(**kwargs)}.csv"


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
get_sparameters_path_lumerical = partial(
    _get_sparameters_path, layer_stack=LAYER_STACK, tool="lumerical"
)
get_sparameters_path_tidy3d = partial(
    _get_sparameters_path, layer_stack=LAYER_STACK, tool="tidy3d"
)

get_sparameters_data_meep = partial(_get_sparameters_data, tool="meep")
get_sparameters_data_lumerical = partial(
    _get_sparameters_data, layer_stack=LAYER_STACK, tool="lumerical"
)


def test_get_sparameters_path(test: bool = True) -> None:
    import gdsfactory as gf

    nm = 1e-3
    layer_stack2 = deepcopy(LAYER_STACK)
    layer_stack2.layers["core"].thickness = 230 * nm

    c = gf.components.straight()

    p1 = get_sparameters_path_lumerical(component=c)
    p2 = get_sparameters_path_lumerical(component=c, layer_stack=layer_stack2)
    p3 = get_sparameters_path_lumerical(c, material_name_to_lumerical=dict(si=3.6))

    if test:
        name1 = "straight_24e96fd6"
        name2 = "straight_6b279ab9"
        name3 = "straight_791f2fdf"

        assert p1.stem == name1, p1.stem
        assert p2.stem == name2, p2.stem
        assert p3.stem == name3, p3.stem
    else:
        print(f"name1 = {p1.stem!r}")
        print(f"name2 = {p2.stem!r}")
        print(f"name3 = {p3.stem!r}")


if __name__ == "__main__":
    # import gdsfactory as gf

    # c = gf.components.mmi1x2()
    # p = get_sparameters_path_lumerical(c)
    # print(p)

    test_get_sparameters_path(test=False)
    # test_get_sparameters_path(test=True)

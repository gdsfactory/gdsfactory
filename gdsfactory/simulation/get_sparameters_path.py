from __future__ import annotations

import hashlib
import pathlib
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np

import gdsfactory as gf
from gdsfactory.name import clean_value
from gdsfactory.pdk import get_sparameters_path
from gdsfactory.typings import ComponentSpec


def get_kwargs_hash(**kwargs) -> str:
    """Returns kwargs parameters hash."""
    kwargs_list = [f"{key}={clean_value(kwargs[key])}" for key in sorted(kwargs.keys())]
    kwargs_string = "_".join(kwargs_list)
    return hashlib.md5(kwargs_string.encode()).hexdigest()


def get_component_hash(component: gf.Component) -> str:
    gdspath = pathlib.Path(component.write_gds())
    h = hashlib.md5(gdspath.read_bytes()).hexdigest()
    gdspath.unlink()
    return h


def _get_sparameters_path(
    component: ComponentSpec,
    dirpath: Optional[Path] = None,
    **kwargs,
) -> Path:
    """Return Sparameters npz filepath hashing simulation settings for \
            a consistent unique name.

    Args:
        component: component or component factory.
        dirpath: directory to store sparameters in CSV.
            Defaults to active Pdk.sparameters_path.
        kwargs: simulation settings.

    """
    dirpath = dirpath or get_sparameters_path()
    component = gf.get_component(component)

    dirpath = pathlib.Path(dirpath)
    dirpath = (
        dirpath / component.function_name
        if hasattr(component, "function_name")
        else dirpath
    )

    component_hash = get_component_hash(component)
    kwargs_hash = get_kwargs_hash(**kwargs)
    simulation_hash = hashlib.md5((component_hash + kwargs_hash).encode()).hexdigest()

    dirpath.mkdir(exist_ok=True, parents=True)
    return dirpath / f"{component.name}_{simulation_hash}.npz"


def _get_sparameters_data(**kwargs) -> np.ndarray:
    """Returns Sparameters data in a pandas DataFrame.

    Keyword Args:
        component: component.
        dirpath: directory path to store sparameters.
        kwargs: simulation settings.

    """
    filepath = _get_sparameters_path(**kwargs)
    return np.load(filepath)


get_sparameters_path_meow = partial(_get_sparameters_path, tool="meow")

get_sparameters_path_meep = partial(_get_sparameters_path, tool="meep")
get_sparameters_path_lumerical = partial(_get_sparameters_path, tool="lumerical")
get_sparameters_path_tidy3d = partial(_get_sparameters_path, tool="tidy3d")

get_sparameters_data_meep = partial(_get_sparameters_data, tool="meep")
get_sparameters_data_lumerical = partial(_get_sparameters_data, tool="lumerical")
get_sparameters_data_tidy3d = partial(_get_sparameters_data, tool="tidy3d")


if __name__ == "__main__":
    c = gf.components.mmi1x2()
    p = get_sparameters_path_lumerical(c)
    sp = np.load(p)
    spd = dict(sp)
    print(spd)

    # test_get_sparameters_path(test=False)
    # test_get_sparameters_path(test=True)

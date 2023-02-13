from __future__ import annotations

import hashlib
import pathlib
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER_STACK
from gdsfactory.name import clean_value
from gdsfactory.pdk import get_modes_path
from gdsfactory.typings import ComponentSpec


def get_kwargs_hash(**kwargs) -> str:
    """Returns kwargs parameters hash."""
    kwargs_list = [f"{key}={clean_value(kwargs[key])}" for key in sorted(kwargs.keys())]
    kwargs_string = "_".join(kwargs_list)
    return hashlib.md5(kwargs_string.encode()).hexdigest()[:8]


def _get_modes_path(
    component: ComponentSpec,
    dirpath: Optional[Path] = None,
    extension: Optional[str] = "npz",
    **kwargs,
) -> Path:
    """Return modes npz filepath hashing simulation settings for \
            a consistent unique name.

    Args:
        component: component.
        xsection_bounds: line where to take a cross-sectional mesh for mode solving,
        dirpath: directory to store sparameters in CSV.
            Defaults to active Pdk.sparameters_path.
        extension: to append to the end of the file.
        kwargs: simulation settings.

    """
    dirpath = dirpath or get_modes_path()
    component = gf.get_component(component)

    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(exist_ok=True, parents=True)
    return dirpath / f"{component.name}_{get_kwargs_hash(**kwargs)}.{extension}"


def _get_modes_data(**kwargs) -> np.ndarray:
    """Returns modes data in a pandas DataFrame.

    Keyword Args:
        cross_section: cross_section or cross_section factory.
        dirpath: directory path to store sparameters.
        kwargs: simulation settings.

    """
    filepath = _get_modes_path(**kwargs)
    return np.load(filepath)


get_modes_path_femwell = partial(_get_modes_path, tool="femwell")


def test_get_modes_path(test: bool = True) -> None:
    import gdsfactory as gf

    nm = 1e-3
    layer_stack2 = deepcopy(LAYER_STACK)
    layer_stack2.layers["core"].thickness = 230 * nm

    cross_section = gf.cross_section.strip()

    p1 = get_modes_path_femwell(cross_section=cross_section)
    p2 = get_modes_path_femwell(cross_section=cross_section, layer_stack=layer_stack2)
    p3 = get_modes_path_femwell(
        cross_section=cross_section, material_name_to_lumerical=dict(si=3.6)
    )

    if test:
        name1 = "xs_adfc05a6_782ce72c"
        name2 = "xs_adfc05a6_a01df6f8"
        name3 = "xs_adfc05a6_e88b3d6d"

        assert p1.stem == name1, p1.stem
        assert p2.stem == name2, p2.stem
        assert p3.stem == name3, p3.stem
    else:
        print(f"name1 = {p1.stem!r}")
        print(f"name2 = {p2.stem!r}")
        print(f"name3 = {p3.stem!r}")


if __name__ == "__main__":
    # cross_section = gf.cross_section.strip(name="strip")
    # p = get_modes_path_femwell(cross_section)
    # print(p)

    # modes = np.load(p)
    # modesd = dict(modes)
    # print(modesd)

    # test_get_modes_path(test=False)
    test_get_modes_path(test=False)

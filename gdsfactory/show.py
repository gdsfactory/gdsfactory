from __future__ import annotations

import typing

import pathlib
from typing import Union
import tempfile

from gdsfactory import klive


if typing.TYPE_CHECKING:
    from gdsfactory.component import Component


def show(component: Union[Component, str, pathlib.Path], **kwargs) -> None:
    """Write GDS and show Component in KLayout.

    Args:
        component: Component or GDS path.

    Keyword Args:
        filename: GDS file path to write to.
        gdsdir: directory for the GDS file. Defaults to /tmp/.
        unit: unit size for objects in library. 1um by default.
        precision: for object dimensions in the library (m). 1nm by default.
        timestamp: Defaults to 2019-10-25. If None uses current time.

    """
    if isinstance(component, pathlib.Path):
        component = str(component)
        return klive.show(component)
    elif isinstance(component, str):
        return klive.show(component)
    elif component is None:
        raise ValueError(
            "Component is None, make sure that your function returns the component"
        )

    elif hasattr(component, "write"):
        gdsdir = kwargs.get(
            "gdsdir", pathlib.Path(tempfile.TemporaryDirectory().name) / "gdsfactory"
        )
        gdsdir.mkdir(exist_ok=True, parents=True)
        with_oasis = kwargs.get("with_oasis", True)
        if with_oasis:
            filename = kwargs.get("filename", gdsdir / f"{component.name}.oas")
        else:
            filename = kwargs.get("filename", gdsdir / f"{component.name}.gds")
        gdsdir = pathlib.Path(gdsdir)
        component.write(filename=filename, **kwargs)
        klive.show(filename)
    else:
        raise ValueError(
            f"Component is {type(component)!r}, make sure pass a Component or a path"
        )

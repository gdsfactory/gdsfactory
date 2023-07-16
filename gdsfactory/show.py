from __future__ import annotations

import pathlib
from typing import Optional, Union

from gdsfactory import klive
from gdsfactory.component import Component


def show(
    component: Union[Component, str, pathlib.Path],
    technology: Optional[str] = None,
    **kwargs,
) -> None:
    """Write GDS and show Component in KLayout.

    Args:
        component: Component or GDS path.
        technology: Name of KLayout technology to load when displaying component.

    Keyword Args:
        gdspath: GDS file path to write to.
        gdsdir: directory for the GDS file. Defaults to /tmp/.
        unit: unit size for objects in library. 1um by default.
        precision: for object dimensions in the library (m). 1nm by default.
        timestamp: Defaults to 2019-10-25. If None uses current time.

    """
    if isinstance(component, pathlib.Path):
        component = str(component)
        return klive.show(component, technology=technology)
    elif isinstance(component, str):
        return klive.show(component, technology=technology)
    elif component is None:
        raise ValueError(
            "Component is None, make sure that your function returns the component"
        )

    elif hasattr(component, "write_gds"):
        # don't raise warnings for uncached cells when simply showing
        gdspath = component.write_gds(
            logging=False, on_uncached_component="ignore", **kwargs
        )
        klive.show(gdspath, technology=technology)
    else:
        raise ValueError(
            f"Component is {type(component)!r}, make sure pass a Component or a path"
        )

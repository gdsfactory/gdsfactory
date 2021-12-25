import pathlib
from typing import Union

from gdsfactory import klive
from gdsfactory.cell import clear_cache as clear_cache_function
from gdsfactory.component import Component
from gdsfactory.config import logger


def show(
    component: Union[Component, str, pathlib.Path], clear_cache: bool = True, **kwargs
) -> None:
    """Write GDS and show Component in klayout

    Args:
        component
        clear_cache: clear_cache after showing the component

    Keyword Args:
        gdspath: GDS file path to write to.
        gdsdir: directory for the GDS file. Defaults to /tmp/
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

    elif isinstance(component, Component):
        gdspath = component.write_gds(logging=False, **kwargs)
        klive.show(gdspath)
        logger.info(f"Klayout show {component!r}")
    else:
        raise ValueError(
            f"Component is {type(component)}, make sure pass a Component or a path"
        )
    if clear_cache:
        clear_cache_function()

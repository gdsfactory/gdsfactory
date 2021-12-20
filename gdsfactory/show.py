import pathlib
from typing import Union

from gdsfactory import klive
from gdsfactory.cell import clear_cache as clear_cache_function
from gdsfactory.component import Component
from gdsfactory.config import logger


def show(
    component: Union[Component, str, pathlib.Path],
    clear_cache: bool = True,
) -> None:
    """Shows Component in klayout

    Args:
        component
        clear_cache: clear_cache after showing the component
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
        gdspath = component.write_gds(logging=False)
        klive.show(gdspath)
        logger.info(f"Klayout show {component!r}")
    else:
        raise ValueError(
            f"Component is {type(component)}, make sure pass a Component or a path"
        )
    if clear_cache:
        clear_cache_function()

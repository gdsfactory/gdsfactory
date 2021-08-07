import pathlib

from gdsfactory import klive
from gdsfactory.cell import clear_cache
from gdsfactory.component import Component


def show(component: Component, clears_cache: bool = True, **kwargs) -> None:
    """Shows Component in klayout

    Args:
        component
        clears_cache: clears_cache
        kwargs: settings for write_gds
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
        gdspath = component.write_gds(**kwargs)
        klive.show(gdspath)
    else:
        raise ValueError(
            f"Component is {type(component)}, make sure pass a Component or a path"
        )
    if clears_cache:
        clear_cache()

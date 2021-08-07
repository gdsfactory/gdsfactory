"""Dummy filing to keep density constant.
"""
from typing import Iterable, Optional, Tuple, Union

from phidl.geometry import fill_rectangle as _fill_rectangle

from gdsfactory.component import Component
from gdsfactory.component_from import phidl
from gdsfactory.types import Layer


def fill_rectangle(
    component: Component,
    fill_size: Tuple[float, float] = (5.0, 5.0),
    avoid_layers: Union[str, Iterable[Layer]] = "all",
    include_layers: Optional[Iterable[Layer]] = None,
    margin: float = 5.0,
    fill_layers: Iterable[Layer] = (0, 1, 3),
    fill_densities: Iterable[float] = (0.5, 0.25, 0.7),
    fill_inverted: Optional[bool] = None,
    bbox: Optional[Tuple[float, float]] = None,
) -> Component:
    """Creates a rectangular fill pattern and fills all empty areas
    in the input component

    Args:
        Component: Component to be filled
        fill_size: Rectangular size of the fill element
        avoid_layers : 'all' or list of layers
            Layers to be avoided (not filled) in D
        include_layers :
            Layers to be included (filled) in D, supercedes avoid_layers
        margin :
            Margin spacing around avoided areas -- fill will not come within
            `margin` of the geometry in D
        fill_layers : list of layers. fill pattern layers
        fill_densities : float between 0 and 1
            Defines the fill pattern density (1.0 == fully filled)
        fill_inverted : Inverts the fill pattern
        bbox: array-like[2][2]
            Limit the fill pattern to the area defined by this bounding box

    """

    component_filled = _fill_rectangle(
        component,
        fill_size=fill_size,
        avoid_layers=avoid_layers,
        include_layers=include_layers,
        margin=margin,
        fill_layers=fill_layers,
        fill_densities=fill_densities,
        fill_inverted=fill_inverted,
        bbox=bbox,
    )
    return phidl(component_filled)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    c = gf.add_padding(c)
    c << fill_rectangle(
        c,
        fill_layers=((2, 0),),
        fill_densities=(1.0),
        avoid_layers=((1, 0),),
        # bbox=(100.0, 100.0),
    )
    c.show()

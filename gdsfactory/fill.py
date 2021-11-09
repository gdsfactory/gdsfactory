"""Dummy fill to keep density constant."""
from typing import Optional, Union

from phidl.geometry import fill_rectangle as _fill_rectangle

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.read import from_phidl
from gdsfactory.types import Float2, Floats, Layers


@cell
def fill_rectangle(
    component: Component,
    fill_layers: Layers,
    fill_size: Float2 = (5.0, 5.0),
    avoid_layers: Optional[Layers] = None,
    include_layers: Optional[Layers] = None,
    margin: float = 5.0,
    fill_densities: Union[float, Floats] = (0.5, 0.25, 0.7),
    fill_inverted: bool = False,
    bbox: Optional[Float2] = None,
) -> Component:
    """Creates a rectangular fill pattern and fills all empty areas
    in the input component and returns a component that contains just the fill
    Dummy fill keeps density constant during fabrication

    Args:
        component: Component to fill
        fill_size: Rectangular size of the fill element
        avoid_layers: Layers to be avoided (not filled) in D
        include_layers: Layers to be filled, supercedes avoid_layers
        margin :
            Margin spacing around avoided areas -- fill will not come within
            `margin` of the geometry in D
        fill_layers: list of layers. fill pattern layers
        fill_densities: float between 0 and 1
            Defines the fill pattern density (1.0 == fully filled)
        fill_inverted: Inverts the fill pattern
        bbox: array-like[2][2]
            Limit the fill pattern to the area defined by this bounding box

    """

    component_filled = _fill_rectangle(
        component,
        fill_size=fill_size,
        avoid_layers=avoid_layers or "all",
        include_layers=include_layers,
        margin=margin,
        fill_layers=fill_layers,
        fill_densities=fill_densities,
        fill_inverted=fill_inverted,
        bbox=bbox,
    )
    return from_phidl(component_filled)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    c = gf.add_padding(c)
    c << fill_rectangle(
        c,
        fill_layers=((2, 0),),
        # fill_densities=(1.0,),
        fill_densities=1.0,
        avoid_layers=((1, 0),),
        # bbox=(100.0, 100.0),
    )
    c.show()

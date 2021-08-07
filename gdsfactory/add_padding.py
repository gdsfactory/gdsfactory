from typing import Iterable, List, Optional, Tuple

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.tech import TECH

LAYER = TECH.layer


def get_padding_points(
    component: Component,
    default: float = 50.0,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    right: Optional[float] = None,
    left: Optional[float] = None,
) -> List[float]:
    """Returns padding points for a component outline.

    Args:
        component
        default: default padding
        top: north padding
        bottom: south padding
        right: east padding
        left: west padding
    """
    c = component
    top = top if top is not None else default
    bottom = bottom if bottom is not None else default
    right = right if right is not None else default
    left = left if left is not None else default
    return [
        [c.xmin - left, c.ymin - bottom],
        [c.xmax + right, c.ymin - bottom],
        [c.xmax + right, c.ymax + top],
        [c.xmin - left, c.ymax + top],
    ]


def add_padding(
    component: Component,
    layers: Iterable[Tuple[int, int]] = (LAYER.PADDING,),
    **kwargs,
) -> Component:
    """Adds padding layers to a component.

    The cell name will be the same as the original component.

    Args:
        component
        layers: list of layers
        suffix for name
        default: default padding
        top: north padding
        bottom: south padding
        right: east padding
        left: west padding
    """
    points = get_padding_points(component, **kwargs)
    layers = layers or []
    for layer in layers:
        component.add_polygon(points, layer=layer)
    return component


@cell
def add_padding_container(
    component: Component,
    layers: Iterable[Tuple[int, int]] = (LAYER.PADDING,),
    **kwargs,
) -> Component:
    """Adds padding layers to a component inside a container.

    Returns the same ports as the component.

    Args:
        component
        layers: list of layers
        default: default padding
        top: north padding
        bottom: south padding
        right: east padding
        left: west padding
    """

    c = Component()
    cref = c << component

    points = get_padding_points(component, **kwargs)
    for layer in layers:
        c.add_polygon(points, layer=layer)
    c.ports = cref.ports
    return c


def add_padding_to_size(
    component: Component,
    layers: Iterable[Tuple[int, int]] = (LAYER.PADDING,),
    xsize: Optional[float] = None,
    ysize: Optional[float] = None,
    left: float = 0,
    bottom: float = 0,
) -> Component:
    """Returns component with padding layers on each side.

    New size is multiple of grid size
    """

    c = component
    top = abs(ysize - component.ysize) if ysize else 0
    right = abs(xsize - component.xsize) if xsize else 0
    points = [
        [c.xmin - left, c.ymin - bottom],
        [c.xmax + right, c.ymin - bottom],
        [c.xmax + right, c.ymax + top],
        [c.xmin - left, c.ymax + top],
    ]

    for layer in layers:
        component.add_polygon(points, layer=layer)

    return component


def test_container():
    import gdsfactory as gf

    c = gf.components.straight(length=128)
    cc = add_padding_container(component=c, layers=[(1, 0)])
    print(len(cc.settings["component"]))
    assert len(cc.settings["component"]) == 5

    cc = add_padding_container(component=c, layers=[(2, 0)])
    assert len(cc.settings["component"]) == 5

    cc = add_padding_container(component=c, layers=[(3, 0)])
    assert len(cc.settings["component"]) == 5

    # cc = add_padding_container(component=c, layers=[(4, 0)])
    # assert isinstance(cc.settings["component"], Component)
    # print(cc.settings["component"])
    # print(len(cc.settings["component"]))


if __name__ == "__main__":

    test_container()
    # c = gf.components.straight(length=128)
    # cc = add_padding_container(component=c, layers=[(2, 0)])
    # cc = add_padding_container(component=c, layers=[(2, 0)])
    # print(cc.settings["component"])
    # cc.show()
    # cc.pprint()

    # c = gf.components.straight(length=5)
    # cc = add_padding_to_size(component=c, xsize=10, layers=[(2, 0)])
    # cc.show()
    # print(cc.settings)
    # print(cc.ports)

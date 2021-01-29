from typing import Tuple

from phidl.device_layout import Label

from pp.component import Component
from pp.container import container
from pp.layers import LAYER


def add_label(component: Component, text: str = "hi") -> Label:
    return component.add_label(text)


@container
def container_instance(component):
    """Returns a container instance."""
    c = Component(name=f"i_{component.name}")
    c << component
    return c


@container
def _add_padding(
    component: Component,
    x: int = 50,
    y: int = 50,
    layers: Tuple[int, int] = (LAYER.PADDING),
    suffix: str = "p",
) -> Component:
    """Adds padding layers to component.
    This is just an example. For the real function see pp.add_padding.
    """
    c = Component(name=f"{component.name}_{suffix}")
    c << component
    points = [
        [c.xmin - x, c.ymin - y],
        [c.xmax + x, c.ymin - y],
        [c.xmax + x, c.ymax + y],
        [c.xmin - x, c.ymax + y],
    ]
    for layer in layers:
        c.add_polygon(points, layer=layer)
    return c


def test_container() -> Component:
    import pp

    old = pp.c.waveguide()
    suffix = "p"
    name = f"{old.name}_{suffix}"
    new = _add_padding(component=old, suffix=suffix)
    assert new != old, f"new component {new} should be different from {old}"
    assert new.name == name, f"new name {new.name} should be {name}"
    assert len(new.ports) == len(
        old.ports
    ), f"new component {len(new.ports)} ports should match original {len(old.ports)} ports"
    return new


def test_container_error() -> Component:
    import pytest

    import pp

    old = pp.c.waveguide()
    with pytest.raises(ValueError):
        _add_padding(component2=old)  # will raise an error
    return old


if __name__ == "__main__":

    # c1 = pp.c.waveguide(length=3, width=0.9)
    # c2 = pp.c.waveguide(length=3)
    # cc1 = container_instance(c1)
    # cc2 = container_instance(c2)
    c = test_container()
    c.show()

"""
a container creates a new component that contains the original component inside with some extra elements:

it makes sure that some of the important settings are copied from the original component to the new component

"""

from typing import Callable
import functools
from inspect import signature
import pp


def container(component_function: Callable) -> Callable:
    """decorator for creating a new component that copies properties from the original component

    - polarization
    - wavelength
    - test_protocol
    - data_analysis_protocol

    Functions decorated with container will return a new component

    .. plot::
      :include-source:

      import pp

      @pp.container
      def add_padding(component, suffix='p', layer=pp.LAYER.M1):
          c = pp.Component(name=f"{component.name}_{suffix}")
          c << component
          w, h = component.xsize, component.ysize
          points = [[w, h], [w, 0], [0, 0], [0, h]]
          c.add_polygon(points, layer=layer)
          return c

      c = pp.c.waveguide()
      cp = add_padding(c)
      pp.plotgds(c)

    """

    @functools.wraps(component_function)
    def wrapper(*args, **kwargs):
        old = kwargs.get("component")
        if not old and args:
            old = args[0]
        if callable(old):
            old = old()
        if not isinstance(old, pp.Component):
            raise ValueError(
                f"container {component_function.__name__} requires a component, got `{old}`"
            )
        old = old or kwargs.get("component")
        new = component_function(*args, **kwargs)

        sig = signature(component_function)
        new.settings.update(**{p.name: p.default for p in sig.parameters.values()})
        new.settings.update(**kwargs)
        new.settings["component"] = old.get_settings()
        new.settings["component_name"] = old.name
        new.settings["function_name"] = component_function.__name__
        new.test_protocol = new.test_protocol or old.test_protocol.copy()
        new.data_analysis_protocol = (
            new.data_analysis_protocol or old.data_analysis_protocol.copy()
        )
        new.wavelength = new.wavelength or old.wavelength
        new.polarization = new.polarization or old.polarization
        new.settings.pop("kwargs", "")
        return new

    return wrapper


@container
def add_padding(component, x=50, y=50, layers=[pp.LAYER.PADDING], suffix="p"):
    """ adds padding layers to component"""
    c = pp.Component(name=f"{component.name}_{suffix}")
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


def test_container():
    old = pp.c.waveguide()
    suffix = "p"
    name = f"{old.name}_{suffix}"
    new = add_padding(component=old, suffix=suffix)
    assert new != old, f"new component {new} should be different from {old}"
    assert new.name == name, f"new name {new.name} should be {name}"
    # assert len(new.ports) == len(
    #     old.ports
    # ), f"new component {len(new.ports)} ports should match original {len(old.ports)} ports"
    # assert len(new.settings) == len(
    #     old.settings
    # ), f"new component {new.settings} settings should match original {old.settings} settings"
    return new


def test_container2():
    old = pp.c.waveguide()
    suffix = "p"
    name = f"{old.name}_{suffix}"
    new = add_padding(old, suffix=suffix)
    assert new != old, f"new component {new} should be different from {old}"
    assert new.name == name, f"new name {new.name} should be {name}"
    # assert len(new.ports) == len(
    #     old.ports
    # ), f"new component {len(new.ports)} ports should match original {len(old.ports)} ports"
    return new


def test_container_error():
    import pytest

    old = pp.c.waveguide()
    with pytest.raises(ValueError):
        add_padding(component2=old)  # will raise an error
    return old


if __name__ == "__main__":
    c = test_container2()
    pp.show(c)

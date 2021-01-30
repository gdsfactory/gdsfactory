"""A container is a new component that contains the original component
- adds some extra elements (routes, grating couplers, labels ...)
- copies settings from the original component to the new component
- if you don't define ports on the new component also takes the ports from the original

"""

import functools
from inspect import signature
from typing import Callable, cast

from pp.component import Component

propagate_attributes = {
    "test_protocol",
    "data_analysis_protocol",
    "wavelength",
    "polarization",
}


def container(func: Callable) -> Callable:
    """Decorator for creating a new component
    contains the original component
    adds some geometry (labels, routes ...)
    copies properties from the original component

    - polarization
    - wavelength
    - test_protocol
    - data_analysis_protocol
    - ports (only if new component has no ports)

    Functions decorated with container will return a new component

    .. plot::
      :include-source:

      import pp

      @pp.container
      def _add_padding(component, suffix='p', layer=pp.LAYER.M1):
          c = pp.Component(name=f"{component.name}_{suffix}")
          c << component
          w, h = component.xsize, component.ysize
          points = [[w, h], [w, 0], [0, 0], [0, h]]
          c.add_polygon(points, layer=layer)
          return c

      c = pp.c.waveguide()
      cp = _add_padding(c)
      c.plot()

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        old = kwargs.get("component")
        if not old and args:
            old = args[0]
        if callable(old):
            old = old()
        if not isinstance(old, Component):
            raise ValueError(
                f"container {func.__name__} requires a component, got `{old}`"
            )
        old = cast(Component, old)
        name = kwargs.pop("name", "")
        old = old or kwargs.get("component")
        new = func(*args, **kwargs)
        assert isinstance(
            new, Component
        ), f"`{func.__name__}` function needs to return a Component, it returned `{new}` "

        sig = signature(func)
        new.settings.update(**{p.name: p.default for p in sig.parameters.values()})
        new.settings.update(**kwargs)
        new.settings["component"] = old.get_settings()
        new.function_name = func.__name__

        # if no ports defined it takes
        if len(new.ports) == 0:
            new.ports = old.ports

        for key in propagate_attributes:
            if hasattr(old, key) and key not in old.ignore:
                value = getattr(old, key)
                if value:
                    setattr(new, key, value)
        new.settings.pop("kwargs", "")
        if name:
            new.name = name
        return new

    return wrapper


if __name__ == "__main__":
    from pp.tests.test_container import test_container

    c = test_container()
    c.show()

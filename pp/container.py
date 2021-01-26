"""A container is a new component that contains the original component
- adds some extra elements (routes, grating couplers, labels ...)
- copies settings from the original component to the new component
- if you don't define ports on the new component also takes the ports from the original

"""

import functools
import hashlib
from inspect import signature
from typing import Callable

from pp.component import Component
from pp.config import MAX_NAME_LENGTH
from pp.name import get_component_name

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
      pp.plotgds(c)

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


@container
def containerize(component: Component, function: Callable, **kwargs) -> Component:
    """Returns a containerize component after applying a function.
    This is an alternative of using the @container decorator.
    However I recommend using the decorator when possible

    Args:
        component: to containerize
        function: that applies to component
        **kwargs: for the function

    .. code::

        import pp

        def add_label(component, text='hi'):
            return component.add_label(text)

        c = pp.c.waveguide()
        cc = pp.containerize(c, function=add_label, text='hi')

    """
    c = Component()
    component_type = f"{component.name}_{function.__name__}"
    name = get_component_name(component_type, **kwargs)
    if len(name) > MAX_NAME_LENGTH:
        c.name_long = name
        name = f"{component_type}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
    c.name = name
    c << component
    function(c, **kwargs)
    return c


if __name__ == "__main__":
    from pp.tests.test_container import test_containerize

    c = test_containerize()
    c.show()

import hashlib
import uuid
from functools import partial, wraps
from inspect import signature
from typing import Callable, Dict, Optional

from pp.component import Component
from pp.config import MAX_NAME_LENGTH
from pp.name import get_component_name

CACHE: Dict[str, Component] = {}


def clear_cache() -> None:
    """Clears the cache of components."""
    global CACHE
    CACHE = {}


def cell(
    func: Callable = None,
    *,
    autoname: bool = True,
    name: Optional[str] = None,
    uid: bool = False,
    cache: bool = True,
) -> Callable:
    """Cell Decorator.

    Args:
        autoname (bool): renames Component by with Keyword arguments
        name (str): Optional (ignored when autoname=True)
        uid (bool): adds a unique id to the name
        cache (bool): get component from the cache if it already exists

    Implements a cache so that if a component has already been build
    it will return the component from the cache.
    This avoids 2 exact cells that are not references of the same cell
    You can always over-ride this with `cache = False`.

    .. plot::
      :include-source:

      import pp

      @pp.cell
      def rectangle(size=(4,2), layer=0):
          c = pp.Component()
          w, h = size
          points = [[w, h], [w, 0], [0, 0], [0, h]]
          c.add_polygon(points, layer=layer)
          return c

      c = rectangle(layer=1)
      pp.plotgds(c)

    """

    if func is None:
        return partial(cell, autoname=autoname, name=name, uid=uid, cache=cache,)

    @wraps(func)
    def _cell(
        autoname: bool = autoname,
        name: Optional[str] = name,
        uid: bool = uid,
        cache: bool = cache,
        *args,
        **kwargs,
    ) -> Component:
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        arguments = ", ".join(args_repr + kwargs_repr)

        if args:
            raise ValueError(
                f"cell supports only Keyword args for `{func.__name__}({arguments})`"
            )

        component_type = func.__name__
        name = name or get_component_name(component_type, **kwargs)

        if uid:
            name += f"_{str(uuid.uuid4())[:8]}"

        kwargs.pop("ignore_from_name", [])
        sig = signature(func)

        # first_letters = [join_first_letters(k) for k in kwargs.keys() if k != "layer"]
        # keys = set(kwargs.keys()) - set(["layer"])
        # if not len(set(first_letters)) == len(first_letters):
        #     print(
        #         f"Warning! Possible Duplicated name in {component_type}. "
        #         f"Args {keys} have repeated first letters {first_letters}"
        #     )

        if "args" not in sig.parameters and "kwargs" not in sig.parameters:
            for key in kwargs.keys():
                if key not in sig.parameters.keys():
                    raise TypeError(
                        f"{component_type}() got an unexpected keyword argument `{key}`\n"
                        f"valid keyword arguments are {list(sig.parameters.keys())}"
                    )

        if cache and autoname and name in CACHE:
            return CACHE[name]
        else:
            assert callable(
                func
            ), f"{func} is not Callable, make sure you only use the @cell decorator with functions"
            component = func(**kwargs)
            assert isinstance(
                component, Component
            ), f"`{func.__name__}` function needs to return a Component, it returned `{component}` "
            component.module = func.__module__
            component.function_name = func.__name__

            if len(name) > MAX_NAME_LENGTH:
                component.name_long = name
                name = f"{component_type}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
            if autoname:
                component.name = name

            if not hasattr(component, "settings"):
                component.settings = {}
            component.settings.update(
                **{
                    p.name: p.default
                    for p in sig.parameters.values()
                    if not callable(p.default)
                }
            )
            component.settings.update(**kwargs)
            component.settings_changed = kwargs.copy()

            CACHE[name] = component
            return component

    return _cell


@cell(autoname=True)
def wg(length: int = 3, width: float = 0.5) -> Component:
    from pp.component import Component

    c = Component("waveguide")
    w = width / 2
    layer = (1, 0)
    c.add_polygon([(0, -w), (length, -w), (length, w), (0, w)], layer=layer)
    c.add_port(name="W0", midpoint=[0, 0], width=width, orientation=180, layer=layer)
    c.add_port(name="E0", midpoint=[length, 0], width=width, orientation=0, layer=layer)
    return c


@cell(autoname=False)
def wg2(length: int = 3, width: float = 0.5) -> Component:
    from pp.component import Component

    c = Component("waveguide")
    w = width / 2
    layer = (1, 0)
    c.add_polygon([(0, -w), (length, -w), (length, w), (0, w)], layer=layer)
    c.add_port(name="W0", midpoint=[0, 0], width=width, orientation=180, layer=layer)
    c.add_port(name="E0", midpoint=[length, 0], width=width, orientation=0, layer=layer)
    return c


@cell
def wg3(length=3, width=0.5):
    from pp.component import Component

    c = Component("waveguide")
    w = width / 2
    layer = (1, 0)
    c.add_polygon([(0, -w), (length, -w), (length, w), (0, w)], layer=layer)
    c.add_port(name="W0", midpoint=[0, 0], width=width, orientation=180, layer=layer)
    c.add_port(name="E0", midpoint=[length, 0], width=width, orientation=0, layer=layer)
    return c


def test_autoname_true() -> None:
    assert wg(length=3).name == "wg_L3"


def test_autoname_false() -> None:
    # print(wg2(length=3).name)
    assert wg2(length=3).name == "waveguide"


@cell
def _dummy(length: int = 3, wg_width: float = 0.5) -> Component:
    c = Component()
    return c


def test_autoname() -> None:
    name_base = _dummy().name
    assert name_base == "_dummy"

    name_int = _dummy(length=3).name
    assert name_int == "_dummy_L3"

    name_float = _dummy(wg_width=0.5).name
    assert name_float == "_dummy_WW500n"

    name_length_first = _dummy(length=3, wg_width=0.5).name
    name_width_first = _dummy(wg_width=0.5, length=3).name
    assert name_length_first == name_width_first

    name_float = _dummy(wg_width=0.5).name
    assert name_float == "_dummy_WW500n"


if __name__ == "__main__":
    import pp

    c = pp.c.waveguide()

    # test_autoname_true()
    # test_autoname_false()
    # test_autoname()

    # c = wg(length=3)
    # c = wg(length=3, autoname=False)

    # c = pp.c.waveguide()
    # c = wg3()
    # print(c)
    # pp.show(c)

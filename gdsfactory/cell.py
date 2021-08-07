import functools
import inspect
import uuid
from typing import Dict

from pydantic import validate_arguments

from gdsfactory.component import Component
from gdsfactory.name import get_component_name, get_name

CACHE: Dict[str, Component] = {}


def clear_cache() -> None:
    """Clears the cache of components."""
    global CACHE
    CACHE = {}


def print_cache():
    for k in CACHE:
        print(k)


def cell_without_validator(func):
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

      import gdsfactory as gf

      @gf.cell
      def rectangle(size=(4,2), layer=0)->gf.Component:
          c = gf.Component()
          w, h = size
          points = [[w, h], [w, 0], [0, 0], [0, h]]
          c.add_polygon(points, layer=layer)
          return c

      c = rectangle(layer=1)
      c.plot()

    """

    @functools.wraps(func)
    def _cell(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        arguments = ", ".join(args_repr + kwargs_repr)

        if args:
            raise ValueError(
                f"cell supports only Keyword args for `{func.__name__}({arguments})`"
            )

        cache = kwargs.pop("cache", True)
        component_type = func.__name__
        name = kwargs.pop("name", None)
        name = name or get_component_name(component_type, **kwargs)
        decorator = kwargs.pop("decorator", None)

        uid = kwargs.pop("uid", False)
        autoname = kwargs.pop("autoname", True)

        if uid:
            name += f"_{str(uuid.uuid4())[:8]}"

        name_long = name
        name = get_name(component_type=component_type, name=name)
        sig = inspect.signature(func)

        # first_letters = [join_first_letters(k) for k in kwargs.keys() if k != "layer"]
        # keys = set(kwargs.keys()) - set(["layer"])
        # if not len(set(first_letters)) == len(first_letters):
        #     print(
        #         f"Warning! Possible Duplicated name in {component_type}. "
        #         f"Args {keys} have repeated first letters {first_letters}"
        #     )

        if (
            "args" not in sig.parameters
            and "kwargs" not in sig.parameters
            and "waveguide_settings" not in sig.parameters
        ):
            for key in kwargs.keys():
                if key not in sig.parameters.keys():
                    raise TypeError(
                        f"{component_type}() got invalid argument `{key}`\n"
                        f"valid arguments are {list(sig.parameters.keys())}"
                    )

        if cache and name in CACHE:
            # print(f"CACHE {func.__name__}({kwargs_repr})")
            return CACHE[name]
        else:
            # print(f"BUILD {func.__name__}({kwargs_repr})")
            assert callable(
                func
            ), f"{func} got decorated with @cell! @cell decorator is only for functions"
            component = func(*args, **kwargs)
            if decorator:
                assert callable(
                    decorator
                ), f"decorator = {type(decorator)} needs to be callable"
                decorator(component)

            if "component" in kwargs and isinstance(kwargs.get("component"), Component):
                component_original = kwargs.pop("component")
                component_original = (
                    component_original()
                    if callable(component_original)
                    else component_original
                )
                component.settings["component"] = component_original.get_settings()

            if not isinstance(component, Component):
                raise ValueError(
                    f"`{func.__name__}` returned `{component}` and not a Component"
                )
            component.module = func.__module__
            component.function_name = func.__name__

            if autoname:
                component.name = name

            component.name_long = name_long

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
        return

    return _cell


def cell(func, *args, **kwargs):
    return cell_without_validator(validate_arguments(func), *args, **kwargs)


@cell
def wg(length: int = 3, width: float = 0.5) -> Component:
    from gdsfactory.component import Component

    c = Component("straight")
    w = width / 2
    layer = (1, 0)
    c.add_polygon([(0, -w), (length, -w), (length, w), (0, w)], layer=layer)
    c.add_port(name="W0", midpoint=[0, 0], width=width, orientation=180, layer=layer)
    c.add_port(name="E0", midpoint=[length, 0], width=width, orientation=0, layer=layer)
    return c


def test_autoname_true() -> None:
    assert wg(length=3).name == "wg_L3"


def test_autoname_false() -> None:
    c = wg(length=3, autoname=False)
    print(c.name)
    assert c.name == "straight"


def test_set_name() -> None:
    c = wg(length=3, name="hi_there")
    print(c.name)
    assert c.name == "hi_there"


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
    # test_raise_error_args()

    # c = gf.components.straight()

    # test_autoname_true()
    # test_autoname_false()
    # test_autoname()
    test_set_name()

    # c = wg(length=3)
    # c = wg(length=3, autoname=False)

    # c = gf.components.straight()
    # c = wg3()
    # print(c)
    # c.show()

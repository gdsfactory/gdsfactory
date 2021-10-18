"""
- INFO_VERSION: 1

"""
import functools
import hashlib
import inspect
from typing import Dict

import omegaconf
from pydantic import validate_arguments

from gdsfactory.component import Component, clean_dict
from gdsfactory.name import MAX_NAME_LENGTH, clean_name, clean_value

CACHE: Dict[str, Component] = {}
INFO_VERSION = 1


class CellReturnTypeError(ValueError):
    pass


def clear_cache() -> None:
    """Clears the cache of components."""
    global CACHE
    CACHE = {}


def print_cache():
    for k in CACHE:
        print(k)


def clean_doc(name: str) -> str:
    """Returns a clean docstring"""
    # replace_map = {
    #     " ": " ",
    #     "  ": " ",
    #     "   ": " ",
    #     "    ": ",",
    #     "     ": " ",
    #     "\n": " ",
    #     "\n\n": " ",
    # }
    # for k, v in list(replace_map.items()):
    #     name = name.replace(k, v)

    # name = ",".join(name.split('\n'))
    # name = " ".join(name.split())
    return name


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

      c = rectangle(layer=(1,0))
      c.plot()

    """

    @functools.wraps(func)
    def _cell(*args, **kwargs):
        prefix = kwargs.pop("prefix", func.__name__)
        autoname = kwargs.pop("autoname", True)
        cache = kwargs.pop("cache", True)
        info = kwargs.pop("info", omegaconf.DictConfig({}))

        sig = inspect.signature(func)
        args_as_kwargs = dict(zip(sig.parameters.keys(), args))
        args_as_kwargs.update(**kwargs)
        args_as_kwargs_string_list = [
            f"{key}={clean_value(args_as_kwargs[key])}"
            for key in sorted(args_as_kwargs.keys())
        ]
        arguments = "_".join(args_as_kwargs_string_list)
        arguments_hash = hashlib.md5(arguments.encode()).hexdigest()[:8]
        name_long = name = (
            clean_name(f"{prefix}_{arguments_hash}") if arguments else prefix
        )
        decorator = kwargs.pop("decorator", None)

        if len(name) > MAX_NAME_LENGTH:
            name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
            name = f"{name[:(MAX_NAME_LENGTH - 9)]}_{name_hash}"

        name_component = kwargs.pop("name", name)

        if (
            "args" not in sig.parameters
            and "kwargs" not in sig.parameters
            and "settings" not in sig.parameters
        ):
            for key in kwargs.keys():
                if key not in sig.parameters.keys():
                    raise TypeError(
                        f"{func.__name__}() got invalid argument `{key}`\n"
                        f"valid arguments are {list(sig.parameters.keys())}"
                    )

        if cache and name in CACHE:
            # print(f"CACHE {func.__name__}({kwargs_repr})")
            return CACHE[name]
        else:
            # print(f"BUILD {name} {func.__name__}({arguments})")
            # print(f"BUILD {name}, {name_long}")
            assert callable(
                func
            ), f"{func} got decorated with @cell! @cell decorator is only for functions"
            component = func(*args, **kwargs)
            if decorator:
                assert callable(
                    decorator
                ), f"decorator = {type(decorator)} needs to be callable"
                decorator(component)

            if not isinstance(component, Component):
                raise CellReturnTypeError(
                    f"function `{func.__name__}` return type = `{type(component)}`",
                    "make sure that functions with @cell decorator return a Component",
                )
            if autoname and getattr(component, "_autoname", True):
                component.name = name_component

            # docstring = func.__doc__ if hasattr(func, "__doc__") else func.func.__doc__
            # component.info.doc = docstring

            component.info.module = func.__module__
            component.info.function_name = func.__name__
            component.info.info_version = INFO_VERSION
            component.info.name_long = name_long
            component.info.name = component.name
            component.info.update(**info)

            default = {
                p.name: p.default
                for p in sig.parameters.values()
                if not callable(p.default)
            }
            full = default.copy()
            full.update(**args_as_kwargs)
            changed = args_as_kwargs.copy()

            clean_dict(full)
            clean_dict(default)
            clean_dict(changed)

            component.info.changed = changed
            component.info.default = default
            component.info.full = full

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
    c.add_port(name="o1", midpoint=[0, 0], width=width, orientation=180, layer=layer)
    c.add_port(name="o2", midpoint=[length, 0], width=width, orientation=0, layer=layer)
    return c


def test_autoname_true() -> None:
    c = wg(length=3)
    # assert c.name == "wg_length3", c.name
    assert c.name == "wg_2dcab9f2", c.name


def test_autoname_false() -> None:
    c = wg(length=3.32, autoname=False)
    assert c.name == "straight", c.name


def test_set_name() -> None:
    c = wg(length=3, name="hi_there")
    assert c.name == "hi_there", c.name


@cell
def _dummy(length: int = 3, wg_width: float = 0.5) -> Component:
    """Dummy cell"""
    c = Component()
    w = length
    h = wg_width
    points = [
        [-w / 2.0, -h / 2.0],
        [-w / 2.0, h / 2],
        [w / 2, h / 2],
        [w / 2, -h / 2.0],
    ]
    c.add_polygon(points)
    return c


def test_autoname() -> None:
    name_base = _dummy().name
    assert name_base == "_dummy", name_base

    name_int = _dummy(length=3).name
    assert name_int == "_dummy_2dcab9f2", name_int

    dummy2 = functools.partial(_dummy, length=3)
    component_int = dummy2(length=3)
    name_int = component_int.name
    assert name_int == "_dummy_2dcab9f2", name_int
    # assert component_int.info.doc == "Dummy cell"

    name_float = _dummy(wg_width=0.5).name
    assert name_float == "_dummy_b78ec006", name_float

    name_length_first = _dummy(length=3, wg_width=0.5).name
    name_width_first = _dummy(wg_width=0.5, length=3).name
    assert (
        name_length_first == name_width_first
    ), f"{name_length_first} != {name_width_first}"

    name_args = _dummy(3).name
    assert name_int == "_dummy_2dcab9f2", name_int

    name_with_prefix = _dummy(prefix="hi").name
    assert name_with_prefix == "hi", name_with_prefix

    name_args = _dummy(3).name
    name_kwargs = _dummy(length=3).name
    assert name_args == name_kwargs, name_with_prefix


if __name__ == "__main__":
    # dummy2 = functools.partial(_dummy, length=3)
    # c = dummy2()

    # c = _dummy()
    # test_raise_error_args()
    # c = gf.components.straight()

    # test_autoname_false()
    # test_autoname_true()
    # test_autoname()
    # test_set_name()

    # c = wg(length=3)
    # c = wg(length=3, autoname=False)

    import gdsfactory as gf

    info = dict(polarization="te")

    c = gf.components.straight()
    c = gf.components.straight(info=info)
    # c = gf.components.straight(length=3, info=info)
    print(c.info.polarization)

    # print(c.settings.info.doc)
    # c = gf.components.spiral_inner_io(length=1e3)
    # c = gf.components.straight(length=3)
    # print(c.name)
    # c.show()

    # D = gf.Component()
    # arc = D << gf.components.bend_circular(
    #     radius=10, width=0.5, angle=90, layer=(1, 0), info=dict(polarization="te")
    # )
    # arc.rotate(90)
    # rect = D << gf.components.bbox(bbox=arc.bbox, layer=(0, 0))

"""
INFO_VERSION

1: original metadata format

"""
import functools
import hashlib
import inspect
from typing import Dict

import omegaconf
from pydantic import validate_arguments

from gdsfactory.component import Component, clean_dict
from gdsfactory.name import MAX_NAME_LENGTH, clean_name, clean_value, get_name_short

CACHE: Dict[str, Component] = {}
INFO_VERSION = 1


class CellReturnTypeError(ValueError):
    pass


def clear_cache() -> None:
    """Clears the component CACHE."""
    global CACHE
    CACHE = {}


def print_cache():
    for k in CACHE:
        print(k)


def cell_without_validator(func):
    @functools.wraps(func)
    def _cell(*args, **kwargs):
        """Decorator for Component functions.

        similar to cell decorator, this one does not validate_arguments using
        type annotations

        kwargs:
            autoname: if True renames component based on args and kwargs
            name (str): Optional (ignored when autoname=True)
            cache (bool): get component from the cache if it already exists.
              Useful in jupyter notebook, so you don't have to clear the cache
            info: updates component.info dict
            prefix: name_prefix, defaults to function name
            max_name_length: truncates name beyond some characters (32) with a hash
            decorator: function to run over the component

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
        autoname = kwargs.pop("autoname", True)
        name = kwargs.pop("name", None)
        cache = kwargs.pop("cache", True)
        info = kwargs.pop("info", omegaconf.DictConfig({}))
        prefix = kwargs.pop("prefix", func.__name__)
        max_name_length = kwargs.pop("max_name_length", MAX_NAME_LENGTH)

        sig = inspect.signature(func)
        args_as_kwargs = dict(zip(sig.parameters.keys(), args))
        args_as_kwargs.update(**kwargs)
        args_as_kwargs_string_list = [
            f"{key}={clean_value(args_as_kwargs[key])}"
            for key in sorted(args_as_kwargs.keys())
        ]

        # for key in sorted(args_as_kwargs.keys()):
        #     print(f"{key}={clean_value(args_as_kwargs[key])}")

        arguments = "_".join(args_as_kwargs_string_list)
        arguments_hash = hashlib.md5(arguments.encode()).hexdigest()[:8]

        name_signature = (
            clean_name(f"{prefix}_{arguments_hash}") if arguments else prefix
        )
        name = name or name_signature
        decorator = kwargs.pop("decorator", None)
        name = get_name_short(name, max_name_length=max_name_length)

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
            # print(f"CACHE LOAD {name} {func.__name__}({arguments})")
            return CACHE[name]
        else:
            # print(f"BUILD {name} {func.__name__}({arguments})")
            assert callable(
                func
            ), f"{func} got decorated with @cell! @cell decorator is only for functions"

            component = func(*args, **kwargs)

            if not isinstance(component, Component):
                raise CellReturnTypeError(
                    f"function `{func.__name__}` return type = `{type(component)}`",
                    "make sure that functions with @cell decorator return a Component",
                )

            if (
                component.get_child_name
                and hasattr(component, "info_child")
                and hasattr(component.info_child, "name")
            ):
                component_name = f"{component.info_child.name}_{name}"
                component_name = get_name_short(
                    component_name, max_name_length=max_name_length
                )
            else:
                component_name = name

            if autoname:
                component.name = component_name
            component.info.name = component_name
            component.info.module = func.__module__
            component.info.function_name = func.__name__
            component.info.info_version = INFO_VERSION

            default = {
                p.name: p.default
                for p in sig.parameters.values()
                if not p.default == inspect._empty
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

            component.info.update(**info)

            if decorator:
                assert callable(
                    decorator
                ), f"decorator = {type(decorator)} needs to be callable"
                component_new = decorator(component)
                if component_new and autoname:
                    component_new.name = get_name_short(
                        f"{component.name}_{clean_value(decorator)}",
                        max_name_length=max_name_length,
                    )
                component = component_new or component

            component.cached = True
            CACHE[name] = component
            return component

    return _cell


def cell(func, *args, **kwargs):
    """Validates type annotations with pydantic"""
    return cell_without_validator(validate_arguments(func), *args, **kwargs)


@cell
def wg(length: int = 3, width: float = 0.5) -> Component:
    """Dummy component for testing."""
    from gdsfactory.component import Component

    c = Component("straight")
    w = width / 2
    layer = (1, 0)
    c.add_polygon([(0, -w), (length, -w), (length, w), (0, w)], layer=layer)
    c.add_port(name="o1", midpoint=[0, 0], width=width, orientation=180, layer=layer)
    c.add_port(name="o2", midpoint=[length, 0], width=width, orientation=0, layer=layer)
    return c


def test_autoname() -> None:
    c = wg(length=3)
    # assert c.name == "wg_length3", c.name
    assert c.name == "wg_2dcab9f2", c.name


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


def test_names() -> None:
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

    name_with_prefix = _dummy(prefix="hi").name
    assert name_with_prefix == "hi", name_with_prefix

    name_args = _dummy(3).name
    name_kwargs = _dummy(length=3).name
    assert name_args == name_kwargs, name_with_prefix


@cell
def straight_with_pins(**kwargs):
    import gdsfactory as gf

    c = gf.Component()
    ref = c << gf.c.straight()
    c.add_ports(ref.ports)
    gf.add_pins(c)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    c = straight_with_pins(decorator=gf.routing.add_fiber_single)
    # c = straight_with_pins()
    c.show()

    # c = wg(name="my_waveguide")
    # print(c.name)

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

    # import gdsfactory as gf
    # info = dict(polarization="te")

    # c = gf.components.straight()
    # c = gf.components.straight(info=info)
    # c = gf.components.straight(length=3, info=info)
    # print(c.info.polarization)

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

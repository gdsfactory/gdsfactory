from typing import Callable
from inspect import signature
import uuid
import hashlib
from functools import wraps, partial
from pp.add_pins import add_pins_and_outline
from pp.name import get_component_name


NAME_TO_DEVICE = {}
MAX_NAME_LENGTH = 32


def clear_cache(components_cache=NAME_TO_DEVICE):
    components_cache = {}
    return components_cache


def cell(
    func=None,
    *,
    autoname=True,
    name=None,
    uid=False,
    cache=True,
    pins=False,
    pins_function=add_pins_and_outline,
) -> Callable:
    """Cell Decorator:

    Args:
        autoname (bool): renames Component by concenating all Keyword arguments if no Keyword argument `name`
        name (str): Optional (ignored when autoname=True)
        uid (bool): adds a unique id to the name
        cache (bool): To avoid that 2 exact cells are not references of the same cell cell has a cache where if component has already been build it will return the component from the cache. You can always over-ride this with `cache = False`.
        pins (bool): add pins
        pins_function (function): function to add pins

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
      c << pp.c.text(text=c.name, size=1)
      pp.plotgds(c)

    """

    if func is None:
        return partial(
            cell,
            autoname=autoname,
            name=name,
            uid=uid,
            cache=cache,
            pins=pins,
            pins_function=pins_function,
        )

    @wraps(func)
    def _cell(
        autoname=autoname,
        name=name,
        pins=pins,
        uid=uid,
        cache=cache,
        pins_function=pins_function,
        *args,
        **kwargs,
    ):
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
        # assert_first_letters_are_different(**sig.parameters)

        if "args" not in sig.parameters and "kwargs" not in sig.parameters:
            for key in kwargs.keys():
                if key not in sig.parameters.keys():
                    raise TypeError(
                        f"{component_type}() got an unexpected keyword argument `{key}`\n"
                        f"valid keyword arguments are {list(sig.parameters.keys())}"
                    )

        if cache and autoname and name in NAME_TO_DEVICE:
            return NAME_TO_DEVICE[name]
        else:
            component = func(**kwargs)
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
            if pins:
                pins_function(component)

            NAME_TO_DEVICE[name] = component
            return component

    return _cell


@cell(autoname=True)
def wg(length=3, width=0.5):
    from pp.component import Component

    c = Component("waveguide")
    w = width / 2
    layer = (1, 0)
    c.add_polygon([(0, -w), (length, -w), (length, w), (0, w)], layer=layer)
    c.add_port(name="W0", midpoint=[0, 0], width=width, orientation=180, layer=layer)
    c.add_port(name="E0", midpoint=[length, 0], width=width, orientation=0, layer=layer)
    return c


@cell(autoname=False)
def wg2(length=3, width=0.5):
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


def test_autoname_true():
    assert wg(length=3).name == "wg_L3"


def test_autoname_false():
    # print(wg2(length=3).name)
    assert wg2(length=3).name == "waveguide"


class _Dummy:
    pass


@cell
def _dummy(length=3, wg_width=0.5):
    c = _Dummy()
    c.name = ""
    c.settings = {}
    return c


def test_cell():
    name_base = _dummy().name
    assert name_base == "_dummy"
    name_int = _dummy(length=3).name
    assert name_int == "_dummy_L3"
    name_float = _dummy(wg_width=0.5).name
    # assert name_float == "_dummy_WW500m"
    name_length_first = _dummy(length=3, wg_width=0.5).name
    name_width_first = _dummy(wg_width=0.5, length=3).name
    assert name_length_first == name_width_first

    name_float = _dummy(wg_width=0.5).name
    # assert name_float == "_dummy_WW0p5"
    print(name_float)


if __name__ == "__main__":
    import pp

    # test_autoname_true()
    # test_autoname_false()

    # c = wg(length=3)
    # c = wg(length=3, pins=False, autoname=False)

    # c = pp.c.waveguide()
    c = wg3(pins=True)
    print(c)
    pp.show(c)

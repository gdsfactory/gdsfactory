""" define names, clean names and values
autoname adds geometric hash

```
import importlib
function_string = c.get_settings()['module']
mod_name, func_name = function_string.rsplit('.',1)
mod = importlib.import_module(mod_name)
func = getattr(mod, func_name)
result = func()
```
"""
import functools
from inspect import signature
import hashlib
import numpy as np
from phidl import Device
from pp.add_pins import add_pins

MAX_NAME_LENGTH = 32


def join_first_letters(name):
    """ join the first letter of a name separated with underscores (taper_length -> TL) """
    return "".join([x[0] for x in name.split("_") if x])


def get_component_name(component_type, max_name_length=MAX_NAME_LENGTH, **kwargs):
    name = component_type

    if kwargs:
        name += "_" + dict2name(**kwargs)

    # If the name is too long, fall back on hashing the longuest arguments
    if len(name) > max_name_length:
        shorter_name = f"{component_type}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        # print(f"{name} -> {shorter_name} ({len(name)} -> {max_name_length})")
        name = shorter_name

    return name


def autoname(component_function):
    """ decorator for auto-naming component functions
    if no Keyword argument `name`  is passed it creates a name by concenating all Keyword arguments

    .. plot::
      :include-source:

      import pp

      @pp.autoname
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

    @functools.wraps(component_function)
    def wrapper(*args, **kwargs):
        if args:
            raise ValueError("autoname supports only Keyword args")
        with_pins = kwargs.pop("with_pins", False)
        add_pins_function = kwargs.pop("add_pins_function", add_pins)
        max_name_length = kwargs.pop("max_name_length", MAX_NAME_LENGTH)
        name = kwargs.pop(
            "name",
            get_component_name(
                component_function.__name__, max_name_length=max_name_length, **kwargs
            ),
        )
        kwargs.pop("ignore_from_name", [])

        component = component_function(**kwargs)
        component.name = name
        component.module = component_function.__module__
        component.function_name = component_function.__name__
        sig = signature(component_function)
        component.settings.update(
            **{p.name: p.default for p in sig.parameters.values()}
        )
        component.settings.update(**kwargs)
        # if hasattr(component, 'hash_geometry'):
        #     component.settings.update(hash=component.hash_geometry())
        if with_pins:
            add_pins_function(component)

        return component

    return wrapper


def dict2name(prefix=None, **kwargs):
    """ returns name from a dict """
    ignore_from_name = kwargs.pop("ignore_from_name", [])

    if prefix:
        label = [prefix]
    else:
        label = []
    for key in sorted(kwargs):
        if key not in ignore_from_name:
            value = kwargs[key]
            key = join_first_letters(key)
            value = clean_value(value)
            label += [f"{key.upper()}{value}"]
    label = "_".join(label)
    return clean_name(label)


def clean_name(name):
    """ Ensures that gds cells are composed of [a-zA-Z0-9]

    FIXME: only a few characters are currently replaced.
        This function has been updated only on case-by-case basis
    """
    replace_map = {
        " ": "_",
        "!": "_",
        "#": "_",
        "%": "_",
        "(": "",
        ")": "",
        "*": "_",
        ",": "_",
        "-": "m",
        ".": "p",
        "/": "_",
        ":": "_",
        "=": "",
        "@": "_",
        "[": "",
        "]": "",
    }
    for k, v in list(replace_map.items()):
        name = name.replace(k, v)
    return name


def clean_value(value):
    """ returns more readable value (integer)
    if number is < 1:
        returns number units in nm (integer)
    """

    if isinstance(value, int):  # integer
        value = str(value)
    elif isinstance(value, (float, np.float64)):  # float
        if 1e9 > value > 1e12:
            value = f"{int(value/1e9)}G"
        elif 1e6 > value > 1e9:
            value = f"{int(value/1e6)}M"
        elif 1e3 > value > 1e6:
            value = f"{int(value/1e3)}K"
        elif 1 > value > 1e-3:
            value = f"{int(value*1e3)}m"
        elif 1e-6 < value < 1e-3:
            value = f"{int(value*1e6)}u"
        elif 1e-9 < value < 1e-6:
            value = f"{int(value*1e9)}n"
        elif 1e-12 < value < 1e-9:
            value = f"{int(value*1e12)}p"
        else:
            value = f"{value:.2f}"
    elif isinstance(value, list):
        value = "_".join(clean_value(v) for v in value)
    elif isinstance(value, tuple):
        value = "_".join(clean_value(v) for v in value)
    elif isinstance(value, dict):
        value = dict2name(**value)
    elif isinstance(value, Device):
        value = clean_name(value.name)
    elif callable(value):
        value = value.__name__
    else:
        value = clean_name(str(value))
    return value


class _Dummy:
    pass


@autoname
def _dummy(length=3, wg_width=0.5):
    c = _Dummy()
    c.name = ""
    c.settings = {}
    return c


def test_autoname():
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


def test_clean_value():
    assert clean_value(0.5) == "500m"
    assert clean_value(5) == "5"


def test_clean_name():
    assert clean_name("wg(:_=_2852") == "wg___2852"


if __name__ == "__main__":
    test_autoname()
    # test_autoname()
    # import pp

    # print(clean_value(pp.c.waveguide))

    # c = pp.c.waveguide(polarization="TMeraer")
    # print(c.get_settings()["polarization"])

    # c = pp.c.waveguide(length=11)
    # print(c)
    # pp.show(c)

    # print(clean_name("Waveguidenol1_(:_=_2852"))
    # print(clean_value(1.2))
    # print(clean_value(0.2))
    # print(clean_value([1, [2.4324324, 3]]))
    # print(clean_value([1, 2.4324324, 3]))
    # print(clean_value((0.001, 24)))
    # print(clean_value({"a": 1, "b": 2}))

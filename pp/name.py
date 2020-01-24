""" define names, clean names and values """
import functools
from inspect import signature
import hashlib
import numpy as np
from phidl import Device

MAX_NAME_LENGTH = 127


def join_first_letters(name):
    """ join the first letter of a name separated with underscores (taper_length -> TL) """
    return "".join([x[0] for x in name.split("_") if x])


def get_component_name(component_type, **kwargs):
    name = component_type

    if kwargs:
        name += "_" + dict2name(**kwargs)

    # If the name is too long, fall back on hashing the longuest arguments
    if len(name) > MAX_NAME_LENGTH:
        name = "{}_{}".format(component_type, hashlib.md5(name.encode()).hexdigest())

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
        if "name" in kwargs:
            name = kwargs.pop("name")
        else:
            name = get_component_name(component_function.__name__, **kwargs)

        component = component_function(**kwargs)
        component.name = name
        component.name_function = component_function.__name__
        sig = signature(component_function)
        component.settings.update(
            **{p.name: p.default for p in sig.parameters.values()}
        )
        component.settings.update(**kwargs)
        component.function_name = component_function.__name__
        return component

    return wrapper


def autoname2(component_function):
    """ pass name to component

    """

    @functools.wraps(component_function)
    def wrapper(*args, **kwargs):
        if args:
            raise ValueError("autoname supports only Keyword args")
        if "name" in kwargs:
            name = kwargs.pop("name")
        else:
            name = get_component_name(component_function.__name__, **kwargs)

        component = component_function(name=name, **kwargs)
        sig = signature(component_function)
        component.settings.update(
            **{p.name: p.default for p in sig.parameters.values()}
        )

        component.name = name
        component.settings.pop("name")
        component.settings.update(**kwargs)
        component.function_name = component_function.__name__
        return component

    return wrapper


def dict2name(prefix=None, **kwargs):
    """ returns name from a dict """
    if prefix:
        label = [prefix]
    else:
        label = []
    for key, value in kwargs.items():
        key = join_first_letters(key)
        label += ["{}{}".format(key.upper(), clean_value(value))]
    label = "_".join(label)
    return clean_name(label)


def clean_name(name):
    """ Ensures that gds cells are composed of [a-zA-Z0-9_\-]

    FIXME: only a few characters are currently replaced.
        This function has been updated only on case-by-case basis
    """
    replace_map = {
        "=": "",
        ",": "_",
        ")": "",
        "(": "",
        "-": "m",
        ".": "p",
        ":": "_",
        "[": "",
        "]": "",
        " ": "_",
    }
    for k, v in list(replace_map.items()):
        name = name.replace(k, v)
    return name


def clean_value(value):
    """ returns more readable value (integer)
    if number is < 1:
        returns number units in nm (integer)
    """

    def f():
        return

    try:
        if isinstance(value, int):  # integer
            return str(value)
        elif type(value) in [float, np.float64]:  # float
            return "{:.4f}".format(value).replace(".", "p").rstrip("0").rstrip("p")
        elif isinstance(value, list):
            return "_".join(clean_value(v) for v in value)
        elif isinstance(value, tuple):
            return "_".join(clean_value(v) for v in value)
        elif isinstance(value, dict):
            return dict2name(**value)
        elif isinstance(value, Device):
            return clean_name(value.name)
        elif callable(value):
            return value.__name__
        else:
            return clean_name(str(value))
    except TypeError:  # use the __str__ method
        return clean_name(str(value))


if __name__ == "__main__":
    import pp

    # print(clean_value(pp.c.waveguide))

    # c = pp.c.waveguide(polarization="TMeraer")
    # print(c.get_settings()["polarization"])

    c = pp.c.waveguide(length=11)
    print(c)
    pp.show(c)

    # print(clean_name("Waveguidenol1_(:_=_2852"))
    # print(clean_value(1.2))
    # print(clean_value(0.2))
    # print(clean_value([1, [2.4324324, 3]]))
    # print(clean_value([1, 2.4324324, 3]))
    # print(clean_value((0.001, 24)))
    # print(clean_value({"a": 1, "b": 2}))

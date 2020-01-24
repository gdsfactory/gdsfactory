import functools
import collections
from inspect import signature
from pp.component import NAME_TO_DEVICE
from pp.name import get_component_name


def update_dicts_recurse(target_dict, default_dict):
    target_dict = target_dict.copy()
    default_dict = default_dict.copy()
    for k, v in default_dict.items():
        if k not in target_dict:
            target_dict[k] = v
        else:
            vtype = type(target_dict[k])
            if vtype == dict or vtype == collections.OrderedDict:
                target_dict[k] = update_dicts_recurse(target_dict[k], default_dict[k])
    return target_dict


def autoname_and_cache(component_function):
    """ decorator for auto-naming component functions

    you can pass a name argument
    otherwise it creates a name by concenating all Keyword arguments

    requires kwargs

    .. code-block:: python
        import pp

        @pp.autoname
        def rectangle(widht=2, height=3):
            c = Component()
            return c
    """

    @functools.wraps(component_function)
    def wrapper(*args, **kwargs):
        if args:
            raise ValueError("autoname supports only Keyword args")
        if "name" in kwargs:
            name = kwargs.pop("name")
        else:
            name = get_component_name(component_function.__name__, **kwargs)

        if name in NAME_TO_DEVICE:
            component = NAME_TO_DEVICE[name]
        else:
            component = component_function(**kwargs)
            component.name = name
            component.name_function = component_function.__name__
            sig = signature(component_function)

            update_dicts_recurse(component.settings, kwargs)
            update_dicts_recurse(
                component.settings, {p.name: p.default for p in sig.parameters.values()}
            )

            # component.settings.update(
            # **{p.name: p.default for p in sig.parameters.values()}
            # )
            # component.settings.update(**kwargs)
            component.function_name = component_function.__name__

            NAME_TO_DEVICE[name] = component

        return component

    return wrapper

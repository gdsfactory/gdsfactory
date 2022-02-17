"""Functional programming tools.
"""
from functools import partial as _partial


def partial(func, *args, **kwargs):
    """Propagate func docstring."""
    new_func = _partial(func, *args, **kwargs)
    new_func.__doc__ = func.__doc__
    return new_func


if __name__ == "__main__":
    import gdsfactory as gf

    # from gdsfactory.serialization import clean_value_json
    # from gdsfactory.serialization import clean_value_name

    c1 = _partial(gf.c.straight, length=1)
    c2 = _partial(c1, length=2)
    c3 = _partial(c2, length=3)

    # print(clean_value_json(c1))
    # print(clean_value_json(c2))
    # print(clean_value_json(c3))

    # print(clean_value_name(c1))
    # print(clean_value_name(c2))
    # print(clean_value_name(c3))

    c1a = _partial(gf.c.straight, length=1)
    c1b = partial(gf.c.straight, length=1)

    c4 = gf.c.mzi(straight=c1a)
    c5 = gf.c.mzi(straight=c1b)

    print(c4.name)
    print(c5.name)

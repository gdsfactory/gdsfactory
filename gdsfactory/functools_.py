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
    from gdsfactory.serialization import clean_value_json

    c1 = gf.partial(gf.c.straight, length=1)
    c2 = gf.partial(c1, length=2)
    c3 = gf.partial(c2, length=3)

    print(c3.func.func.func.__name__)
    print(clean_value_json(c1))
    print(clean_value_json(c2))
    print(clean_value_json(c3))

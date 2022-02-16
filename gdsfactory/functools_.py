"""Functional programming tools.
"""
from functools import partial as _partial


def partial(func, *args, **kwargs):
    """Propagate func docstring."""
    f = _partial(func, *args, **kwargs)
    f.__doc__ = func.__doc__
    f.func = func
    return f

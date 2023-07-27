from __future__ import annotations

import gdsfactory as gf
from gdsfactory.snap import is_on_grid
from gdsfactory.typings import Component, ComponentReference, Optional
import inspect
import itertools as it
from collections import deque


def is_valid_transformation(
    ref: ComponentReference, grid_size: Optional[float] = None
) -> bool:
    """Returns True if the component has valid transformations.

    Args:
        component: the component reference to check.
        grid_size: the GDS grid size, in um, defaults to active PDK.get_grid_size()
            any translations with higher resolution than this are considered invalid.
    """
    from gdsfactory.pdk import get_grid_size

    grid_size = grid_size or get_grid_size()
    nm = int(grid_size * 1e3)
    origin_is_on_grid = all(is_on_grid(x, nm) for x in ref.origin)
    rotation_is_regular = ref.rotation is None or ref.rotation % 90 == 0
    return origin_is_on_grid and rotation_is_regular


def has_valid_transformations(component: Component) -> bool:
    """Returns True if the component has valid transformations."""
    refs = component.references
    return all(is_valid_transformation(ref) for ref in refs)


def flatten_invalid_refs(component: Component, grid_size: Optional[float] = None):
    """Flattens component references which have invalid transformations.

    (i.e. non-90 deg rotations or sub-grid translations).

    This is an in-place operation, so you should use it as a decorator.
    flattens only individual references with invalid transformations.

    Deprecated Use Component.write_gds(flatten_invalid_refs=True)

    Args:
        component: the component to fix (in place).
        grid_size: the GDS grid size, in um, defaults to active PDK.get_grid_size()
            any translations with higher resolution than this are considered invalid.
    """
    refs = component.references.copy()
    for ref in refs:
        if is_invalid_ref(ref, grid_size):
            component.flatten_reference(ref)
    return component


def is_invalid_ref(ref, grid_size: Optional[float] = None) -> bool:
    from gdsfactory.pdk import get_grid_size

    grid_size = grid_size or get_grid_size()
    nm = int(grid_size * 1e3)
    origin_is_on_grid = all(is_on_grid(x, nm) for x in ref.origin)
    rotation_is_regular = ref.rotation is None or ref.rotation % 90 == 0
    return not origin_is_on_grid or not rotation_is_regular


def defaultsfrom(funcOrClass):
    """Return a decorator d so that d(func) updates func's default arguments.

    From https://code.activestate.com/recipes/440702-reusing-default-function-arguments/
    """

    def decorator(newfunc):
        if inspect.isclass(funcOrClass):
            func = getattr(funcOrClass, newfunc.__name__)
        else:
            func = funcOrClass
        args, _, _, defaults = inspect.getargspec(func)
        # map each default argument of func to its value
        arg2default = dict(zip(args[-len(defaults) :], defaults))
        newargs, _, _, newdefaults = inspect.getargspec(newfunc)
        if newdefaults is None:
            newdefaults = ()
        nondefaults = newargs[: len(newargs) - len(newdefaults)]
        # starting from the last non-default argument towards the first, as
        # long as the non-defaults of newfunc are default in func, make them
        # default in newfunc too
        iter_nondefaults = reversed(nondefaults)
        newdefaults = deque(newdefaults)
        for arg in it.takewhile(arg2default.__contains__, iter_nondefaults):
            newdefaults.appendleft(arg2default[arg])
        # all inherited defaults should be placed together; no gaps allowed
        for _ in it.ifilter(arg2default.__contains__, iter_nondefaults):
            raise TypeError(
                "%s cannot inherit the default arguments of " "%s" % (newfunc, func)
            )
        newfunc.func_defaults = tuple(newdefaults)
        return newfunc

    return decorator


@gf.cell
def _demo_non_manhattan() -> Component:
    """Returns component with Manhattan snapping issues."""
    c = Component()
    b = c << gf.components.bend_circular(angle=30)
    s = c << gf.components.straight(length=5)
    s.connect("o1", b.ports["o2"])
    return c


def test_flatten_invalid_refs() -> None:
    c1 = _demo_non_manhattan()
    assert not has_valid_transformations(c1)

    c2 = _demo_non_manhattan(decorator=flatten_invalid_refs)
    assert has_valid_transformations(c2)


if __name__ == "__main__":
    test_flatten_invalid_refs()
    # c = _demo_non_manhattan(decorator=flatten_invalid_refs)
    # c.show()

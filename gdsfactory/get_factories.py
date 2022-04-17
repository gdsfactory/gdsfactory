import inspect
from collections.abc import Iterable
from inspect import getmembers

from gdsfactory.types import Component, ComponentFactory, Dict


def get_cells(modules) -> Dict[str, ComponentFactory]:
    """Returns Pcells (component functions) from a module."""

    modules = modules if isinstance(modules, Iterable) else [modules]

    return {
        t[0]: t[1]
        for module in modules
        for t in getmembers(module)
        if callable(t[1]) and inspect.signature(t[1]).return_annotation == Component
        # if isfunction(t[1]) and id(t[1]) in _FACTORY
    }


def validate_module_factories(modules) -> None:
    """Iterates over module functions and makes sure they have a valid signature."""
    modules = modules if isinstance(modules, Iterable) else [modules]

    for module in modules:
        for t in getmembers(module):
            try:
                if callable(t[1]):
                    inspect.signature(t[1]).return_annotation
            except Exception:
                print(f"error in {t[0]}")


if __name__ == "__main__":
    import ubcpdk

    f = get_cells(ubcpdk.components)
    print(f.keys())

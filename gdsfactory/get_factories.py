import inspect
from inspect import getmembers

from gdsfactory.types import Component, ComponentFactory, Dict


def get_cells(module) -> Dict[str, ComponentFactory]:
    """Returns Pcells (component functions) from a module."""

    return {
        t[0]: t[1]
        for t in getmembers(module)
        if callable(t[1]) and inspect.signature(t[1]).return_annotation == Component
        # if isfunction(t[1]) and id(t[1]) in _FACTORY
    }


def validate_module_factories(module) -> None:
    """Iterates over module functions and makes sure they have a valid signature."""

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

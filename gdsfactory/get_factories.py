from __future__ import annotations

from collections.abc import Iterable
from inspect import getmembers, signature

from gdsfactory.typings import Component, ComponentFactory, Dict


def get_cells(modules, verbose: bool = False) -> Dict[str, ComponentFactory]:
    """Returns PCells (component functions) from a module or list of modules.

    Args:
        modules: module or iterable of modules.
        verbose: prints in case any errors occur.

    """
    modules = modules if isinstance(modules, Iterable) else [modules]

    cells = {}
    for module in modules:
        for t in getmembers(module):
            if callable(t[1]) and t[0] != "partial":
                try:
                    r = signature(t[1]).return_annotation
                    if r == Component or (
                        isinstance(r, str) and r.endswith("Component")
                    ):
                        cells[t[0]] = t[1]
                except ValueError:
                    if verbose:
                        print(f"error in {t[0]}")
    return cells


if __name__ == "__main__":
    import ubcpdk

    f = get_cells(ubcpdk.components)
    print(f.keys())

from __future__ import annotations

from functools import partial

import gdsfactory.components as components


def test_exported_component_partials_preserve_docstrings() -> None:
    partials = {
        name: component
        for name in components.__all__
        if isinstance(component := getattr(components, name, None), partial)
    }

    assert partials
    for name, component in partials.items():
        wrapped = component.func
        while isinstance(wrapped, partial):
            wrapped = wrapped.func
        assert component.__doc__ == wrapped.__doc__, name

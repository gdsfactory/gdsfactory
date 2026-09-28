"""Sub-component lookup for component functions."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from gdsfactory.component import Component
from gdsfactory.pdk import get_active_pdk
from gdsfactory.typings import ComponentSpec


class ComponentFallbackWarning(UserWarning):
    """A component function fell back to a gdsfactory.components cell.

    Raised when the active PDK does not register a cell that a component
    function asked for by name.
    """


def _spec_name(component: ComponentSpec) -> str | None:
    if isinstance(component, str):
        return component
    if isinstance(component, dict):
        name = component.get("component") or component.get("function")
        return str(name).split(".")[-1] if name else None
    return None


def get_component(
    component: ComponentSpec,
    settings: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Component:
    """Returns a sub-component, looked up in the active PDK.

    Component functions resolve every sub-cell through this function so that
    PDK overrides apply. If the active PDK does not register a named cell,
    this falls back to the ``gdsfactory.components`` cell of the same name and
    issues a ``ComponentFallbackWarning``.

    Args:
        component: Component, ComponentFactory, string or dict.
        settings: settings to override.
        kwargs: settings to override.
    """
    import gdsfactory.components

    pdk = get_active_pdk()
    name = _spec_name(component)

    if name is None or name in pdk.cells or name in pdk.containers:
        return pdk.get_component(component, settings=settings, **kwargs)

    factory = getattr(gdsfactory.components, name, None)
    if not callable(factory):
        return pdk.get_component(component, settings=settings, **kwargs)

    warnings.warn(
        f"{name!r} is not in PDK {pdk.name!r}. Falling back to "
        f"gdsfactory.components.{name}. Register {name!r} in the PDK cells "
        "to silence this warning.",
        ComponentFallbackWarning,
        stacklevel=2,
    )
    merged: dict[str, Any] = {}
    if isinstance(component, dict):
        merged.update(component.get("settings", {}))
    merged.update(settings or {})
    return pdk.get_component(factory, settings=merged, **kwargs)

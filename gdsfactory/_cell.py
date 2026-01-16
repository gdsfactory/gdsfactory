from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, cast, overload

import kfactory as kf
from cachetools import Cache
from kfactory import cell as _cell
from kfactory import vcell as _vcell
from kfactory.conf import CheckInstances
from kfactory.decorators import PortsDefinition
from kfactory.schematic import DSchematic
from kfactory.serialization import clean_name
from kfactory.typings import MetaData

if TYPE_CHECKING:
    from gdsfactory.component import Component, ComponentAllAngle
    from gdsfactory.typings import ComponentFactory, RoutingStrategies

ComponentParams = ParamSpec("ComponentParams")


class ComponentFunc(Protocol[ComponentParams]):
    __name__: str

    def __call__(
        self, *args: ComponentParams.args, **kwargs: ComponentParams.kwargs
    ) -> Component: ...


@overload
def cell(
    _func: ComponentFunc[ComponentParams], /
) -> ComponentFunc[ComponentParams]: ...


@overload
def cell(
    *,
    set_settings: bool = True,
    set_name: bool = True,
    check_ports: bool = True,
    check_instances: CheckInstances | None = None,
    snap_ports: bool = True,
    add_port_layers: bool = True,
    cache: Cache[int, Any] | dict[int, Any] | None = None,
    basename: str | None = None,
    drop_params: list[str] | None = None,
    register_factory: bool = True,
    overwrite_existing: bool | None = None,
    layout_cache: bool | None = None,
    info: dict[str, MetaData] | None = None,
    post_process: Iterable[Callable[[Component], None]] | None = None,
    debug_names: bool | None = None,
    tags: list[str] | None = None,
    with_module_name: bool = False,
    lvs_equivalent_ports: list[list[str]] | None = None,
) -> Callable[[ComponentFunc[ComponentParams]], ComponentFunc[ComponentParams]]: ...


def cell(
    _func: ComponentFunc[ComponentParams] | None = None,
    /,
    *,
    set_settings: bool = True,
    set_name: bool = True,
    check_ports: bool = True,
    check_instances: CheckInstances | None = None,
    snap_ports: bool = True,
    add_port_layers: bool = True,
    cache: Cache[int, Any] | dict[int, Any] | None = None,
    basename: str | None = None,
    drop_params: list[str] | None = None,
    register_factory: bool = True,
    overwrite_existing: bool | None = None,
    layout_cache: bool | None = None,
    info: dict[str, MetaData] | None = None,
    post_process: Iterable[Callable[[Component], None]] | None = None,
    debug_names: bool | None = None,
    tags: list[str] | None = None,
    with_module_name: bool = False,
    lvs_equivalent_ports: list[list[str]] | None = None,
    ports: PortsDefinition | None = None,
) -> (
    ComponentFunc[ComponentParams]
    | Callable[[ComponentFunc[ComponentParams]], ComponentFunc[ComponentParams]]
):
    """Decorator to convert a function into a Component."""
    from gdsfactory.component import Component

    if with_module_name and _func is not None:
        mod = _func.__module__
        basename = basename or clean_name(
            _func.__name__ if mod == "__main__" else f"{_func.__name__}_{mod}"
        )

    if drop_params is None:
        drop_params = ["self", "cls"]
    if post_process is None:
        post_process = []
    c = _cell(  # type: ignore[call-overload,misc]
        _func,
        output_type=Component,
        set_settings=set_settings,
        set_name=set_name,
        check_ports=check_ports,
        check_instances=check_instances,
        snap_ports=snap_ports,
        add_port_layers=add_port_layers,
        cache=cache,
        basename=basename,
        drop_params=drop_params,
        register_factory=register_factory,
        overwrite_existing=overwrite_existing,
        layout_cache=layout_cache,
        info=info,
        post_process=post_process,
        debug_names=debug_names,
        tags=tags,
        lvs_equivalent_ports=lvs_equivalent_ports,
        ports=ports,
    )

    if _func is not None:
        c.is_gf_cell = True
        return cast(ComponentFunc[ComponentParams], c)

    @wraps(c)
    def wrapper(
        func: ComponentFunc[ComponentParams],
    ) -> ComponentFunc[ComponentParams]:
        decorated = c(func)
        decorated.is_gf_cell = True
        return cast(ComponentFunc[ComponentParams], decorated)

    return wrapper


class ComponentAllAngleFunc(Protocol[ComponentParams]):
    __name__: str

    def __call__(
        self, *args: ComponentParams.args, **kwargs: ComponentParams.kwargs
    ) -> ComponentAllAngle: ...


@overload
def vcell(
    _func: ComponentAllAngleFunc[ComponentParams], /
) -> ComponentAllAngleFunc[ComponentParams]: ...


@overload
def vcell(
    *,
    set_settings: bool = True,
    set_name: bool = True,
    check_ports: bool = True,
    basename: str | None = None,
    drop_params: tuple[str, ...] = ("self", "cls"),
    register_factory: bool = True,
    ports: PortsDefinition | None = None,
) -> Callable[
    [ComponentAllAngleFunc[ComponentParams]], ComponentAllAngleFunc[ComponentParams]
]: ...


def vcell(
    _func: ComponentAllAngleFunc[ComponentParams] | None = None,
    /,
    *,
    set_settings: bool = True,
    set_name: bool = True,
    add_port_layers: bool = True,
    cache: Cache[int, Any] | dict[int, Any] | None = None,
    basename: str | None = None,
    drop_params: tuple[str, ...] = ("self", "cls"),
    register_factory: bool = True,
    check_ports: bool = True,
    ports: PortsDefinition | None = None,
) -> (
    ComponentAllAngleFunc[ComponentParams]
    | Callable[
        [ComponentAllAngleFunc[ComponentParams]], ComponentAllAngleFunc[ComponentParams]
    ]
):
    from gdsfactory.component import ComponentAllAngle

    vc = _vcell(  # type: ignore[call-overload]
        _func,
        output_type=ComponentAllAngle,
        set_settings=set_settings,
        set_name=set_name,
        add_port_layers=add_port_layers,
        cache=cache,
        basename=basename,
        drop_params=list(drop_params),
        register_factory=register_factory,
        check_ports=check_ports,
        ports=ports,
    )

    if _func is not None:
        vc.is_gf_vcell = True
        return cast(ComponentAllAngleFunc[ComponentParams], vc)

    @wraps(vc)
    def wrapper(
        func: ComponentAllAngleFunc[ComponentParams],
    ) -> ComponentAllAngleFunc[ComponentParams]:
        decorated = vc(func)
        decorated.is_gf_vcell = True
        return cast(ComponentAllAngleFunc[ComponentParams], decorated)

    return wrapper


def override_defaults(
    func: Callable[[ComponentFunc[ComponentParams]], ComponentFunc[ComponentParams]],
    **kwargs: Any,
) -> Callable[[ComponentFunc[ComponentParams]], ComponentFunc[ComponentParams]]:
    return partial(func, **kwargs)


cell_with_module_name = override_defaults(cell, with_module_name=True)


@overload
def schematic_cell(
    _func: Callable[ComponentParams, DSchematic], /
) -> ComponentFunc[ComponentParams]: ...
@overload
def schematic_cell(
    *,
    set_settings: bool = True,
    set_name: bool = True,
    check_ports: bool = True,
    check_instances: CheckInstances | None = None,
    snap_ports: bool = True,
    add_port_layers: bool = True,
    cache: Cache[int, Any] | dict[int, Any] | None = None,
    basename: str | None = None,
    register_factory: bool = True,
    overwrite_existing: bool | None = None,
    layout_cache: bool | None = None,
    info: dict[str, MetaData] | None = None,
    debug_names: bool | None = None,
    tags: list[str] | None = None,
    with_module_name: bool = False,
    factories: dict[str, ComponentFactory] | None = None,
    routing_strategies: RoutingStrategies | None = None,
) -> Callable[[ComponentFunc[ComponentParams]], ComponentFunc[ComponentParams]]: ...


def schematic_cell(
    _func: Callable[ComponentParams, DSchematic] | None = None,
    /,
    *,
    set_settings: bool = True,
    set_name: bool = True,
    check_ports: bool = True,
    check_instances: CheckInstances | None = None,
    snap_ports: bool = True,
    add_port_layers: bool = True,
    cache: Cache[int, Any] | dict[int, Any] | None = None,
    basename: str | None = None,
    register_factory: bool = True,
    overwrite_existing: bool | None = None,
    layout_cache: bool | None = None,
    info: dict[str, MetaData] | None = None,
    debug_names: bool | None = None,
    tags: list[str] | None = None,
    with_module_name: bool = False,
    lvs_equivalent_ports: list[list[str]] | None = None,
    ports: PortsDefinition | None = None,
    factories: dict[str, ComponentFactory] | None = None,
    routing_strategies: RoutingStrategies | None = None,
) -> Any:
    import gdsfactory as gf

    pdk = gf.get_active_pdk()

    factories = factories or pdk.cells
    routing_strategies = routing_strategies or pdk.routing_strategies
    if _func is None:
        return kf.kcl.schematic_cell(
            output_type=gf.Component,
            set_settings=set_settings,
            set_name=set_name,
            check_ports=check_ports,
            check_instances=check_instances,
            snap_ports=snap_ports,
            add_port_layers=add_port_layers,
            cache=cache,
            basename=basename,
            register_factory=register_factory,
            overwrite_existing=overwrite_existing,
            layout_cache=layout_cache,
            info=info,
            debug_names=debug_names,
            tags=tags,
            factories=factories,
            routing_strategies=routing_strategies,
        )

    return kf.kcl.schematic_cell(
        output_type=gf.Component,
        set_settings=set_settings,
        set_name=set_name,
        check_ports=check_ports,
        check_instances=check_instances,
        snap_ports=snap_ports,
        add_port_layers=add_port_layers,
        cache=cache,
        basename=basename,
        register_factory=register_factory,
        overwrite_existing=overwrite_existing,
        layout_cache=layout_cache,
        info=info,
        debug_names=debug_names,
        tags=tags,
        factories=factories,
        routing_strategies=routing_strategies,
    )(_func)

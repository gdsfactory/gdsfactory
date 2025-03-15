from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, overload

from cachetools import Cache
from kfactory import cell as _cell
from kfactory import vcell as _vcell
from kfactory.conf import CheckInstances
from kfactory.typings import MetaData

if TYPE_CHECKING:
    from gdsfactory.component import Component, ComponentAllAngle

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
) -> (
    ComponentFunc[ComponentParams]
    | Callable[[ComponentFunc[ComponentParams]], ComponentFunc[ComponentParams]]
):
    """Decorator to convert a function into a Component."""
    from gdsfactory import component

    if drop_params is None:
        drop_params = ["self", "cls"]
    if post_process is None:
        post_process = []
    c = _cell(  # type: ignore[call-overload,misc]
        _func,
        output_type=component.Component,
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
    )
    c.is_gf_cell = True
    return c  # type: ignore[no-any-return]


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
) -> Callable[
    [ComponentAllAngleFunc[ComponentParams]], ComponentAllAngleFunc[ComponentParams]
]: ...


def vcell(
    _func: ComponentAllAngleFunc[ComponentParams] | None = None,
    /,
    *,
    set_settings: bool = True,
    set_name: bool = True,
    check_ports: bool = True,
    add_port_layers: bool = True,
    cache: Cache[int, Any] | dict[int, Any] | None = None,
    basename: str | None = None,
    drop_params: tuple[str, ...] = ("self", "cls"),
    register_factory: bool = True,
) -> (
    ComponentAllAngleFunc[ComponentParams]
    | Callable[
        [ComponentAllAngleFunc[ComponentParams]], ComponentAllAngleFunc[ComponentParams]
    ]
):
    vc = _vcell(  # type: ignore[call-overload]
        _func,
        set_settings=set_settings,
        set_name=set_name,
        check_ports=check_ports,
        basename=basename,
        drop_params=list(drop_params),
        register_factory=register_factory,
    )
    vc.is_gf_vcell = True
    return vc  # type: ignore[no-any-return]

"""Cell decorator for functions that return a Component."""
from __future__ import annotations

import functools
import hashlib
import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Type

import toolz
from pydantic import BaseModel, validate_arguments

from gdsfactory.component import Component
from gdsfactory.name import clean_name, get_name_short
from gdsfactory.serialization import clean_dict, clean_value_name

CACHE: Dict[str, Component] = {}

INFO_VERSION = 2

_F = TypeVar("_F", bound=Callable)


class CellReturnTypeError(ValueError):
    pass


def clear_cache() -> None:
    """Clears Component CACHE."""
    global CACHE
    CACHE = {}


def print_cache() -> None:
    for k in CACHE:
        print(k)


def get_source_code(func: Callable) -> str:
    if isinstance(func, functools.partial):
        source = inspect.getsource(func.func)
    elif isinstance(func, toolz.functoolz.Compose):
        source = inspect.getsource(func.first)
    elif callable(func):
        source = inspect.getsource(func)
    else:
        raise ValueError(f"{func!r} needs to be callable")
    return source


class Settings(BaseModel):
    name: str
    module: str
    function_name: str

    info: Dict[str, Any]  # derived properties (length, resistance)
    info_version: int = INFO_VERSION

    full: Dict[str, Any]
    changed: Dict[str, Any]
    default: Dict[str, Any]

    child: Optional[Dict[str, Any]] = None


def cell_without_validator(func: _F) -> _F:
    """Decorator for Component functions.

    Similar to cell decorator but does not enforce argument types.

    I recommend using @cell instead.
    """

    @functools.wraps(func)
    def _cell(*args, **kwargs):
        from gdsfactory.pdk import get_active_pdk

        active_pdk = get_active_pdk()
        cell_decorator_settings = active_pdk.cell_decorator_settings

        with_hash = kwargs.pop("with_hash", cell_decorator_settings.with_hash)
        autoname = kwargs.pop("autoname", cell_decorator_settings.autoname)
        name = kwargs.pop("name", cell_decorator_settings.name)
        cache = kwargs.pop("cache", cell_decorator_settings.cache)
        flatten = kwargs.pop("flatten", cell_decorator_settings.flatten)
        info = kwargs.pop("info", cell_decorator_settings.info)
        prefix = kwargs.pop(
            "prefix",
            func.__name__
            if cell_decorator_settings.prefix is None
            else cell_decorator_settings.prefix,
        )
        max_name_length = kwargs.pop(
            "max_name_length", cell_decorator_settings.max_name_length
        )

        sig = inspect.signature(func)
        args_as_kwargs = dict(zip(sig.parameters.keys(), args))
        args_as_kwargs.update(kwargs)

        default = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.default != inspect._empty
        }

        changed = args_as_kwargs
        full = default.copy()
        full.update(**args_as_kwargs)

        default2 = default.copy()
        changed2 = changed.copy()

        # list of default args as strings
        default_args_list = [
            f"{key}={clean_value_name(default2[key])}" for key in sorted(default.keys())
        ]
        # list of explicitly passed args as strings
        passed_args_list = [
            f"{key}={clean_value_name(changed2[key])}" for key in sorted(changed.keys())
        ]

        # get only the args which are explicitly passed and different from defaults
        changed_arg_set = set(passed_args_list).difference(default_args_list)
        changed_arg_list = sorted(changed_arg_set)

        # if any args were different from default, append a hash of those args.
        # else, keep only the base name
        named_args_string = "_".join(changed_arg_list)

        if changed_arg_list:
            named_args_string = (
                hashlib.md5(named_args_string.encode()).hexdigest()[:8]
                if with_hash
                or len(named_args_string) > 28
                or "'" in named_args_string
                or "{" in named_args_string
                else named_args_string
            )
            name_signature = clean_name(f"{prefix}_{named_args_string}")
        else:
            name_signature = prefix

        # filter the changed dictionary to only keep entries which have truly changed
        changed_arg_names = [carg.split("=")[0] for carg in changed_arg_list]
        changed = {k: changed[k] for k in changed_arg_names}
        default_decorator = active_pdk.default_decorator if active_pdk else None
        name = name or name_signature
        decorator = kwargs.pop("decorator", default_decorator)
        name = get_name_short(name, max_name_length=max_name_length)

        if (
            "args" not in sig.parameters
            and "kwargs" not in sig.parameters
            and "settings" not in sig.parameters
        ):
            for key in kwargs:
                if key not in sig.parameters.keys():
                    raise TypeError(
                        f"{func.__name__!r}() got invalid argument {key!r}\n"
                        f"valid arguments are {list(sig.parameters.keys())}"
                    )

        if cache and name in CACHE:
            # print(f"CACHE LOAD {name} {func.__name__}({named_args_string})")
            return CACHE[name]
        # print(f"BUILD {name} {func.__name__}({named_args_string})")

        if not callable(func):
            raise ValueError(
                f"{func!r} is not callable! @cell decorator is only for functions"
            )

        component = func(*args, **kwargs)

        # if the component is already in the cache, but under a different alias,
        # make sure we use a copy, so we don't run into mutability errors
        if id(component) in [id(v) for v in CACHE.values()]:
            component = component.copy()

        metadata_child = (
            dict(component.child.settings) if hasattr(component, "child") else None
        )

        if not isinstance(component, Component):
            raise CellReturnTypeError(
                f"function {func.__name__!r} return type = {type(component)}",
                "make sure that functions with @cell decorator return a Component",
            )

        if metadata_child and component.get_child_name:
            component_name = f"{metadata_child.get('name')}_{name}"
            component_name = get_name_short(
                component_name, max_name_length=max_name_length
            )
        else:
            component_name = name

        if autoname and not hasattr(component, "imported_gds"):
            component.name = component_name

        if component.info is None:
            component.info = {}

        component.info.update(**info)

        if not hasattr(component, "imported_gds"):
            component.settings = Settings(
                name=component_name,
                module=func.__module__,
                function_name=func.__name__,
                changed=clean_dict(changed),
                default=clean_dict(default),
                full=clean_dict(full),
                info=component.info,
                child=metadata_child,
            )

        if decorator:
            if not callable(decorator):
                raise ValueError(f"decorator = {type(decorator)} needs to be callable")
            component_new = decorator(component)
            # if component_new is not component:
            #     component_new.name = name
            component = component_new or component

        if flatten:
            component = component.flatten()

        component.lock()
        CACHE[name] = component
        return component

    return _cell


def cell(func: _F) -> _F:
    """Decorator for Component functions.

    Wraps cell_without_validator
    Validates type annotations with pydantic.

    Implements a cache so that if a component has already been build
    it will return the component from the cache directly.
    This avoids creating two exact Components that have the same name.

    When decorate your functions with @cell you get:

    - cache: avoids creating duplicated Components.
    - name: names Components uniquely name based on parameters.
    - metadata: adds Component.metadata with default, changed and full Args.

    Note the cell decorator does not take any arguments.
    Keyword Args are applied the resulting Component.

    Keyword Args:
        autoname (bool): True renames Component based on args and kwargs.
            True by default.
        name (str): Optional name.
        cache (bool): returns Component from the CACHE if it already exists.
            Avoids having duplicated cells with the same name.
            If False overrides CACHE creates a new Component.
        flatten (bool): False by default. True flattens component hierarchy.
        info: updates Component.info dict.
        prefix (str): name_prefix, defaults to function name.
        max_name_length (int): truncates name beyond some characters with a hash.
        decorator (Callable): function to apply to Component.


    A decorator is a function that runs over a function, so when you do.

    .. code::

        import gdsfactory as gf

        @gf.cell
        def mzi_with_bend():
            c = gf.Component()
            mzi = c << gf.components.mzi()
            bend = c << gf.components.bend_euler()
            return c

    it’s equivalent to

    .. code::

        def mzi_with_bend():
            c = gf.Component()
            mzi = c << gf.components.mzi()
            bend = c << gf.components.bend_euler(radius=radius)
            return c

        mzi_with_bend_decorated = gf.cell(mzi_with_bend)

    """
    return cell_without_validator(validate_arguments(func))


def declarative_cell(cls: Type[Any]) -> Callable[..., Component]:
    """
    TODO:

    - add placements
    - add routes

    """
    cls = dataclass(cls)

    @wraps(cls)
    def cell(*args, **kwargs):
        decl = cls(*args, **kwargs)

        sig = inspect.signature(cls)
        args_as_kwargs = dict(zip(sig.parameters.keys(), args))
        args_as_kwargs.update(kwargs)

        args_list = [
            f"{key}={clean_value_name(args_as_kwargs[key])}"
            for key in sorted(args_as_kwargs.keys())
        ]
        named_args_string = "_".join(args_list)
        component_name = clean_name(f"{cls.__name__}_{named_args_string}")
        if component_name in CACHE:
            return CACHE[component_name]

        decl.instances()
        comp = Component()
        comp.name = component_name

        for k, c in vars(decl).items():
            if not isinstance(c, Component):
                continue
            ref = comp << c
            setattr(comp, k, ref)
            setattr(decl, k, ref)
        for p1, p2 in decl.connections():
            p1.reference.connect(p1.name, p2.reference.ports[p2.name])
        for name, p in decl.ports().items():
            comp.add_port(name, port=p.reference.ports[p.name])
        CACHE[component_name] = comp
        return comp

    return cell


@cell
def wg(length: int = 3, layer: Tuple[int, int] = (1, 0)) -> Component:
    """Dummy component for testing."""
    c = Component()
    width = 0.5
    w = width / 2
    c.add_polygon([(0, -w), (length, -w), (length, w), (0, w)], layer=layer)
    c.add_port(name="o1", center=[0, 0], width=width, orientation=180, layer=layer)
    c.add_port(name="o2", center=[length, 0], width=width, orientation=0, layer=layer)
    return c


@cell
def wg2(wg1=wg):
    """Dummy component for testing."""
    c = Component()
    w = wg1()
    w1 = c << w
    w1.rotate(90)
    c.copy_child_info(w)
    c.add_ports(w1.ports)
    return c


def test_set_name() -> None:
    c = wg(length=3, name="hi_there")
    assert c.name == "hi_there", c.name


@cell
def demo(length: int = 3, wg_width: float = 0.5) -> Component:
    """Demo Dummy cell."""
    c = Component()
    w = length
    h = wg_width
    points = [
        [-w / 2.0, -h / 2.0],
        [-w / 2.0, h / 2],
        [w / 2, h / 2],
        [w / 2, -h / 2.0],
    ]
    c.add_polygon(points)
    return c


def test_names() -> None:
    name_base = demo().name
    assert name_base.split("_")[0] == "demo", name_base

    demo2 = functools.partial(demo, length=3)
    c1 = demo2(length=3)
    c2 = demo(length=3)
    assert c1.name == c2.name, "{c1.name} != {c2.name}"

    c1 = demo(length=3, wg_width=0.5).name
    c2 = demo(wg_width=0.5, length=3).name
    assert c1 == c2, f"{c1} != {c2}"

    name_with_prefix = demo(prefix="hi").name
    assert name_with_prefix.split("_")[0] == "hi", name_with_prefix

    name_args = demo(3).name
    name_kwargs = demo(length=3).name
    assert name_args == name_kwargs, name_with_prefix

    c = wg(length=3.1)
    assert c.settings.changed["length"] == 3.1


@cell
def straight_with_pins(**kwargs) -> Component:
    import gdsfactory as gf

    c = gf.Component()
    ref = c << gf.components.straight()
    c.add_ports(ref.ports)
    gf.add_pins.add_pins(c)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    @gf.declarative_cell
    class mzi:
        delta_length: float = 10.0

        def instances(self):
            self.mmi_in = gf.components.mmi1x2()
            self.mmi_out = gf.components.mmi2x2()
            self.straight_top1 = gf.components.straight(length=self.delta_length / 2)
            self.straight_top2 = gf.components.straight(length=self.delta_length / 2)
            self.bend_top1 = gf.components.bend_euler()
            self.bend_top2 = gf.components.bend_euler().mirror()
            self.bend_top3 = gf.components.bend_euler().mirror()
            self.bend_top4 = gf.components.bend_euler()
            self.bend_btm1 = gf.components.bend_euler().mirror()
            self.bend_btm2 = gf.components.bend_euler()
            self.bend_btm3 = gf.components.bend_euler()
            self.bend_btm4 = gf.components.bend_euler().mirror()

        def connections(self):
            return [
                (self.bend_top1.ports["o1"], self.mmi_in.ports["o2"]),
                (self.straight_top1.ports["o1"], self.bend_top1.ports["o2"]),
                (self.bend_top2.ports["o1"], self.straight_top1.ports["o2"]),
                (self.bend_top3.ports["o1"], self.bend_top2.ports["o2"]),
                (self.straight_top2.ports["o1"], self.bend_top3.ports["o2"]),
                (self.bend_top4.ports["o1"], self.straight_top2.ports["o2"]),
                (self.bend_btm1.ports["o1"], self.mmi_in.ports["o3"]),
                (self.bend_btm2.ports["o1"], self.bend_btm1.ports["o2"]),
                (self.bend_btm3.ports["o1"], self.bend_btm2.ports["o2"]),
                (self.bend_btm4.ports["o1"], self.bend_btm3.ports["o2"]),
                (self.mmi_out.ports["o1"], self.bend_btm4.ports["o2"]),
            ]

        def ports(self):
            return {
                "o1": self.mmi_in.ports["o1"],
                "o2": self.mmi_out.ports["o3"],
                "o3": self.mmi_out.ports["o4"],
            }

    @gf.declarative_cell
    class mzi_lattice_fixme:
        """does not work"""

        delta_lengths: Tuple[float, ...] = (10.0, 100)

        def instances(self):
            self.mzis = {}
            for i, d in enumerate(self.delta_lengths):
                self.mzis[i] = mzi(delta_length=d).ref()

        def connections(self):
            return [
                (self.mzis[i + 1]["o1"], self.mzis[i]["o2"])
                for i in range(len(self.mzis) - 1)
            ]

        def ports(self):
            return {
                "o1": self.mzis[0].ports["o1"],
                "o2": self.mzis[len(self.mzis) - 1].ports["o2"],
            }

    @gf.declarative_cell
    class mzi_lattice:
        """Works"""

        delta_length1: float = 10.0
        delta_length2: float = 100.0

        def instances(self):
            self.mzi1 = mzi(delta_length=self.delta_length1)
            self.mzi2 = mzi(delta_length=self.delta_length2)

        def connections(self):
            return [
                (self.mzi2["o1"], self.mzi1.ports["o2"]),
            ]

        def ports(self):
            return {
                "o1": self.mzi1.ports["o1"],
                "o2": self.mzi2.ports["o2"],
            }

    @gf.declarative_cell
    class mzi_lattice_with_routes:
        """Works"""

        delta_length1: float = 10.0
        delta_length2: float = 100.0

        def instances(self):
            self.mzi1 = mzi(delta_length=self.delta_length1)
            self.mzi2 = mzi(delta_length=self.delta_length2)

        def placements(self):
            pass

        def routes(self):
            return [
                (self.mzi2["o1"], self.mzi1.ports["o2"]),
            ]

        def ports(self):
            return {
                "o1": self.mzi1.ports["o1"],
                "o2": self.mzi2.ports["o2"],
            }

    # c = mzi(delta_length=10)
    # c = mzi()
    c = mzi_lattice()
    c.show()

    # test_names()
    # c = wg()
    # test_import_gds_settings()

    # import gdsfactory as gf

    # c = gf.components.straight()
    # c = gf.components.straight()
    # print(c.name)

    # c = wg3()
    # print(c.name)

    # print(wg(length=3).name)
    # print(wg(length=3.0).name)
    # print(wg().name)

    # import gdsfactory as gf

    # gdspath = gf.PATH.gdsdir / "mzi2x2.gds"
    # c = gf.import_gds(gdspath)
    # c3 = gf.routing.add_fiber_single(c)
    # c3.show()

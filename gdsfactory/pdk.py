"""PDK stores layers, cross_sections, cell functions ..."""

from __future__ import annotations

import importlib
import pathlib
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property, partial
from typing import Any, cast, overload

import kfactory as kf
import yaml
from kfactory.layer import LayerEnum
from pydantic import BaseModel, ConfigDict, Field

from gdsfactory import logger
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.config import CONF
from gdsfactory.cross_section import CrossSection, Section
from gdsfactory.cross_section import xsection as cross_section_xsection
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.read.from_yaml_template import cell_from_yaml_template
from gdsfactory.serialization import clean_value_json, convert_tuples_to_lists
from gdsfactory.symbols import floorplan_with_block_letters
from gdsfactory.technology import LayerStack, LayerViews, klayout_tech
from gdsfactory.typings import (
    CellAllAngleSpec,
    CellSpec,
    ComponentAllAngleFactory,
    ComponentFactory,
    ComponentSpec,
    ConnectivitySpec,
    CrossSectionFactory,
    CrossSectionSpec,
    LayerSpec,
    LayerTransitions,
    MaterialSpec,
    PathType,
    RoutingStrategies,
)

_ACTIVE_PDK: Pdk | None = None
component_settings = ["function", "component", "settings"]
cross_section_settings = ["function", "cross_section", "settings"]

constants = {
    "fiber_input_to_output_spacing": 200.0,
    "metal_spacing": 10.0,
    "pad_pitch": 100.0,
    "pad_size": (80, 80),
}

nm = 1e-3


def evanescent_coupler_sample() -> None:
    """Evanescent coupler example.

    Args:
      coupler_length: length of coupling (min: 0.0, max: 200.0, um).
    """
    pass


def extract_args_from_docstring(docstring: str) -> dict[str, Any]:
    """This function extracts settings from a function's docstring for uPDK format.

    Args:
        docstring: The function from which to extract YAML in the docstring.

    Returns:
        settings (dict): The extracted YAML data as a dictionary.
    """
    args_dict: dict[str, Any] = {}

    docstring_lines = docstring.split("\n")
    for line in docstring_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Args:"):
            continue
        if len(line.split(":")) != 2:
            continue
        name, description = line.split(":")
        name = name.strip()
        description_parts = description.split("(")
        doc = description_parts[0].strip()
        try:
            min_max_unit = description_parts[1].strip(")").split(",")
            min_val = float(min_max_unit[0].split(":")[1].strip())
            max_val = float(min_max_unit[1].split(":")[1].strip())
            unit = min_max_unit[2].strip()
        except IndexError:
            min_val = max_val = 0
            unit = None

        args_dict[name] = {
            "doc": doc,
            "min": min_val,
            "max": max_val,
            "type": "float",
            "unit": unit,
            "value": (min_val + max_val) / 2,  # setting default value as the midpoint
        }

    return args_dict


class Pdk(BaseModel):
    """Store layers, cross_sections, cell functions, simulation_settings ...

    only one Pdk can be active at a given time.

    Parameters:
        name: PDK name.
        version: PDK version.
        cross_sections: dict of cross_sections factories.
        cells: dict of parametric cells that return Components.
        containers: dict of containers that return Components. A container is a cell that contains other cells.
        models: dict of models names to functions.
        symbols: dict of symbols names to functions.
        default_symbol_factory:
        base_pdk: a pdk to copy from and extend.
        default_decorator: decorate all cells, if not otherwise defined on the cell.
        layers: maps name to gdslayer/datatype.
            For example dict(si=(1, 0), sin=(34, 0)).
        layer_stack: maps name to layer numbers, thickness, zmin, sidewall_angle.
            if can also contain material properties
            (refractive index, nonlinear coefficient, sheet resistance ...).
        layer_views: includes layer name to color, opacity and pattern.
        layer_transitions: transitions between different cross_sections.
        constants: dict of constants for the PDK.
        materials_index: material spec names to material spec, which can be:
            string: material name.
            float: refractive index.
            float, float: refractive index real and imaginary part.
            function: function of wavelength.
        routing_strategies: functions enabled to route.
        bend_points_distance: default points distance for bends in um.
        connectivity: defines connectivity between layers through vias.

    """

    name: str
    version: str = ""
    cross_sections: dict[str, CrossSectionFactory] = Field(
        default_factory=dict, exclude=True
    )
    cross_section_default_names: dict[str, str] = Field(
        default_factory=dict, exclude=True
    )
    cells: dict[str, ComponentFactory] = Field(default_factory=dict, exclude=True)
    containers: dict[str, ComponentFactory] = Field(default_factory=dict, exclude=True)
    models: dict[str, Callable[..., Any]] = Field(default_factory=dict, exclude=True)
    symbols: dict[str, ComponentFactory] = Field(default_factory=dict)
    default_symbol_factory: ComponentFactory = Field(
        default=floorplan_with_block_letters, exclude=True
    )
    base_pdks: list[Pdk] = Field(default_factory=list)
    default_decorator: Callable[[Component], None] | None = Field(
        default=None, exclude=True
    )
    layers: type[LayerEnum] | None = None
    layer_stack: LayerStack | None = None
    layer_views: LayerViews | None = None
    layer_transitions: LayerTransitions = Field(default_factory=dict)
    constants: dict[str, Any] = constants
    materials_index: dict[str, MaterialSpec] = Field(default_factory=dict)
    routing_strategies: RoutingStrategies | None = None
    bend_points_distance: float = 20 * nm
    connectivity: Sequence[ConnectivitySpec] | None = None
    max_cellname_length: int = CONF.max_cellname_length

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def xsection(
        self, func: Callable[..., CrossSection]
    ) -> Callable[..., CrossSection]:
        """Decorator to register a cross section function.

        Ensures that the cross-section name matches the name of the function
        that generated it when created using default parameters.

        Reuses the core xsection decorator from cross_section.py while maintaining
        PDK-specific storage of cross sections.

        .. code-block:: python

            @pdk.xsection
            def xs_sc(width=TECH.width_sc, radius=TECH.radius_sc):
                return gf.cross_section.cross_section(width=width, radius=radius)
        """
        decorated_func = cross_section_xsection(func)

        self.cross_sections[func.__name__] = decorated_func

        return decorated_func

    def activate(self, force: bool = False) -> None:
        """Set current pdk to the active pdk (if not already active)."""
        global _ACTIVE_PDK
        if not force and _ACTIVE_PDK and _ACTIVE_PDK.name is self.name:
            return

        logger.debug(f"{self.name!r} PDK {self.version} is now active")

        for pdk in self.base_pdks:
            cross_sections = pdk.cross_sections
            cross_sections.update(self.cross_sections)
            cells = pdk.cells
            self.cross_sections = cross_sections
            cells.update(self.cells)
            self.cells.update(cells)

        _set_active_pdk(self)

    def register_cells(self, **kwargs: Any) -> None:
        """Register cell factories."""
        for name, cell in kwargs.items():
            if not callable(cell):
                raise ValueError(
                    f"{cell} is not callable, make sure you register "
                    "cells functions that return a Component"
                )
            if name in self.cells:
                warnings.warn(f"Overwriting cell {name!r}", stacklevel=3)

            self.cells[name] = cell

    def register_cross_sections(self, **kwargs: Any) -> None:
        """Register cross_sections factories."""
        for name, cross_section in kwargs.items():
            if not callable(cross_section):
                raise ValueError(
                    f"{cross_section} is not callable, make sure you register "
                    "cross_section functions that return a CrossSection"
                )
            if name in self.cross_sections:
                warnings.warn(f"Overwriting cross_section {name!r}", stacklevel=3)
            self.cross_sections[name] = cross_section

    def register_cells_yaml(
        self,
        dirpath: PathType | None = None,
        update: bool = False,
        **kwargs: Any,
    ) -> None:
        """Load *.pic.yml YAML files and register them as cells.

        Args:
            dirpath: directory to recursive search for YAML cells.
            update: does not raise ValueError if cell already registered.
            kwargs: cell_name: cell function. To update cells dict.

        Keyword Args:
            cell_name: cell function. To update cells dict.

        """
        message = "Updated" if update else "Registered"

        if dirpath:
            dirpath = pathlib.Path(dirpath)

            if not dirpath.is_dir():
                raise ValueError(f"{dirpath!r} needs to be a directory.")

            for filepath in dirpath.glob("**/*.pic.yml"):
                name = filepath.stem.split(".")[0]
                if not update and name in self.cells:
                    raise ValueError(
                        f"ERROR: Cell name {name!r} from {filepath} already registered."
                    )
                self.cells[name] = cell_from_yaml_template(filepath, name=name)
                logger.info(f"{message} cell {name!r}")

        for k, v in kwargs.items():
            if not update and k in self.cells:
                raise ValueError(f"ERROR: Cell name {k!r} already registered.")
            self.cells[k] = v
            logger.info(f"{message} cell {k!r}")

    def remove_cell(self, name: str) -> None:
        """Removes cell from a PDK."""
        if name not in self.cells:
            raise ValueError(f"{name!r} not in {list(self.cells.keys())}")
        self.cells.pop(name)
        logger.info(f"Removed cell {name!r}")

    @overload
    def get_cell(self, cell: CellSpec, **kwargs: Any) -> ComponentFactory: ...
    @overload
    def get_cell(
        self, cell: CellAllAngleSpec, **kwargs: Any
    ) -> ComponentAllAngleFactory: ...

    def get_cell(
        self, cell: CellSpec | CellAllAngleSpec, **kwargs: Any
    ) -> ComponentFactory | ComponentAllAngleFactory:
        """Returns ComponentFactory from a cell spec."""
        cells_and_containers = self._get_cells_and_containers()

        if callable(cell):
            return cell
        elif isinstance(cell, str):
            if cell not in cells_and_containers:
                matching_cells = [c for c in cells_and_containers if cell in c]
                raise ValueError(
                    f"{cell!r} from PDK {self.name!r} not in cells: Did you mean {matching_cells}?"
                )
            return cells_and_containers[cell]
        else:
            for key in cell.keys():
                if key not in component_settings:
                    raise ValueError(
                        f"Invalid setting {key!r} not in {component_settings}"
                    )
            settings = dict(cell.get("settings", {}))
            settings.update(**kwargs)

            cell_name = cell.get("function")
            if not isinstance(cell_name, str) or cell_name not in cells_and_containers:
                matching_cells = [c for c in cells_and_containers if c in cell.keys()]
                raise ValueError(
                    f"{cell!r} from PDK {self.name!r} not in cells: Did you mean {matching_cells}?"
                )
            cell = cells_and_containers[cell_name]
            return partial(cell, **settings)

    def get_component(
        self,
        component: ComponentSpec,
        settings: Mapping[str, Any] | None = None,
        include_containers: bool = True,
        **kwargs: Any,
    ) -> Component:
        """Returns component from a component spec."""
        if include_containers:
            cells = self._get_cells_and_containers()
        else:
            cells = self.cells

        return self._get_component(
            component=component, cells=cells, settings=settings, **kwargs
        )

    def _get_cells_and_containers(self) -> dict[str, ComponentFactory]:
        """Returns a dictionary of cells and containers."""
        cells_and_containers = {**self.cells, **self.containers}
        conflicting_names = set(self.cells.keys()).intersection(self.containers.keys())
        if conflicting_names:
            raise ValueError(
                f"PDK {self.name!r} has overlapping cell names between cells and containers: {list(conflicting_names)}. "
            )
        return cells_and_containers

    def get_symbol(self, component: ComponentSpec, **kwargs: Any) -> Component:
        """Returns a component's symbol from a component spec."""
        # this is a pretty rough first implementation
        try:
            return self._get_component(
                component=component, cells=self.symbols, **kwargs
            )
        except ValueError:
            component = self.get_component(component, **kwargs)
            return self.default_symbol_factory(component)

    def _get_component(
        self,
        component: ComponentSpec,
        cells: dict[str, ComponentFactory],
        settings: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Component:
        """Returns component from a component spec.

        Args:
            component: Component, ComponentFactory, string or dict.
            cells: dict of cells.
            settings: settings to override.
            kwargs: settings to override.
        """
        cell_names = sorted(cells)

        settings = settings or {}
        kwargs = kwargs or {}
        kwargs.update(settings)

        if isinstance(component, kf.ProtoTKCell):
            return Component(base=component.base)
        elif isinstance(component, kf.VKCell):
            return ComponentAllAngle(base=component.base)
        elif callable(component):
            _component = component(**kwargs)
            return type(_component)(base=_component.base)  # type: ignore[call-overload,no-any-return]
        elif isinstance(component, str):
            if component not in cell_names:
                substring = component
                matching_cells: list[str] = []

                # Reduce the length of the cell string until we find matches
                while substring and not matching_cells:
                    matching_cells = [c for c in cells if substring in c]
                    if not matching_cells:
                        substring = substring[:-1]  # Remove the last character

                raise ValueError(
                    f"{component!r} not in PDK {self.name!r}. Did you mean {matching_cells}?"
                )
            return cells[component](**kwargs)
        elif isinstance(component, dict):
            for key in component.keys():
                if key not in component_settings:
                    raise ValueError(
                        f"Invalid setting {key!r} not in {component_settings}"
                    )
            settings = dict(component.get("settings", {}))
            settings.update(**kwargs)

            cell_name = component.get("component", None)
            cell_name = cell_name or component.get("function")
            cell_name = cell_name.split(".")[-1]

            if not isinstance(cell_name, str) or cell_name not in cells:
                matching_cells = [c for c in cells if cell_name in c]
                raise ValueError(
                    f"{cell_name!r} from PDK {self.name!r} not in cells: Did you mean {matching_cells}?"
                )
            return cells[cell_name](**settings)
        else:
            raise ValueError(
                "get_component expects a ComponentSpec (Component, ComponentFactory, "
                f"string or dict), got {type(component)}"
            )

    def get_cross_section(
        self, cross_section: CrossSectionSpec, **kwargs: Any
    ) -> CrossSection:
        """Returns cross_section from a cross_section spec.

        Args:
            cross_section: CrossSection, CrossSectionFactory, Transition, string or dict.
            kwargs: settings to override.
        """
        if callable(cross_section):
            return cross_section(**kwargs)
        elif isinstance(cross_section, str):
            if cross_section not in self.cross_sections:
                cross_sections = list(self.cross_sections.keys())
                raise ValueError(f"{cross_section!r} not in {cross_sections}")
            xs = self.cross_sections[cross_section]
            return xs(**kwargs)
        elif isinstance(cross_section, dict):
            xs_name = cross_section.get("cross_section", None)
            settings = cross_section.get("settings", {})
            return self.get_cross_section(xs_name, **settings)
        elif isinstance(cross_section, CrossSection):
            if kwargs:
                warnings.warn(
                    f"{kwargs} ignored for cross_section {cross_section.name!r}",
                    stacklevel=3,
                )

            return cross_section
        elif isinstance(cross_section, kf.DCrossSection | kf.SymmetricalCrossSection):
            if isinstance(cross_section, kf.DCrossSection):
                cross_section_ = cross_section.base
            else:
                cross_section_ = cross_section
            section_ = Section(
                width=gf.kcl.to_um(cross_section_.width),
                layer=gf.kcl.layout.layer(cross_section_.main_layer),
            )
            xs_ = CrossSection(
                sections=(section_,),
                radius=kf.kcl.to_um(cross_section_.radius),
                radius_min=kf.kcl.to_um(cross_section_.radius_min),
            )
            xs_._name = cross_section_.name
            return xs_
        else:
            raise ValueError(
                "get_cross_section expects a CrossSectionSpec (CrossSection, "
                f"CrossSectionFactory, Transition, string or dict), got {type(cross_section)}"
            )

    def get_layer(self, layer: LayerSpec | kf.kdb.LayerInfo) -> LayerEnum | int:
        """Returns layer from a layer spec."""
        if isinstance(layer, LayerEnum | int):
            return layer
        elif isinstance(layer, tuple | list):
            if len(layer) != 2:
                raise ValueError(f"{layer!r} needs two integer numbers.")
            return kf.kcl.layout.layer(*layer)
        elif isinstance(layer, kf.kdb.LayerInfo):
            return layer.layer
        else:
            if not hasattr(self.layers, layer):
                raise ValueError(f"{layer!r} not in {self.layers}")
            return cast(LayerEnum, getattr(self.layers, layer))

    def get_layer_name(self, layer: LayerSpec) -> str:
        layer_index = self.get_layer(layer)
        assert self.layers is not None
        try:
            return str(self.layers[layer_index])  # type: ignore[index]
        except Exception:
            try:
                return str(self.layers(layer_index))  # type: ignore[call-arg]
            except Exception:
                raise ValueError(f"Could not find name for layer {layer_index}")

    def get_layer_views(self) -> LayerViews:
        if self.layer_views is None:
            raise ValueError(f"layer_views for Pdk {self.name!r} is None")
        return self.layer_views

    def get_layer_stack(self) -> LayerStack:
        if self.layer_stack is None:
            raise ValueError(f"layer_stack for Pdk {self.name!r} is None")
        return self.layer_stack

    def get_constant(self, key: str) -> Any:
        if key not in self.constants:
            constants = list(self.constants.keys())
            raise ValueError(f"{key!r} not in {constants}")
        return self.constants[key]

    def to_updk(self, exclude: Sequence[str] | None = None) -> str:
        """Export to uPDK YAML definition."""
        from gdsfactory.components import bbox_to_points

        exclude = exclude or []
        _blocks = {
            cell_name: cell()
            for cell_name, cell in self.cells.items()
            if cell_name not in exclude
        }
        blocks: dict[str, dict[str, Any]] = {}
        for name, c in _blocks.items():
            if c.__doc__ is None:
                continue
            extra_args = extract_args_from_docstring(c.__doc__)

            blocks[name] = dict(
                bbox=bbox_to_points(c.dbbox()),
                doc=c.__doc__.split("\n")[0],
                settings=extra_args,
                parameters={
                    sname: {
                        "value": clean_value_json(svalue),
                        "type": str(svalue.__class__.__name__),
                        "doc": extra_args.get(sname, {}).get("doc", None),
                        "min": extra_args.get(sname, {}).get("min", 0),
                        "max": extra_args.get(sname, {}).get("max", 0),
                        "unit": extra_args.get(sname, {}).get("unit", None),
                    }
                    for sname, svalue in c.settings
                    if isinstance(svalue, str | float | int)
                },
                pins={
                    port.name: {
                        "width": port.width,
                        "xsection": port.cross_section.name
                        if hasattr(port, "cross_section")
                        else "",
                        "xya": [
                            float(port.center[0]),
                            float(port.center[1]),
                            float(port.orientation),
                        ],
                        "alias": port.info.get("alias"),
                        "doc": port.info.get("doc"),
                    }
                    for port in c.ports
                },
            )
        xsections = {
            xs_name: self.get_cross_section(xs_name)
            for xs_name in self.cross_sections.keys()
        }
        xsections_widths = {
            xs_name: dict(width=xsection.width)
            for xs_name, xsection in xsections.items()
        }

        header = dict(description=self.name)

        d = {"blocks": blocks, "xsections": xsections_widths, "header": header}
        return yaml.dump(convert_tuples_to_lists(d))

    def get_cross_section_name(self, cross_section: CrossSection) -> str:
        xs_name = next(
            (
                key
                for key, value in self.cross_sections.items()
                if value() == cross_section
            ),
            None,
        )
        return xs_name or cross_section.name

    @cached_property
    def klayout_technology(self) -> klayout_tech.KLayoutTechnology:
        """Returns a KLayoutTechnology from the PDK.

        Raises:
            UserWarning if required properties for generating a KLayoutTechnology are not defined.
        """
        try:
            return klayout_tech.KLayoutTechnology(
                name=self.name,
                layer_views=self.layer_views,
                connectivity=self.connectivity,
                layer_map=self.layers,  # type: ignore[arg-type]
                layer_stack=self.layer_stack,
            )
        except AttributeError as e:
            raise UserWarning(
                "Required properties for generating a KLayoutTechnology are not defined. "
                "Check the error for missing property"
            ) from e


def get_active_pdk(name: str | None = None) -> Pdk:
    """Returns active PDK.

    By default it will return the PDK defined in the name or config file.
    Otherwise it will return the generic PDK.
    """
    global _ACTIVE_PDK

    if _ACTIVE_PDK is None:
        name = name or CONF.pdk
        if name == "generic":
            return get_generic_pdk()
        elif name:
            pdk_module = importlib.import_module(name or CONF.pdk)
            pdk_module.PDK.activate()

        else:
            raise ValueError("no active pdk")
    assert _ACTIVE_PDK is not None, "Could not find active PDK"
    return _ACTIVE_PDK


def get_material_index(material: MaterialSpec, *args: Any, **kwargs: Any) -> Component:
    active_pdk = get_active_pdk()
    if not hasattr(active_pdk, "get_material_index"):
        raise NotImplementedError(
            "The active PDK does not implement 'get_material_index'"
        )
    return active_pdk.get_material_index(material, *args, **kwargs)  # type: ignore[no-any-return]


def get_component(
    component: ComponentSpec, settings: Mapping[str, Any] | None = None, **kwargs: Any
) -> Component:
    return get_active_pdk().get_component(component, settings=settings, **kwargs)


@overload
def get_cell(cell: CellSpec, **kwargs: Any) -> ComponentFactory: ...
@overload
def get_cell(cell: CellAllAngleSpec, **kwargs: Any) -> ComponentAllAngleFactory: ...


def get_cell(
    cell: CellSpec | CellAllAngleSpec, **kwargs: Any
) -> ComponentFactory | ComponentAllAngleFactory:
    return get_active_pdk().get_cell(cell, **kwargs)


def get_cross_section(cross_section: CrossSectionSpec, **kwargs: Any) -> CrossSection:
    return get_active_pdk().get_cross_section(cross_section, **kwargs)


def get_layer(layer: LayerSpec | kf.kdb.LayerInfo) -> LayerEnum | int:
    return get_active_pdk().get_layer(layer)


def get_layer_name(layer: LayerSpec) -> str:
    return get_active_pdk().get_layer_name(layer)


def get_layer_tuple(layer: LayerSpec) -> tuple[int, int]:
    """Returns layer tuple (layer, datatype) from a layer spec."""
    layer_index = get_layer(layer)
    info = kf.kcl.get_info(layer_index)
    return info.layer, info.datatype


def get_layer_views() -> LayerViews:
    return get_active_pdk().get_layer_views()


def get_layer_stack() -> LayerStack:
    return get_active_pdk().get_layer_stack()


def get_constant(constant_name: Any) -> Any:
    """If constant_name is a string returns a the value from the dict."""
    return (
        get_active_pdk().get_constant(constant_name)
        if isinstance(constant_name, str)
        else constant_name
    )


def _set_active_pdk(pdk: Pdk) -> None:
    global _ACTIVE_PDK
    _ACTIVE_PDK = pdk


def get_routing_strategies() -> RoutingStrategies:
    """Gets a dictionary of named routing functions available to the PDK, if defined, or gdsfactory defaults otherwise."""
    from gdsfactory.routing.factories import (
        routing_strategies as default_routing_strategies,
    )

    routing_strategies = get_active_pdk().routing_strategies
    if routing_strategies is None:
        routing_strategies = default_routing_strategies
    return routing_strategies


if __name__ == "__main__":
    import gdsfactory as gf

    sample_routing_sbend = """
instances:
    cp1:
      component: coupler

    cp2:
      component: coupler

placements:
    cp1:
        x: 0

    cp2:
        x: 300
        y: 300

routes:
    bundle1:
        links:
          cp1,o3: cp2,o2
        routing_strategy: route_bundle_sbend

"""

    c = gf.read.from_yaml(sample_routing_sbend)
    c.show()

    # l1 = get_layer((1, 0))
    # l2 = get_layer((3, 0))
    # print(l1)
    # print(l2)

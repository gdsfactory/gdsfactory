"""PDK stores layers, cross_sections, cell functions ..."""

from __future__ import annotations

import importlib
import pathlib
import warnings
from collections.abc import Callable
from functools import cached_property, partial
from typing import Any

import kfactory as kf
import numpy as np
import omegaconf
from kfactory import LayerEnum
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field

from gdsfactory import logger
from gdsfactory.config import CONF
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.read.from_yaml_template import cell_from_yaml_template
from gdsfactory.symbols import floorplan_with_block_letters
from gdsfactory.technology import LayerStack, LayerViews, klayout_tech
from gdsfactory.typings import (
    CellSpec,
    Component,
    ComponentBase,
    ComponentFactory,
    ComponentSpec,
    ConnectivitySpec,
    CrossSection,
    CrossSectionOrFactory,
    CrossSectionSpec,
    Layer,
    LayerSpec,
    MaterialSpec,
    PathType,
    Transition,
)

_ACTIVE_PDK = None
component_settings = ["function", "component", "settings"]
cross_section_settings = ["function", "cross_section", "settings"]

constants = {
    "fiber_array_spacing": 127.0,
    "fiber_spacing": 50.0,
    "fiber_input_to_output_spacing": 200.0,
    "metal_spacing": 10.0,
    "pad_spacing": 100.0,
    "pad_size": (80, 80),
}

nm = 1e-3


def evanescent_coupler_sample() -> None:
    """Evanescent coupler example.

    Args:
      coupler_length: length of coupling (min: 0.0, max: 200.0, um).
    """
    pass


def extract_args_from_docstring(docstring: str) -> dict[str, Any] | None:
    """This function extracts settings from a function's docstring for uPDK format.

    Args:
        docstring: The function from which to extract YAML in the docstring.

    Returns:
        settings (dict): The extracted YAML data as a dictionary.
    """
    args_dict = {}

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
        cross_sections: dict of cross_sections factories.
        cells: dict of parametric cells that return Components.
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
    cross_sections: dict[str, CrossSectionOrFactory] = Field(
        default_factory=dict, exclude=True
    )
    cells: dict[str, ComponentFactory] = Field(default_factory=dict, exclude=True)
    models: dict[str, Callable] = Field(default_factory=dict, exclude=True)
    symbols: dict[str, ComponentFactory] = Field(default_factory=dict)
    default_symbol_factory: Callable = Field(
        default=floorplan_with_block_letters, exclude=True
    )
    base_pdks: list[Pdk] = Field(default_factory=list)
    default_decorator: Callable[[Component], None] | None = Field(
        default=None, exclude=True
    )
    layers: type[LayerEnum] | None = None
    layer_stack: LayerStack | None = None
    layer_views: LayerViews | None = None
    layer_transitions: dict[LayerSpec | tuple[Layer, Layer], ComponentSpec] = Field(
        default_factory=dict
    )
    constants: dict[str, Any] = constants
    materials_index: dict[str, MaterialSpec] = Field(default_factory=dict)
    routing_strategies: dict[str, Callable] | None = None
    bend_points_distance: float = 20 * nm
    connectivity: list[ConnectivitySpec] | None = None
    max_cellname_length: int = CONF.max_cellname_length

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def activate(self) -> None:
        """Set current pdk to the active pdk (if not already active)."""
        logger.debug(f"{self.name!r} PDK is now active")

        for pdk in self.base_pdks:
            cross_sections = pdk.cross_sections
            cross_sections.update(self.cross_sections)
            cells = pdk.cells
            self.cross_sections = cross_sections
            cells.update(self.cells)
            self.cells.update(cells)

        _set_active_pdk(self)

    def register_cells(self, **kwargs) -> None:
        """Register cell factories."""
        for name, cell in kwargs.items():
            if not callable(cell):
                raise ValueError(
                    f"{cell} is not callable, make sure you register "
                    "cells functions that return a Component"
                )
            if name in self.cells:
                warnings.warn(f"Overwriting cell {name!r}")

            self.cells[name] = cell

    def register_cross_sections(self, **kwargs) -> None:
        """Register cross_sections factories."""
        for name, cross_section in kwargs.items():
            if not callable(cross_section):
                raise ValueError(
                    f"{cross_section} is not callable, make sure you register "
                    "cross_section functions that return a CrossSection"
                )
            if name in self.cross_sections:
                warnings.warn(f"Overwriting cross_section {name!r}")
            self.cross_sections[name] = cross_section

    def register_cells_yaml(
        self,
        dirpath: PathType | None = None,
        update: bool = False,
        **kwargs,
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

    def remove_cell(self, name: str):
        """Removes cell from a PDK."""
        if name not in self.cells:
            raise ValueError(f"{name!r} not in {list(self.cells.keys())}")
        self.cells.pop(name)
        logger.info(f"Removed cell {name!r}")

    def get_cell(self, cell: CellSpec, **kwargs) -> ComponentFactory:
        """Returns ComponentFactory from a cell spec."""
        cells = set(self.cells.keys())

        if callable(cell):
            return cell
        elif isinstance(cell, str):
            if cell not in cells:
                cells = list(self.cells.keys())
                raise ValueError(
                    f"{cell!r} from PDK {self.name!r} not in cells: {cells} "
                )
            return self.cells[cell]
        elif isinstance(cell, dict | DictConfig):
            for key in cell.keys():
                if key not in component_settings:
                    raise ValueError(
                        f"Invalid setting {key!r} not in {component_settings}"
                    )
            settings = dict(cell.get("settings", {}))
            settings.update(**kwargs)

            cell_name = cell.get("function")
            if not isinstance(cell_name, str) or cell_name not in cells:
                cells = list(self.cells.keys())
                raise ValueError(
                    f"{cell_name!r} from PDK {self.name!r} not in cells: {cells} "
                )
            cell = self.cells[cell_name]
            return partial(cell, **settings)
        else:
            raise ValueError(
                "get_cell expects a CellSpec (ComponentFactory, string or dict),"
                f"got {type(cell)}"
            )

    def get_component(
        self, component: ComponentSpec, settings=None, **kwargs
    ) -> Component:
        """Returns component from a component spec."""
        return self._get_component(
            component=component, cells=self.cells, settings=settings, **kwargs
        )

    def get_symbol(self, component: ComponentSpec, **kwargs) -> Component:
        """Returns a component's symbol from a component spec."""
        # this is a pretty rough first implementation
        try:
            self._get_component(component=component, cells=self.symbols, **kwargs)
        except ValueError:
            component = self.get_component(component, **kwargs)
            return self.default_symbol_factory(component)

    def _get_component(
        self,
        component: ComponentSpec,
        cells: dict[str, Callable],
        settings: dict[str, Any] | None = None,
        **kwargs,
    ) -> ComponentBase:
        """Returns component from a component spec.

        Args:
            component: Component, ComponentFactory, string or dict.
            cells: dict of cells.
            settings: settings to override.
            kwargs: settings to override.

        """
        cells = set(cells.keys())

        settings = settings or {}
        kwargs = kwargs or {}
        kwargs.update(settings)

        if isinstance(component, ComponentBase):
            if kwargs:
                raise ValueError(f"Cannot apply kwargs {kwargs} to {component.name!r}")
            return component
        elif isinstance(component, kf.KCell):
            return Component.from_kcell(component)
        elif callable(component):
            return component(**kwargs)
        elif isinstance(component, str):
            if component not in cells:
                raise ValueError(
                    f"{component!r} not in PDK {self.name!r} cells: {cells} "
                )
            return self.cells[component](**kwargs)
        elif isinstance(component, dict | DictConfig):
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
                raise ValueError(
                    f"{cell_name!r} from PDK {self.name!r} not in cells: {cells} "
                )
            return self.cells[cell_name](**settings)
        else:
            raise ValueError(
                "get_component expects a ComponentSpec (Component, ComponentFactory, "
                f"string or dict), got {type(component)}"
            )

    def get_cross_section(
        self, cross_section: CrossSectionSpec, **kwargs
    ) -> CrossSection | Transition:
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
            return xs(**kwargs) if callable(xs) else xs.copy(**kwargs)
        elif isinstance(cross_section, dict | DictConfig):
            xs_name = cross_section.get("cross_section", None)
            settings = cross_section.get("settings", {})
            return self.get_cross_section(xs_name, **settings)
        elif isinstance(cross_section, CrossSection | Transition):
            if kwargs:
                warnings.warn(
                    f"{kwargs} are ignored for cross_section {cross_section.name!r}"
                )
            return cross_section
        else:
            raise ValueError(
                "get_cross_section expects a CrossSectionSpec (CrossSection, "
                f"CrossSectionFactory, Transition, string or dict), got {type(cross_section)}"
            )

    def get_layer(self, layer: LayerSpec) -> LayerEnum:
        """Returns layer from a layer spec."""
        if isinstance(layer, LayerEnum):
            return layer
        elif isinstance(layer, tuple | list):
            if len(layer) != 2:
                raise ValueError(f"{layer!r} needs two integer numbers.")
            return kf.kcl.layer(*layer)
        elif isinstance(layer, str):
            if not hasattr(self.layers, layer):
                raise ValueError(f"{layer!r} not in {self.layers}")
            return getattr(self.layers, layer)
        elif isinstance(layer, int):
            return layer
        elif layer is np.nan:
            return np.nan
        else:
            raise ValueError(
                f"{layer!r} needs to be a LayerSpec (string, int or (int, int) or LayerEnum), got {type(layer)}"
            )

    def get_layer_name(self, layer: LayerSpec) -> str:
        layer_index = self.get_layer(layer)
        return self.layers[layer_index]

    def get_layer_views(self) -> LayerViews:
        if self.layer_views is None:
            raise ValueError(f"layer_views for Pdk {self.name!r} is None")
        return self.layer_views

    def get_layer_stack(self) -> LayerStack:
        if self.layer_stack is None:
            raise ValueError(f"layer_stack for Pdk {self.name!r} is None")
        return self.layer_stack

    def get_constant(self, key: str) -> Any:
        if not isinstance(key, str):
            return key
        if key not in self.constants:
            constants = list(self.constants.keys())
            raise ValueError(f"{key!r} not in {constants}")
        return self.constants[key]

    def to_updk(self) -> str:
        """Export to uPDK YAML definition."""
        from gdsfactory.components.bbox import bbox_to_points

        blocks = {cell_name: cell() for cell_name, cell in self.cells.items()}
        blocks = {
            name: dict(
                bbox=bbox_to_points(c.dbbox()),
                doc=c.__doc__.split("\n")[0],
                settings=extract_args_from_docstring(c.__doc__),
                parameters={
                    sname: {
                        "value": svalue,
                        "type": str(svalue.__class__.__name__),
                        "doc": extract_args_from_docstring(c.__doc__)
                        .get(sname, {})
                        .get("doc", None),
                        "min": extract_args_from_docstring(c.__doc__)
                        .get(sname, {})
                        .get("min", 0),
                        "max": extract_args_from_docstring(c.__doc__)
                        .get(sname, {})
                        .get("max", 0),
                        "unit": extract_args_from_docstring(c.__doc__)
                        .get(sname, {})
                        .get("unit", None),
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
                            float(port.dcenter[0]),
                            float(port.dcenter[1]),
                            float(port.orientation),
                        ],
                        "alias": port.info.get("alias"),
                        "doc": port.info.get("doc"),
                    }
                    for port in c.ports
                },
            )
            for name, c in blocks.items()
        }
        xsections = {
            xs_name: self.get_cross_section(xs_name)
            for xs_name in self.cross_sections.keys()
        }
        xsections = {
            xs_name: dict(width=xsection.width)
            for xs_name, xsection in xsections.items()
        }

        header = dict(description=self.name)

        d = {"blocks": blocks, "xsections": xsections, "header": header}
        return omegaconf.OmegaConf.to_yaml(d)

    def get_cross_section_name(self, cross_section: CrossSection) -> str:
        xs_name = next(
            (
                key
                for key, value in self.cross_sections.items()
                if value == cross_section
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
                layer_map=self.layers,
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
    return _ACTIVE_PDK


def get_material_index(material: MaterialSpec, *args, **kwargs) -> Component:
    return get_active_pdk().get_material_index(material, *args, **kwargs)


def get_component(component: ComponentSpec, **kwargs) -> Component:
    return get_active_pdk().get_component(component, **kwargs)


def get_cell(cell: CellSpec, **kwargs) -> ComponentFactory:
    return get_active_pdk().get_cell(cell, **kwargs)


def get_cross_section(
    cross_section: CrossSectionSpec, **kwargs
) -> CrossSection | Transition:
    return get_active_pdk().get_cross_section(cross_section, **kwargs)


def get_layer(layer: LayerSpec) -> int:
    return get_active_pdk().get_layer(layer)


def get_layer_name(layer: LayerSpec) -> str:
    layer_index = get_layer(layer)
    return str(get_active_pdk().layers(layer_index))


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


def get_routing_strategies() -> dict[str, Callable]:
    """Gets a dictionary of named routing functions available to the PDK, if defined, or gdsfactory defaults otherwise."""
    from gdsfactory.routing.factories import (
        routing_strategy as default_routing_strategies,
    )

    routing_strategies = get_active_pdk().routing_strategies
    if routing_strategies is None:
        routing_strategies = default_routing_strategies
    return routing_strategies


if __name__ == "__main__":
    l1 = get_layer((1, 0))
    l2 = get_layer((3, 0))
    print(l1)
    print(l2)

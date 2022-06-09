import logging
import pathlib
import warnings
from functools import partial
from typing import Callable, Optional

import numpy as np
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from gdsfactory.components import cells
from gdsfactory.containers import containers as containers_default
from gdsfactory.cross_section import cross_sections
from gdsfactory.events import Event
from gdsfactory.read.from_yaml import from_yaml
from gdsfactory.show import show
from gdsfactory.tech import LAYER
from gdsfactory.types import (
    CellSpec,
    Component,
    ComponentFactory,
    ComponentSpec,
    CrossSection,
    CrossSectionFactory,
    CrossSectionSpec,
    Dict,
    Layer,
    LayerSpec,
    PathType,
)

logger = logging.root
component_settings = ["function", "component", "settings"]
cross_section_settings = ["function", "cross_section", "settings"]
layers_required = ["DEVREC", "PORT", "PORTE"]


class Pdk(BaseModel):
    """Pdk Library to store cell and cross_section functions.

    Attributes:
        name: PDK name.
        cross_sections: dict of cross_sections factories.
        cells: dict of parametric cells that return Components.
        layers: layers dict.
        containers: dict of pcells that contain other cells.
        base_pdk: a pdk to copy from and extend.
        default_decorator: decorate all cells, if not otherwise defined on the cell.
    """

    name: str
    cross_sections: Dict[str, CrossSectionFactory] = Field(default_factory=dict)
    cells: Dict[str, ComponentFactory] = Field(default_factory=dict)
    layers: Dict[str, Layer] = Field(default_factory=dict)
    containers: Dict[str, ComponentFactory] = containers_default
    base_pdk: Optional["Pdk"] = None
    default_decorator: Optional[Callable[[Component], None]] = None

    def validate_layers(self):
        for layer in layers_required:
            if layer not in self.layers:
                raise ValueError(
                    f"{layer!r} not in Pdk.layers {list(self.layers.keys())}"
                )

    def activate(self) -> None:
        """Set current pdk to as the active pdk."""
        if self.base_pdk:
            cross_sections = self.base_pdk.cross_sections
            cross_sections.update(self.cross_sections)
            self.cross_sections = cross_sections

            cells = self.base_pdk.cells
            cells.update(self.cells)
            self.cells.update(cells)

            containers = self.base_pdk.containers
            containers.update(self.containers)
            self.containers.update(containers)

            layers = self.base_pdk.layers
            layers.update(self.layers)
            self.layers.update(layers)

            if not self.default_decorator:
                self.default_decorator = self.base_pdk.default_decorator
        self.validate_layers()
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
            on_cell_registered.fire(name=name, cell=cell, pdk=self)

    def register_containers(self, **kwargs) -> None:
        """Register container factories."""
        for name, cell in kwargs.items():
            if not callable(cell):
                raise ValueError(
                    f"{cell} is not callable, make sure you register "
                    "cells functions that return a Component"
                )
            if name in self.containers:
                warnings.warn(f"Overwriting container {name!r}")

            self.containers[name] = cell
            on_container_registered.fire(name=name, cell=cell, pdk=self)

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
            on_cross_section_registered.fire(
                name=name, cross_section=cross_section, pdk=self
            )

    def register_cells_yaml(
        self,
        dirpath: Optional[PathType] = None,
        update: bool = False,
        **kwargs,
    ) -> None:
        """Load *.pic.yml YAML files and register them as cells.

        Args:
            dirpath: directory to recursive search for YAML cells.
            update: does not raise ValueError if cell already registered.

        Keyword Args:
            cell_name: cell function. To update cells dict.

        """
        message = "Updated" if update else "Registered"

        if dirpath:
            dirpath = pathlib.Path(dirpath)

            if not dirpath.is_dir():
                raise ValueError(f"{dirpath!r} needs to be a directory.")

            for filepath in dirpath.glob("*/**/*.pic.yml"):
                name = filepath.stem.split(".")[0]
                if not update and name in self.cells:
                    raise ValueError(
                        f"ERROR: Cell name {name!r} from {filepath} already registered."
                    )
                self.cells[name] = partial(from_yaml, filepath)
                on_yaml_cell_registered.fire(name=name, cell=self.cells[name], pdk=self)
                logger.info(f"{message} cell {name!r}")

        for k, v in kwargs.items():
            if not update and k in self.cells:
                raise ValueError(f"ERROR: Cell name {k!r} already registered.")
            self.cells[k] = v
            logger.info(f"{message} cell {k!r}")

    def remove_cell(self, name: str):
        if name not in self.cells:
            raise ValueError(f"{name!r} not in {list(self.cells.keys())}")
        self.cells.pop(name)
        logger.info(f"Removed cell {name!r}")

    def get_cell(self, cell: CellSpec, **kwargs) -> ComponentFactory:
        """Returns ComponentFactory from a cell spec."""
        cells_and_containers = set(self.cells.keys()).union(set(self.containers.keys()))

        if callable(cell):
            return cell
        elif isinstance(cell, str):
            if cell not in cells_and_containers:
                cells = list(self.cells.keys())
                containers = list(self.containers.keys())
                raise ValueError(
                    f"{cell!r} from PDK {self.name!r} not in cells: {cells} "
                    f"or containers: {containers}"
                )
            cell = self.cells[cell] if cell in self.cells else self.containers[cell]
            return cell
        elif isinstance(cell, (dict, DictConfig)):
            for key in cell.keys():
                if key not in component_settings:
                    raise ValueError(
                        f"Invalid setting {key!r} not in {component_settings}"
                    )
            settings = dict(cell.get("settings", {}))
            settings.update(**kwargs)

            cell_name = cell.get("function")
            if not isinstance(cell_name, str) or cell_name not in cells_and_containers:
                cells = list(self.cells.keys())
                containers = list(self.containers.keys())
                raise ValueError(
                    f"{cell_name!r} from PDK {self.name!r} not in cells: {cells} "
                    f"or containers: {containers}"
                )
            cell = (
                self.cells[cell_name]
                if cell_name in self.cells
                else self.containers[cell_name]
            )
            return partial(cell, **settings)
        else:
            raise ValueError(
                "get_cell expects a CellSpec (ComponentFactory, string or dict),"
                f"got {type(cell)}"
            )

    def get_component(self, component: ComponentSpec, **kwargs) -> Component:
        """Returns component from a component spec."""

        cells_and_containers = set(self.cells.keys()).union(set(self.containers.keys()))

        if isinstance(component, Component):
            if kwargs:
                raise ValueError(f"Cannot apply kwargs {kwargs} to {component.name!r}")
            return component
        elif callable(component):
            return component(**kwargs)
        elif isinstance(component, str):
            if component not in cells_and_containers:
                cells = list(self.cells.keys())
                containers = list(self.containers.keys())
                raise ValueError(
                    f"{component!r} not in PDK {self.name!r} cells: {cells} "
                    f"or containers: {containers}"
                )
            cell = (
                self.cells[component]
                if component in self.cells
                else self.containers[component]
            )
            return cell(**kwargs)
        elif isinstance(component, (dict, DictConfig)):
            for key in component.keys():
                if key not in component_settings:
                    raise ValueError(
                        f"Invalid setting {key!r} not in {component_settings}"
                    )
            settings = dict(component.get("settings", {}))
            settings.update(**kwargs)

            cell_name = component.get("component", None)
            cell_name = cell_name or component.get("function")
            if not isinstance(cell_name, str) or cell_name not in cells_and_containers:
                cells = list(self.cells.keys())
                containers = list(self.containers.keys())
                raise ValueError(
                    f"{cell_name!r} from PDK {self.name!r} not in cells: {cells} "
                    f"or containers: {containers}"
                )
            cell = (
                self.cells[cell_name]
                if cell_name in self.cells
                else self.containers[cell_name]
            )
            component = cell(**settings)
            component = (
                self.default_decorator(component) or component
                if self.default_decorator
                else component
            )
            return component
        else:
            raise ValueError(
                "get_component expects a ComponentSpec (Component, ComponentFactory, "
                f"string or dict), got {type(component)}"
            )

    def get_cross_section(
        self, cross_section: CrossSectionSpec, **kwargs
    ) -> CrossSection:
        """Returns cross_section from a cross_section spec."""
        if isinstance(cross_section, CrossSection):
            if kwargs:
                raise ValueError(f"Cannot apply {kwargs} to a defined CrossSection")
            return cross_section
        elif callable(cross_section):
            return cross_section(**kwargs)
        elif isinstance(cross_section, str):
            if cross_section not in self.cross_sections:
                cross_sections = list(self.cross_sections.keys())
                raise ValueError(f"{cross_section!r} not in {cross_sections}")
            cross_section_factory = self.cross_sections[cross_section]
            return cross_section_factory(**kwargs)
        elif isinstance(cross_section, (dict, DictConfig)):
            for key in cross_section.keys():
                if key not in cross_section_settings:
                    raise ValueError(
                        f"Invalid setting {key!r} not in {cross_section_settings}"
                    )
            cross_section_factory_name = cross_section.get("cross_section", None)
            cross_section_factory_name = (
                cross_section_factory_name or cross_section.get("function")
            )
            if (
                not isinstance(cross_section_factory_name, str)
                or cross_section_factory_name not in self.cross_sections
            ):
                cross_sections = list(self.cross_sections.keys())
                raise ValueError(
                    f"{cross_section_factory_name!r} not in {cross_sections}"
                )
            cross_section_factory = self.cross_sections[cross_section_factory_name]
            settings = dict(cross_section.get("settings", {}))
            settings.update(**kwargs)

            return cross_section_factory(**settings)
        else:
            raise ValueError(
                "get_cross_section expects a CrossSectionSpec (CrossSection, "
                f"CrossSectionFactory, string or dict), got {type(cross_section)}"
            )

    def get_layer(self, layer: LayerSpec) -> Layer:
        """Returns layer from a layer spec."""
        if isinstance(layer, (tuple, list)):
            if len(layer) != 2:
                raise ValueError(f"{layer!r} needs two integer numbers.")
            return layer
        elif isinstance(layer, int):
            return (layer, 0)
        elif isinstance(layer, str):
            if layer not in self.layers:
                raise ValueError(f"{layer!r} not in {self.layers.keys()}")
            return self.layers[layer]
        elif layer is np.nan:
            return np.nan
        elif layer is None:
            return
        else:
            raise ValueError(
                f"{layer!r} needs to be a LayerSpec (string, int or Layer)"
            )

    # _on_cell_registered = Event()
    # _on_container_registered: Event = Event()
    # _on_yaml_cell_registered: Event = Event()
    # _on_cross_section_registered: Event = Event()
    #
    # @property
    # def on_cell_registered(self) -> Event:
    #     return self._on_cell_registered
    #
    # @property
    # def on_container_registered(self) -> Event:
    #     return self._on_container_registered
    #
    # @property
    # def on_yaml_cell_registered(self) -> Event:
    #     return self._on_yaml_cell_registered
    #
    # @property
    # def on_cross_section_registered(self) -> Event:
    #     return self._on_cross_section_registered


GENERIC = Pdk(
    name="generic", cross_sections=cross_sections, cells=cells, layers=LAYER.dict()
)
_ACTIVE_PDK = GENERIC


def get_component(component: ComponentSpec, **kwargs) -> Component:
    return _ACTIVE_PDK.get_component(component, **kwargs)


def get_cell(cell: CellSpec, **kwargs) -> ComponentFactory:
    return _ACTIVE_PDK.get_cell(cell, **kwargs)


def get_cross_section(cross_section: CrossSectionSpec, **kwargs) -> CrossSection:
    return _ACTIVE_PDK.get_cross_section(cross_section, **kwargs)


def get_layer(layer: LayerSpec) -> Layer:
    return _ACTIVE_PDK.get_layer(layer)


def get_active_pdk() -> Pdk:
    return _ACTIVE_PDK


def _set_active_pdk(pdk: Pdk) -> None:
    global _ACTIVE_PDK
    old_pdk = _ACTIVE_PDK
    _ACTIVE_PDK = pdk
    on_pdk_activated.fire(old_pdk=old_pdk, new_pdk=pdk)


on_pdk_activated: Event = Event()
on_cell_registered: Event = Event()
on_container_registered: Event = Event()
on_yaml_cell_registered: Event = Event()
on_yaml_cell_modified: Event = Event()
on_cross_section_registered: Event = Event()

on_container_registered.add_handler(on_cell_registered.fire)
on_yaml_cell_registered.add_handler(on_cell_registered.fire)
on_yaml_cell_modified.add_handler(show)


if __name__ == "__main__":
    # c = _ACTIVE_PDK.get_component("straight")
    # print(c.settings)
    # on_pdk_activated += print
    # set_active_pdk(GENERIC)
    c = Pdk(
        name="demo",
        cells=cells,
        cross_sections=cross_sections,
        # layers=dict(DEVREC=(3, 0), PORTE=(3, 5)),
    )
    print(c.layers)

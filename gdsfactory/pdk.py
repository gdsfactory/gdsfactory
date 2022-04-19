import warnings
from functools import partial

from omegaconf import DictConfig
from pydantic import BaseModel

from gdsfactory.components import cells
from gdsfactory.cross_section import cross_sections
from gdsfactory.types import (
    CellSpec,
    Component,
    ComponentFactory,
    ComponentSpec,
    CrossSection,
    CrossSectionFactory,
    CrossSectionSpec,
    Dict,
)


class Pdk(BaseModel):
    """Pdk Library to store cell and cross_section functions."""

    name: str
    cross_sections: Dict[str, CrossSectionFactory]
    cells: Dict[str, ComponentFactory]

    def activate(self):
        set_active_pdk(self)

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
            self.cross_section[name] = cross_section

    def load_yaml(self):
        """Load *.pic.yml YAML files and register them as cells."""
        pass

    def get_cell(self, cell: CellSpec, **kwargs) -> ComponentFactory:
        """Returns ComponentFactory from a cell spec."""
        if callable(cell):
            return cell
        elif isinstance(cell, str):
            if cell not in self.cells:
                cells = list(self.cells.keys())
                raise ValueError(f"{cell!r} not in {cells}")
            cell = self.cells[cell]
            return cell
        elif isinstance(cell, (dict, DictConfig)):
            for key in cell.keys():
                if key not in ["function", "component", "settings"]:
                    raise ValueError(
                        f"Invalid setting {key!r} not in (component, function, settings)"
                    )
            settings = dict(cell.get("settings", {}))
            settings.update(**kwargs)

            cell_name = cell.get("function")
            if not isinstance(cell_name, str) or cell_name not in self.cells:
                cells = list(self.cells.keys())
                raise ValueError(f"{cell_name!r} not in {cells}")
            cell = self.cells[cell_name]
            return partial(cell, **settings)
        else:
            raise ValueError(
                "get_cell expects a CellSpec (ComponentFactory, string or dict),"
                f"got {type(cell)}"
            )

    def get_component(self, component: ComponentSpec, **kwargs) -> Component:
        """Returns component from a component spec."""
        if isinstance(component, Component):
            if kwargs:
                raise ValueError(f"Cannot apply kwargs {kwargs} to {component.name!r}")
            return component
        elif callable(component):
            return component(**kwargs)
        elif isinstance(component, str):
            if component not in self.cells:
                cells = list(self.cells.keys())
                raise ValueError(f"{component!r} not in {cells}")
            cell = self.cells[component]
            return cell(**kwargs)
        elif isinstance(component, (dict, DictConfig)):
            for key in component.keys():
                if key not in ["function", "component", "settings"]:
                    raise ValueError(
                        f"Invalid setting {key!r} not in (component, function, settings)"
                    )
            settings = dict(component.get("settings", {}))
            settings.update(**kwargs)

            cell_name = component.get("component", None)
            cell_name = cell_name or component.get("function")
            if not isinstance(cell_name, str) or cell_name not in self.cells:
                cells = list(self.cells.keys())
                raise ValueError(f"{cell_name!r} not in {cells}")
            cell = self.cells[cell_name]
            return cell(**settings)
        else:
            raise ValueError(
                "get_component expects a ComponentSpec (Component, ComponentFactory, string or dict),"
                f"got {type(component)}"
            )

    def get_cross_section(
        self, cross_section: CrossSectionSpec, **kwargs
    ) -> CrossSection:
        """Returns component from a cross_section spec."""
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
                if key not in ["function", "cross_section", "settings"]:
                    raise ValueError(
                        f"Invalid setting {key!r} not in (cross_section, function, settings)"
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
                f"get_cross_section expects a CrossSectionSpec (CrossSection, CrossSectionFactory, string or dict), got {type(cross_section)}"
            )


_ACTIVE_PDK = Pdk(name="generic", cross_sections=cross_sections, cells=cells)


def get_component(component: ComponentSpec, **kwargs) -> Component:
    return _ACTIVE_PDK.get_component(component, **kwargs)


def get_cell(cell: CellSpec, **kwargs) -> ComponentFactory:
    return _ACTIVE_PDK.get_cell(cell, **kwargs)


def get_cross_section(cross_section: CrossSectionSpec, **kwargs) -> CrossSection:
    return _ACTIVE_PDK.get_cross_section(cross_section, **kwargs)


def get_active_pdk() -> Pdk:
    return _ACTIVE_PDK


def set_active_pdk(pdk: Pdk):
    global _ACTIVE_PDK
    _ACTIVE_PDK = pdk


if __name__ == "__main__":
    c = _ACTIVE_PDK.get_component("straight")
    print(c.settings)

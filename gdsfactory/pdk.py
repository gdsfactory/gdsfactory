import warnings

from pydantic import BaseModel

from gdsfactory.components import cells
from gdsfactory.cross_section import cross_sections
from gdsfactory.types import (
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
        global ACTIVE_PDK
        ACTIVE_PDK = self

    def register_cell(self, name: str, function: ComponentFactory) -> None:
        if name in cells:
            warnings.warn(f"Overwriting cell {name!r}")
        self.cells[name] = function

    def load(self):
        """find pdk.yml register all YAML components into cells.
        TODO:
        """
        pass

    def get_component(self, component: ComponentSpec, **kwargs) -> Component:
        if isinstance(component, Component):
            if kwargs:
                warnings.warn(f"Cannot apply kwargs {kwargs} to {component.name!r}")
            return component
        elif callable(component):
            return component(**kwargs)
        elif isinstance(component, str):
            cell = self.cells[component]
            return cell(**kwargs)
        elif isinstance(component, dict):
            component = component.get("component")
            if not isinstance(component, str) or component not in self.cells:
                components = list(self.components.keys())
                raise ValueError(f"{component!r} {type(component)} not in {components}")
            cell = self.cells[component]
            settings = dict(component.get("settings", {}))
            settings.update(**kwargs)
            return cell(**settings)
        else:
            raise ValueError(
                f"get_component expects a ComponentSpec (Component, ComponentFactory, string or dict), got {type(component)}"
            )

    def get_cross_section(
        self, cross_section: CrossSectionSpec, **kwargs
    ) -> CrossSection:
        if isinstance(cross_section, CrossSection):
            if kwargs:
                warnings.warn(f"Cannot apply {kwargs} to a defined CrossSection")
            return cross_section
        elif callable(cross_section):
            return cross_section(**kwargs)
        elif isinstance(cross_section, str):
            cross_section_factory = self.cross_sections[cross_section]
            return cross_section_factory(**kwargs)
        elif isinstance(cross_section, dict):
            cross_section_factory_name = cross_section.get("cross_section")
            if (
                not isinstance(cross_section_factory_name, str)
                or cross_section_factory_name not in self.cross_sections
            ):
                cross_sections = list(self.cross_sections.keys())
                raise ValueError(
                    f"{cross_section_factory_name!r} {type(cross_section_factory_name)} "
                    f"not in {cross_sections}"
                )
            cross_section_factory = self.cross_sections[cross_section_factory_name]
            settings = dict(cross_section.get("settings", {}))
            settings.update(**kwargs)
            return cross_section_factory(**settings)
        else:
            raise ValueError(
                f"get_cross_section expects a CrossSectionSpec (CrossSection, CrossSectionFactory, string or dict), got {type(cross_section)}"
            )


ACTIVE_PDK = Pdk(name="generic", cross_sections=cross_sections, cells=cells)


def get_component(component: ComponentSpec, **kwargs) -> Component:
    return ACTIVE_PDK.get_component(component, **kwargs)


def get_cross_section(cross_section: CrossSectionSpec, **kwargs) -> CrossSection:
    return ACTIVE_PDK.get_cross_section(cross_section, **kwargs)

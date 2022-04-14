from pydantic import BaseModel

from gdsfactory.types import (
    Component,
    ComponentFactory,
    ComponentSpec,
    CrossSection,
    CrossSectionFactory,
    Dict,
)

ACTIVE_PDK = None


class Pdk(BaseModel):
    """Pdk Library to store cell functions and CrosSection factories."""

    cross_sections: Dict[str, CrossSectionFactory]
    cells: Dict[str, ComponentFactory]

    def activate(self):
        global ACTIVE_PDK
        ACTIVE_PDK = self

    def load(self):
        """find pdk.yml
        register all YAML components into cells.
        """
        pass

    def get_component(self, component: ComponentSpec, **kwargs) -> Component:
        if isinstance(component, Component):
            if kwargs:
                raise ValueError(f"Cannot apply kwargs {kwargs} to {component.name!r}")
            return component
        elif callable(component):
            return component(**kwargs)
        elif isinstance(component, str):
            cell = self.cells[component]
            return cell(**kwargs)
        else:
            component = component.get("component")
            cell = self.cells[component]
            settings = dict(component.get("settings", {}))
            settings.update(**kwargs)
            return cell(**settings)

    def get_cross_section(self) -> CrossSection:
        pass


def get_component(component: ComponentSpec, **kwargs) -> Component:
    return ACTIVE_PDK.get_component(component, **kwargs)

""" Write component GDS + metadata
write_component_type: try to load a component from library or creates if it does not exist
write_component: write component and metadata
"""

from typing import Optional
import pathlib
from pathlib import PosixPath
import json
from phidl import device_layout as pd

from pp.config import CONFIG, conf
from pp.cell import get_component_name
from pp.components import component_factory
from pp import klive
from pp.component import Component
from pp.cell import clear_cache

from pp.layers import LAYER


def get_component_type(component_type, component_factory=component_factory, **kwargs):
    """ returns a component from the factory """
    component_name = get_component_name(component_type, **kwargs)
    return component_factory[component_type](name=component_name, **kwargs)


def write_component_type(
    component_type,
    overwrite=True,
    path_directory=CONFIG["gds_directory"],
    factory=component_factory,
    **kwargs,
):
    """write_component by type or function

    Args:
        component_type: can be function or factory name
        overwrite: if False and component exists
        path_directory: to store GDS + metadata
        component_factory: factory dictionary
        **kwargs: component args
    """
    if callable(component_type):
        component_type = component_type.__name__

    assert type(component_type) == str

    component_name = get_component_name(component_type, **kwargs)
    gdspath = path_directory / (component_name + ".gds")
    path_directory.mkdir(parents=True, exist_ok=True)

    if not gdspath.exists() or overwrite:
        component = factory[component_type](name=component_name, **kwargs)
        write_component(component, gdspath)

    return gdspath


def write_component_report(component, json_path=None):
    """write component GDS and metadata:

    Args:
        component:
        json_path
    """

    json_path = json_path or CONFIG["gds_directory"] / f"{component.name}.json"
    ports_path = json_path.with_suffix(".ports")

    if len(component.ports) > 0:
        with open(ports_path, "w") as fw:
            for port in component.ports.values():
                fw.write(
                    f"{port.name}, {port.x:.3f}, {port.y:.3f}, {int(port.orientation)}, {port.width:.3f}, {port.layer}\n"
                )

    with open(json_path, "w+") as fw:
        fw.write(json.dumps(component.get_json(), indent=2))
    return json_path


def write_component(
    component: Component,
    gdspath: Optional[PosixPath] = None,
    path_library: PosixPath = CONFIG["gds_directory"],
    precision: float = 1e-9,
    with_settings_label: bool = conf.tech.with_settings_label,
) -> str:
    """write component GDS and metadata:

    - gds
    - ports
    - properties

    Args:
        component:
        gdspath:
        path_library
        precision: to save GDS points
    """

    gdspath = gdspath or path_library / (component.name + ".gds")
    gdspath = pathlib.Path(gdspath)
    ports_path = gdspath.with_suffix(".ports")
    json_path = gdspath.with_suffix(".json")

    """ write GDS """
    gdspath = write_gds(
        component=component,
        gdspath=str(gdspath),
        precision=precision,
        with_settings_label=with_settings_label,
    )

    """ write .ports in CSV"""
    if len(component.ports) > 0:
        with open(ports_path, "w") as fw:
            for port in component.ports.values():
                layer, purpose = pd._parse_layer(port.layer)
                fw.write(
                    f"{port.name}, {port.x:.3f}, {port.y:.3f}, {int(port.orientation)}, {port.width:.3f}, {layer}, {purpose}\n"
                )

    """ write JSON """
    with open(json_path, "w+") as fw:
        fw.write(json.dumps(component.get_json(), indent=2))
    return gdspath


def write_json(json_path, **settings):
    """ write properties dict into a json_path file"""

    with open(json_path, "w+") as fw:
        fw.write(json.dumps(settings))


def write_gds(
    component: Component,
    gdspath: Optional[PosixPath] = None,
    unit: float = 1e-6,
    precision: float = 1e-9,
    remove_previous_markers: bool = False,
    auto_rename: bool = False,
    with_settings_label: bool = conf.tech.with_settings_label,
    label_layer=LAYER.LABEL,
) -> str:
    """Write component to GDS and returs gdspath

    Args:
        component: gdsfactory Component.
        gdspath: GDS file path to write to.
        unit unit size for objects in library.
        precision: for the dimensions of the objects in the library (m).
        remove_previous_markers: clear previous ones to avoid duplicates.
        auto_rename: If True, fixes any duplicate cell names.
        with_settings_label: write component settings into a label.

    Returns:
        gdspath
    """

    gdspath = gdspath or CONFIG["gds_directory"] / (component.name + ".gds")
    gdspath = pathlib.Path(gdspath)
    gdsdir = gdspath.parent
    gdspath = str(gdspath)
    gdsdir.mkdir(parents=True, exist_ok=True)

    if remove_previous_markers:
        component.remove_layers([LAYER.PORT])
        component.remove_layers([LAYER.TEXT])

    # write component settings into a label
    if with_settings_label:
        settings = component.get_settings()
        settings_string = f"settings={json.dumps(settings)}"
        component.add_label(
            position=component.center, text=settings_string, layer=label_layer
        )

    component.write_gds(
        gdspath, unit=unit, precision=precision, auto_rename=auto_rename,
    )
    component.path = gdspath
    return gdspath


def clean_value(value):
    """Returns JSON serializable value."""
    if isinstance(value, Component):
        value = value.name
    elif callable(value):
        value = value.__name__
    elif isinstance(value, dict):
        value = {k: clean_value(v) for k, v in value.items()}
    elif hasattr(value, "__iter__"):
        value = [clean_value(v) for v in value]
    return value


def show(
    component: Component, gdspath: PosixPath = CONFIG["gdspath"], **kwargs
) -> None:
    """write component GDS and shows it in klayout

    Args:
        component
        gdspath: where to save the gds
    """
    if isinstance(component, pathlib.Path):
        component = str(component)
        return klive.show(component)
    elif isinstance(component, str):
        return klive.show(component)
    elif hasattr(component, "path"):
        return klive.show(component.path)
    elif component is None:
        raise ValueError(
            "Component is None, make sure that your function returns the component"
        )

    elif isinstance(component, Component):
        write_gds(
            component, gdspath, **kwargs,
        )
        klive.show(gdspath)
    else:
        raise ValueError(
            f"Component is {type(component)}, make sure pass a Component or a path"
        )
    clear_cache()


if __name__ == "__main__":
    import pp

    # c = pp.c.waveguide(length=1.0016)  # rounds to 1.002 with 1nm precision
    # c = pp.c.waveguide(length=1.006)  # rounds to 1.005 with 5nm precision

    # c = pp.c.waveguide(length=1.009)  # rounds to 1.010 with 5nm precision
    # cc = pp.routing.add_fiber_array(c)
    # pp.write_component(cc, precision=5e-9)
    # pp.show(cc)

    c = pp.c.waveguide(length=1.009, pins=True)
    pp.write_component(c, gdspath="wg.gds", with_settings_label=True)
    pp.show(c)

    # print(c.settings)
    # gdspath = pp.write_component(c, precision=5e-9)
    # pp.show(gdspath)

    # cc = pp.routing.add_fiber_array(c)
    # gdspath = write_component(cc)

    # gdspath = write_component_type("ring_double_bus", overwrite=True, flatten=False)
    # gdspath = write_component_type("waveguide", length=5, overwrite=True)
    # gdspath = write_component_type("mmi1x2", width_mmi=5, overwrite=True)
    # gdspath = write_component_type("mzi2x2", overwrite=True)
    # gdspath = write_component_type("bend_circular", overwrite=True)
    # print(gdspath)
    # print(type(gdspath))
    # klive.show(gdspath)

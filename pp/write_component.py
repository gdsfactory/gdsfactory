"""Write component GDS, metadata in JSON and ports in CSV

"""

import json
import pathlib
import tempfile
from pathlib import Path
from typing import Optional

from phidl import device_layout as pd

from pp import klive
from pp.cell import clear_cache
from pp.component import Component
from pp.config import CONFIG

tmp = pathlib.Path(tempfile.TemporaryDirectory().name).parent / "gdsfactory"
tmp.mkdir(exist_ok=True)


def write_component_report(component: Component, json_path=None) -> Path:
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
    gdspath: Optional[Path] = None,
    gdsdir: Path = tmp,
    precision: float = 1e-9,
) -> Path:
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

    gdspath = gdspath or gdsdir / (component.name + ".gds")
    gdspath = pathlib.Path(gdspath)
    ports_path = gdspath.with_suffix(".ports")
    json_path = gdspath.with_suffix(".json")

    gdspath = write_gds(component=component, gdspath=str(gdspath), precision=precision,)

    # component.ports CSV
    if len(component.ports) > 0:
        with open(ports_path, "w") as fw:
            for port in component.ports.values():
                layer, purpose = pd._parse_layer(port.layer)
                fw.write(
                    f"{port.name}, {port.x:.3f}, {port.y:.3f}, {int(port.orientation)}, {port.width:.3f}, {layer}, {purpose}\n"
                )

    # component.json metadata dict
    with open(json_path, "w+") as fw:
        fw.write(json.dumps(component.get_json(), indent=2))
    return gdspath


def write_json(json_path, **settings):
    """ write properties dict into a json_path file"""

    with open(json_path, "w+") as fw:
        fw.write(json.dumps(settings))


def write_gds(
    component: Component,
    gdspath: Optional[Path] = None,
    gdsdir: Path = tmp,
    unit: float = 1e-6,
    precision: float = 1e-9,
    auto_rename: bool = False,
) -> Path:
    """Write component to GDS and returs gdspath

    Args:
        component: gdsfactory Component.
        gdspath: GDS file path to write to.
        unit unit size for objects in library.
        precision: for the dimensions of the objects in the library (m).
        remove_previous_markers: clear previous ones to avoid duplicates.
        auto_rename: If True, fixes any duplicate cell names.

    Returns:
        gdspath
    """

    gdsdir = pathlib.Path(gdsdir)
    gdspath = gdspath or gdsdir / (component.name + ".gds")
    gdspath = pathlib.Path(gdspath)
    gdsdir = gdspath.parent
    gdsdir.mkdir(exist_ok=True, parents=True)

    component.write_gds(
        str(gdspath), unit=unit, precision=precision, auto_rename=auto_rename,
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


def show(component: Component, **kwargs) -> None:
    """write component GDS and shows it in klayout

    Args:
        component
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
        gdspath = write_gds(component, **kwargs)
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

    c = pp.c.waveguide()
    c = pp.add_pins(c)
    gdspath = pp.CONFIG["gdsdir"] / "waveguide_with_pins.gds"
    pp.write_component(c, gdspath=gdspath)
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

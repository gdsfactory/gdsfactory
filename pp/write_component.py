""" Write component GDS + metadata
write_component_type: try to load a component from library or creates if it does not exist
write_component: write component and metadata
"""

import os
import pathlib
import json
from phidl import device_layout as pd

from pp import CONFIG
from pp.name import get_component_name
from pp.components import component_type2factory
from pp.ports import add_port_markers
from pp import klive

from pp.layers import LAYER


def get_component_type(
    component_type, component_type2factory=component_type2factory, **kwargs
):
    """ returns a component from the factory """
    component_name = get_component_name(component_type, **kwargs)
    return component_type2factory[component_type](name=component_name, **kwargs)


def write_component_type(
    component_type,
    overwrite=False,
    path_directory=CONFIG["gds_directory"],
    component_type2factory=component_type2factory,
    add_port_pins=True,
    flatten=False,
    **kwargs,
):
    """ write_component by type or function

    Args:
        component_type: can be function or factory name
        overwrite:
        path_directory: to store GDS + metadata
        component_type2factory: factory dictionary
        flatten: False
        **kwargs: component args
    """
    if callable(component_type):
        component_type = component_type.__name__

    assert type(component_type) == str

    if not os.path.isdir(path_directory):
        os.makedirs(path_directory)

    component_name = get_component_name(component_type, **kwargs)
    gdspath = path_directory / (component_name + ".gds")

    if not os.path.isfile(gdspath) or overwrite:
        component = component_type2factory[component_type](
            name=component_name, **kwargs
        )
        component.type = component_type
        if flatten:
            component.flatten()
        write_component(component, gdspath, add_port_pins=add_port_pins)

    return gdspath


def write_component_report(component, json_path=None):
    """ write component GDS and metadata:

    Args:
        component:
        json_path
    """

    if json_path is None:
        json_path = CONFIG["gds_directory"] / component.name + ".json"

    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path))

    ports_path = json_path[:-5] + ".ports"

    """ write ports """
    if len(component.ports) > 0:
        with open(ports_path, "w") as fw:
            for _, port in component.ports.items():
                fw.write(
                    "{}, {:.3f}, {:.3f}, {}, {:.3f}, {}\n".format(
                        port.name,
                        port.x,
                        port.y,
                        int(port.orientation),
                        port.width,
                        port.layer,
                    )
                )

    """ write json """
    with open(json_path, "w+") as fw:
        fw.write(json.dumps(component.get_json(), indent=2))
    return json_path


def write_component(
    component,
    gdspath=None,
    path_library=CONFIG["gds_directory"],
    add_port_pins=True,
    add_ports_to_all_cells=False,
    store_hash_geometry=False,
    with_component_label=False,
    precision=1e-9,
    settings=None,
):
    """ write component GDS and metadata:

    - gds
    - ports
    - properties

    Args:
        component:
        gdspath:
        path_library
        add_port_pins: adds port metadata
        add_ports_to_all_cells: make sure that all sub-cells have port (necessary for netlist extraction)
        store_hash_geometry:
        with_component_label: adds a label to component
        precision: to save GDS points
        settings: dict of settings
    """

    gdspath = gdspath or path_library / (component.name + ".gds")
    gdspath = pathlib.Path(gdspath)
    path_library.mkdir(exist_ok=True)
    ports_path = gdspath.with_suffix(".ports")
    json_path = gdspath.with_suffix(".json")

    """ write GDS """
    gdspath = write_gds(
        component=component,
        gdspath=str(gdspath),
        add_port_pins=add_port_pins,
        add_ports_to_all_cells=add_ports_to_all_cells,
        store_hash_geometry=store_hash_geometry,
        with_component_label=with_component_label,
        precision=precision,
    )

    """ write .ports in CSV"""
    if len(component.ports) > 0:
        with open(ports_path, "w") as fw:
            for _, port in component.ports.items():
                layer, texttype = pd._parse_layer(port.layer)

                fw.write(
                    "{}, {:.3f}, {:.3f}, {}, {:.3f}, {}, {}\n".format(
                        port.name,
                        port.x,
                        port.y,
                        int(port.orientation),
                        port.width,
                        layer,
                        texttype,
                    )
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
    component,
    gdspath=None,
    add_ports_to_all_cells=False,
    add_port_pins=True,
    store_hash_geometry=False,
    with_component_label=False,
    unit=1e-6,
    precision=1e-9,
    remove_previous_markers=False,
    auto_rename=False,
):
    """ write component to GDS and returs gdspath

    Args:
        component (required)
        gdspath: by default saves it into CONFIG['gds_directory']
        add_ports_to_all_cells: to child cells - required to export netlists
        add_port_pins: show port metadata
        auto_rename: False by default (otherwise it calls it top_cell)
        with_component_label
        unit
        precission

    Returns:
        gdspath
    """

    gdspath = gdspath or CONFIG["gds_directory"] / (component.name + ".gds")
    gdspath = str(gdspath)

    if remove_previous_markers:
        # If the component HAS ports AND markers and we want to
        # avoid duplicate port markers, then we clear the previous ones
        port_layer = (LAYER.PORT,)
        label_layer = (LAYER.TEXT,)
        component.remove_layers([port_layer])
        component.remove_layers([label_layer])

    if add_port_pins and add_ports_to_all_cells:
        referenced_cells = list(component.get_dependencies(recursive=True))
        all_cells = [component] + referenced_cells
        all_cells = list(set(all_cells))
        for c in all_cells:
            add_port_markers(c)
    elif add_port_pins:
        add_port_markers(component)

    folder = os.path.split(gdspath)[0]
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)

    if store_hash_geometry:
        # Remove any label on hash layer
        old_label = [l for l in component.labels if l.layer == LAYER.INFO_GEO_HASH]
        if len(old_label) > 0:
            for l in old_label:
                component.labels.remove(l)

        # Add new hash label
        # component.label(
        # text=component.hash_geometry(),
        # position=component.size_info.cc,
        # layer=LAYER.INFO_GEO_HASH,
        # )
        # Add new hash label
        if not hasattr(component, "settings"):
            component.settings = {}
        component.settings.update(dict(hash_geometry=component.hash_geometry()))

    # write component settings into text layer
    if with_component_label:
        for i, (k, v) in enumerate(component.settings.items()):
            component.label(
                text=f"{k}={v}",
                position=component.center + [0, i * 0.4],
                layer=LAYER.TEXT,
            )

    component.write_gds(gdspath, precision=precision, auto_rename=auto_rename)
    component.path = gdspath
    return gdspath


def show(
    component,
    gdspath=CONFIG["gdspath"],
    add_ports_to_all_cells=False,
    add_port_pins=True,
    **kwargs,
):
    """ write component GDS and shows it in klayout

    Args:
        component
        gdspath: where to save the gds
    """
    if isinstance(component, pathlib.Path):
        component = str(component)
    elif isinstance(component, str):
        return klive.show(component)
    elif hasattr(component, "path"):
        return klive.show(component.path)
    elif component is None:
        raise ValueError(
            "Component is None, make sure that your function returns the component"
        )

    else:
        write_gds(
            component,
            gdspath,
            add_ports_to_all_cells=add_ports_to_all_cells,
            add_port_pins=add_port_pins,
            **kwargs,
        )
        klive.show(gdspath)


if __name__ == "__main__":
    import pp

    # c = pp.c.waveguide(length=1.0016)  # rounds to 1.002 with 1nm precision
    # c = pp.c.waveguide(length=1.006)  # rounds to 1.005 with 5nm precision

    c = pp.c.waveguide(length=1.009)  # rounds to 1.010 with 5nm precision
    cc = pp.routing.add_io_optical(c)
    pp.write_component(cc, precision=5e-9)
    pp.show(cc)

    print(c.settings)

    # gdspath = pp.write_component(c, precision=5e-9)
    # pp.show(gdspath)

    # cc = pp.routing.add_io_optical(c)
    # gdspath = write_component(cc)

    # gdspath = write_component_type("ring_double_bus", overwrite=True, flatten=False)
    # gdspath = write_component_type("waveguide", length=5, overwrite=True, add_port_pins=False)
    # gdspath = write_component_type("mmi1x2", width_mmi=5, overwrite=True)
    # gdspath = write_component_type("mzi2x2", overwrite=True)
    # gdspath = write_component_type("bend_circular", overwrite=True)
    # print(gdspath)
    # print(type(gdspath))
    # klive.show(gdspath)

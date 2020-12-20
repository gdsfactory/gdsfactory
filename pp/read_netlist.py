import json
from typing import Union

from pp.component import Component
from pp.layers import LAYER
from pp.port import read_port_markers


def read_netlist(
    component_or_gdspath: Union[str, Component],
    outline_layer=LAYER.DEVREC,
    label_layer=LAYER.LABEL,
):
    """From a Component or GDS returns netlist.
    FIXME: need to detect connections.
    """
    component = (
        component_or_gdspath
        if isinstance(component_or_gdspath, Component)
        else pp.import_gds(component_or_gdspath)
    )
    markers = read_port_markers(component, [outline_layer])

    instances = {}
    placements = {}
    connections = {}

    xy2settings = {
        (int(label.x), int(label.y)): json.loads(label.text[9:])
        for label in component.get_labels()
        if label.layer == label_layer[0]
    }

    for p in markers.polygons:
        x = int(p.x)
        y = int(p.y)
        d = xy2settings[(x, y)]
        instance_name = f"{d['name']}_{x}_{y}"
        instances[instance_name] = {}
        instances[instance_name]["component"] = d["function_name"]
        instances[instance_name]["settings"] = d["settings"]

        placements[instance_name] = {}
        placements[instance_name]["x"] = x
        placements[instance_name]["y"] = y

    return instances, placements, connections


if __name__ == "__main__":
    import pp

    c = pp.c.mzi()
    component = c
    label_layer = LAYER.LABEL

    xy2settings = {
        (int(label.x), int(label.y)): json.loads(label.text[9:])
        for label in component.get_labels()
        if label.layer == label_layer[0]
    }
    instances, placements, connections = read_netlist(c)

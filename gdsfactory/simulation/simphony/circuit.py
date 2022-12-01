from __future__ import annotations

from typing import Dict

from simphony import Model
from simphony.layout import Circuit

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.simulation.simphony.components import model_factory
from gdsfactory.simulation.simphony.types import ModelFactory


def connect_pins(connections, model_names, components, circuit: Circuit):
    for k, v in connections.items():
        model1_name, port1_name = k.split(",")
        model2_name, port2_name = v.split(",")

        if model1_name in model_names and model2_name in model_names:
            index1 = [i for i, c in enumerate(components) if c.name == model1_name]
            p1_name = [
                i
                for c in components
                if c.name == model1_name
                for i, p in enumerate(c.pins)
                if p.name == port1_name
            ]
            index2 = [i for i, c in enumerate(components) if c.name == model2_name]
            p2_name = [
                i
                for c in components
                if c.name == model2_name
                for i, p in enumerate(c.pins)
                if p.name == port2_name
            ]

            circuit._get_components()[index1[0]][p1_name[0]].connect(
                circuit._get_components()[index2[0]][p2_name[0]]
            )
    return circuit


def rename_pins(circuit, components):
    i = 0
    for c in components:
        for p in c.pins:
            c[p.name].rename(f"pin{i}")
            i += 1
    return circuit


def component_to_circuit(
    component: Component,
    model_factory: Dict[str, ModelFactory] = model_factory,
) -> Circuit:
    """Returns Simphony circuit from a gdsfactory component netlist.

    Args:
        component: component factory or instance.
        model_factory: dict of component_type.

    """
    netlist = component.get_netlist()
    instances = netlist["instances"]
    connections = netlist["connections"]
    model_names = []
    component_models = list(model_factory.keys())

    a = 0
    for name, metadata in instances.items():
        component_type = metadata["component"]
        component_settings = metadata.get("settings", {})

        if component_type is None:
            raise ValueError(f"instance {name!r} has no component_type")
            # continue

        if component_type not in model_factory:
            raise ValueError(
                f"Model for {component_type!r} not found in {component_models}"
            )
        model_function = model_factory[component_type]
        model = model_function(**component_settings)
        if not isinstance(model, Model):
            raise ValueError(f"model {model!r} is not a simphony Model")
        model_names.append(name)
        model.name = name

        if a == 0:
            circuit = Circuit(model)
            a += 1
        else:
            circuit._add(model)

    components = circuit._get_components()

    circuit = connect_pins(connections, model_names, components, circuit)

    for i, pin in enumerate(circuit.pins, start=1):
        pin.rename(f"o{i}")
    return circuit


if __name__ == "__main__":
    import gdsfactory.simulation.simphony as gs

    c = gf.components.mzi()
    n = c.get_netlist()

    cm = component_to_circuit(c)
    gs.plot_circuit(cm, pin_in=cm.pins[0].name, pins_out=[cm.pins[-1].name])

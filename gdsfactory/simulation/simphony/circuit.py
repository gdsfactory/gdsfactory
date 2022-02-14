from typing import Dict

from omegaconf import OmegaConf
from simphony.elements import Model
from simphony.netlist import Subcircuit

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.simulation.simphony.components import model_factory
from gdsfactory.simulation.simphony.types import ModelFactory


def component_to_circuit(
    component: Component,
    model_factory: Dict[str, ModelFactory] = model_factory,
) -> Subcircuit:
    """Returns Simphony circuit from a gdsfactory component netlist.

    Args:
        component: component factory or instance
        model_factory: dict of component_type
    """
    netlist = component.get_netlist()
    instances = netlist["instances"]
    connections = netlist["connections"]

    circuit = Subcircuit(component.name)
    model_names = []
    model_name_tuple = []
    component_models = list(model_factory.keys())

    for name, metadata in instances.items():
        component_type = metadata.function_name

        if component_type is None:
            continue

        if component_type not in model_factory:
            raise ValueError(
                f"Model for {component_type!r} not found in {component_models}"
            )
        component_settings = OmegaConf.to_container(metadata.full)
        model_function = model_factory[component_type]
        model = model_function(**component_settings)
        assert isinstance(model, Model), f"model {model!r} is not a simphony Model"
        model_names.append(name)
        model_name_tuple.append((model, name))

    circuit.add(model_name_tuple)

    for k, v in connections.items():
        model1_name, port1_name = k.split(",")
        model2_name, port2_name = v.split(",")

        if model1_name in model_names and model2_name in model_names:
            circuit.connect(model1_name, port1_name, model2_name, port2_name)

    circuit.info = netlist
    return circuit


if __name__ == "__main__":
    import gdsfactory.simulation.simphony as gs

    c = gf.components.mzi(delta_length=100)
    cm = component_to_circuit(c)
    p2 = cm.pins.pop()
    p2.name = "o2"
    gs.plot_circuit(cm)

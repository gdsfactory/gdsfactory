"""gdsfactory simphony circuit simulation plugin"""

try:
    from simphony.tools import freq2wl, wl2freq
except ImportError:
    print("To install simphony plugin make sure you `pip install gdsfactory[full]`")

import gdsfactory.simulation.simphony.components as components
from gdsfactory.simulation.simphony.add_gc import add_gc
from gdsfactory.simulation.simphony.circuit import component_to_circuit
from gdsfactory.simulation.simphony.components import model_factory
from gdsfactory.simulation.simphony.model_from_gdsfactory import model_from_gdsfactory
from gdsfactory.simulation.simphony.model_from_sparameters import (
    model_from_filepath,
    model_from_sparameters,
)
from gdsfactory.simulation.simphony.plot_circuit import plot_circuit
from gdsfactory.simulation.simphony.plot_circuit_montecarlo import (
    plot_circuit_montecarlo,
)
from gdsfactory.simulation.simphony.plot_model import plot_model

__all__ = [
    "add_gc",
    "component_to_circuit",
    "components",
    "model_factory",
    "model_from_gdsfactory",
    "model_from_sparameters",
    "model_from_filepath",
    "plot_model",
    "plot_circuit",
    "plot_circuit_montecarlo",
    "freq2wl",
    "wl2freq",
]

"""gdsfactory simphony circuit simulation plugin."""
from __future__ import annotations

try:
    from simphony.tools import freq2wl, wl2freq
except ImportError as e:
    print("To install simphony plugin make sure you `pip install simphony`")
    raise e

try:
    import SiPANN as _SIPANN
except ImportError as e:
    print("To install sipann plugin make sure you `pip install SiPANN`")
    raise e

from gdsfactory.simulation.simphony import components
from gdsfactory.simulation.simphony.add_gc import add_gc
from gdsfactory.simulation.simphony.circuit import component_to_circuit
from gdsfactory.simulation.simphony.components import model_factory
from gdsfactory.simulation.simphony.model_from_gdsfactory import (
    GDSFactorySimphonyWrapper,
)
from gdsfactory.simulation.simphony.model_from_sparameters import SimphonyFromFile
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
    "GDSFactorySimphonyWrapper",
    "SimphonyFromFile",
    "plot_model",
    "plot_circuit",
    "plot_circuit_montecarlo",
    "freq2wl",
    "wl2freq",
    "_SIPANN",
]

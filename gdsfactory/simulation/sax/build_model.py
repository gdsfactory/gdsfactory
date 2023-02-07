import copy
from pathlib import Path
from typing import Dict, Optional, Union

import jax.numpy as jnp
from tqdm.contrib.itertools import product

from gdsfactory.simulation.sax.interpolators import nd_nd_interpolation
from gdsfactory.simulation.sax.parameter import LayerStackThickness, NamedParameter
from gdsfactory.technology import LayerStack
from gdsfactory.typings import PortSymmetries


class Model:
    def __init__(
        self,
        trainable_component: callable,
        layerstack: LayerStack,
        trainable_parameters: Dict[
            str, Union[LayerStackThickness, NamedParameter]
        ] = None,
        non_trainable_parameters: Dict[
            str, Union[LayerStackThickness, NamedParameter]
        ] = None,
        simulation_settings: Optional[Dict[str, Union[float, str, int, Path]]] = None,
        num_modes: int = 2,
        port_symmetries: Optional[PortSymmetries] = None,
    ) -> None:
        """Utility class which simplifies the execution of simulations to build compact models for circuit simulations.

        It contains shared utilities for the different types of Models:
            - Optional simulation hyperparameter tuning routine (TODO)
            - Looping over training variables to generate training examples.
            - Consistent formatting of training examples for model building.

        Other functionality such as
            - Simulation setup, execution, caching/loading
            - Assembly of simulation results into S-parameters
        is solver-dependent, and hence resides in child classes.

        Attributes:
            trainable_component: callable wrapping component associated with model
            layerstack: complete layerstack associated with model
            trainable parameters: parameters that are model features that need to be learned (e.g. width, layer thicknesses, wavelength, materials properties)
            non_trainable_parameters: parameters for input_dict whose effect on S-parameters is known and does not need to be trained (e.g. length for a waveguide)
            simulation_settings: simulation parameters that can impact model quality (e.g. resolution, domain size). Set their converged property to True to skip convergence tuning.
            sim_settings: other simulation settings the solver can use
            num_modes: number of modes to consider for each port in S-parameter calculation.
            port_symmetries: as defined in FDTD solvers. Dict establishing equivalency between S-parameters.
                e.g. {"o1@0,o1@0": ["o2@0,o2@0", "o3@0,o3@0", "o4@0,o4@0"]} means that S22 = S33 = S44 are the same as S11,
                and hence the output vector for model training will only contain the key S11 to reduce the number of variables to fit.
        """
        self.component = trainable_component
        self.layerstack = layerstack
        self.trainable_parameters = trainable_parameters or {}
        self.non_trainable_parameters = non_trainable_parameters or {}
        self.simulation_settings = simulation_settings or {}
        self.num_modes = num_modes

        # Extract inputs and outputs vector size data
        self.size_inputs = len(self.trainable_parameters)
        self.num_ports = len(
            trainable_component(self.get_nominal_dict()).get_ports_list(
                port_type="optical"
            )
        )

        # Extract input and output vector label data
        self.input_vector_labels = list(self.trainable_parameters) + list(
            self.non_trainable_parameters.keys()
        )

        # self.size_outputs = self.num_ports * self.num_modes
        self.port_symmetries = port_symmetries

    def get_nominal_dict(self):
        """Return input_dict of nominal parameter values."""
        return {
            name: parameter.nominal_value
            for name, parameter in self.trainable_parameters.items()
        }

    def get_random_dict(self):
        """Return input_dict of randomly sampled parameter values."""
        return {
            name: parameter.sample()
            for name, parameter in self.trainable_parameters.items()
        }

    def parse_input_dict(self, input_dict):
        """Separates between LayerStackThickness inputs and NamedParameter inputs.

        Args:
            input_dict: key needs to match the keys in self.trainable_parameters
                values are the new values to assign to these parameters.
        """
        param_dict = {}
        layerstack_param_dict = {}
        for key, value in input_dict.items():
            if type(self.trainable_parameters[key]) is NamedParameter:
                param_dict[key] = value
            elif type(self.trainable_parameters[key]) is LayerStackThickness:
                layerstack_param_dict[key] = value
        return param_dict, layerstack_param_dict

    def perturb_layerstack(self, layerstack_param_dict):
        """Returns a temporary LayerStack with a new thickness value for the (currently) LayerStackThickness objects in layerstack_param_dict.

        Args:
            layerstack_param_dict: key needs to match a key in self.trainable_parameters having for value a LayerStackThickness object
                                    value is the thickness to assign to this parameter
        """
        perturbed_layerstack = copy.deepcopy(self.layerstack)
        for key, thickness in layerstack_param_dict.items():
            LayerStackThickness_obj = self.trainable_parameters[key]
            perturbed_layerstack.layers[
                LayerStackThickness_obj.layername
            ].thickness = thickness
        return perturbed_layerstack

    def get_model_input_output(self):
        """Generate training data

        Retrieve the input and output data for training the model by getting results on all trainable parameter input combinations.
        """
        ranges_dict = {
            name: parameter.arange()
            for name, parameter in self.trainable_parameters.items()
        }
        input_vectors = []
        output_vectors = []

        for values in product(*ranges_dict.values(), desc="Getting examples: "):
            # Compute results for this example
            input_dict = dict(
                zip(ranges_dict.keys(), [float(value) for value in values])
            )
            new_inputs, output_vector = self.outputs_from_inputs(input_dict)
            input_vector = list(values)
            input_vector.extend(iter(new_inputs))
            input_vectors.append(input_vector)
            output_vectors.append(output_vector)

        return (
            jnp.array(input_vectors),
            jnp.array(output_vectors),
        )

    def set_nd_nd_interp(self):
        """Returns ND-ND interpolator."""
        input_vectors, output_vectors = self.get_model_input_output()
        self.inference = nd_nd_interpolation(input_vectors, output_vectors)

    def validate(self, num_samples=1):
        """Validates the model by calculating random points within the interpolation range, and comparing to predictions."""
        calculated_outputs = []
        inferred_outputs = []
        validation_inputs = []
        for _index in range(num_samples):
            validation_inputs_local = self.get_random_dict()
            validation_inputs.append(validation_inputs_local)
            calculated_outputs.append(
                self.outputs_from_inputs(validation_inputs_local)[1]
            )
            inferred_outputs_local = [
                self.inference[mode](validation_inputs_local.values())
                for mode in range(self.num_modes)
            ]
            inferred_outputs.append(inferred_outputs_local)

        return validation_inputs, calculated_outputs, inferred_outputs

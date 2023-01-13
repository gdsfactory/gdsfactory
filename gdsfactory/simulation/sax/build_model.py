import copy
import itertools
from pathlib import Path
from typing import Dict, Optional, Union

import jax.numpy as jnp
from tqdm.contrib.itertools import product

from gdsfactory.simulation.sax.interpolators import nd_nd_interpolation
from gdsfactory.simulation.sax.parameter import LayerStackThickness, NamedParameter
from gdsfactory.technology import LayerStack
from gdsfactory.types import PortSymmetries


class Model:
    def __init__(
        self,
        component: callable,
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
            component: component associated with model
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
        self.component = component
        self.layerstack = layerstack
        self.trainable_parameters = trainable_parameters or {}
        self.non_trainable_parameters = non_trainable_parameters or {}
        self.simulation_settings = simulation_settings or {}
        self.num_modes = num_modes

        # Extract inputs and outputs vector size data
        self.size_inputs = len(self.trainable_parameters)
        self.num_ports = len(
            component(self.get_nominal_dict()).get_ports_list(port_type="optical")
        )
        self.size_outputs = self.num_ports * self.num_modes
        self.port_symmetries = port_symmetries

    """
    Parse parameters
    """

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

        Arguments:
            input_dict:  key needs to match the keys in self.trainable_parameters
                         values are the new values to assign to these parameters
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

        Arguments:
            layerstack_param_dict:  key needs to match a key in self.trainable_parameters having for value a LayerStackThickness object
                                    value is the thickness to assign to this parameter
        """
        perturbed_layerstack = copy.deepcopy(self.layerstack)
        for key, thickness in layerstack_param_dict.items():
            LayerStackThickness_obj = self.trainable_parameters[key]
            perturbed_layerstack.layers[
                LayerStackThickness_obj.layername
            ].thickness = thickness
        return perturbed_layerstack

    """
    Generate training data
    """

    def define_output_vector_labels(self):
        """Uses number of component ports, number of modes solved for, and port_symmetries to define smallest output vector."""
        output_vector_labels_iter = itertools.product(
            range(1, self.num_ports + 1),
            range(self.num_modes),
            range(1, self.num_ports + 1),
            range(self.num_modes),
        )
        output_vector_labels = []
        for output_label in output_vector_labels_iter:
            output_key1 = f"o{output_label[0]}@{output_label[1]}"
            output_key2 = f"o{output_label[2]}@{output_label[3]}"
            if output_key1 != output_key2:
                output_key = f"{output_key1},{output_key2}"
                output_vector_labels.append(output_key)

        if self.port_symmetries:
            for value_list in self.port_symmetries.values():
                for value in value_list:
                    output_vector_labels.remove(value)

        return output_vector_labels

    def get_results(self, input_dict):
        """To be overridden by child classes.

        Returns output vector requiring a compact model that is then used by get_Sparameters.
        """
        return None

    def get_data(self):
        """Retrieve the input and output data for training the model by getting results on all parameter input combinations."""
        ranges_dict = {
            name: parameter.arange()
            for name, parameter in self.trainable_parameters.items()
        }
        self.output_vector_labels = self.define_output_vector_labels()
        self.input_vector_labels = ranges_dict.keys()

        input_vectors = []
        output_vectors = []

        for values in product(*ranges_dict.values(), desc="Getting examples: "):
            # Compute results for this example
            input_dict = dict(
                zip(ranges_dict.keys(), [float(value) for value in values])
            )
            # sp = self.get_results(input_dict)
            results = self.get_results(input_dict)
            # Save input consistently
            input_vectors.append(values)
            # Save output consistently
            # output_vector = [sp[output_key] for output_key in self.output_vector_labels]
            # output_vectors.append(output_vector)
            output_vectors.append(results)

        return (
            jnp.array(input_vectors),
            jnp.array(output_vectors),
        )

    """
    Fitting data.
    """

    def get_nd_nd_interp(self):
        """Returns ND-ND interpolator."""
        input_vectors, output_vectors = self.get_data()
        return nd_nd_interpolation(
            input_vectors, self.output_vector_labels, output_vectors
        )

    """
    Compute S-parameters from the fits.
    """

    def get_Sparameters(self, input_dict):
        """To be overridden by child classes.

        Uses model prediction and non-trainable parameters to generate the Sparameters.
        """
        return None

import copy
from pathlib import Path
from typing import Dict, Optional, Union

import jax.numpy as jnp
from tqdm.contrib.itertools import product

from gdsfactory.simulation.sax.interpolators import nd_nd_interpolation
from gdsfactory.simulation.sax.mlp import mlp_regression
from gdsfactory.simulation.sax.parameter import (
    LayerStackThickness,
    NamedParameter,
    LithoParameter,
)
from gdsfactory.technology import LayerStack
from gdsfactory.typings import PortSymmetries
import ray


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
        address: str = None,
        dashboard_port: int = 8265,
        num_cpus: int = None,
        num_cpus_per_task: int = 1,
        # num_gpus_per_task: int = 0,
        restart_cluster: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Utility class which simplifies the execution of simulations to build compact models for circuit simulations.
        Aims to be agnostic to the specific simulator being used.

        It contains shared utilities for the different types of Models:
            - Optional simulation hyperparameter tuning routine (TODO)
            - Looping over training variables to generate training examples.
            - Consistent formatting of training examples for model building.
            - Interface with Ray clusters for distributed processing (even locally, this is beneficial)

        Other functionality such as
            - Simulation setup, execution, caching/loading
            - Assembly of simulation results into S-parameters
        is solver-dependent, and hence resides in child classes.

        TODO:
            - more consistent ordering of input/output data
            - more JAX, less pure Python
            - reuse Ray cluster across different model instances for simultaneous training

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
            address: of the Ray cluster to connect to. Defaults to finding a local running instance.
            dashboard_port: IP address of the dashboard to monitor the cluster
            num_cpus: available to the cluster (if not autoscaling)
            num_cpus_per_task: number of CPUs to assign to each task
            num_gpus_per_task: number of GPUs to assign to each task
            restart_cluster: if instantiating multiple models in the same Python session, whether to restart the cluster.
        """
        self.trainable_component = trainable_component
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

        # Cluster resources
        self.num_cpus_per_task = num_cpus_per_task
        # self.num_gpus_per_task = num_gpus_per_task
        if restart_cluster and ray.is_initialized():
            ray.shutdown()
        if not ray.is_initialized():
            ray.init(dashboard_port=dashboard_port, num_cpus=num_cpus)

    """
    PARAMETERS
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
        """Separates between LayerStackThickness, NamedParameter, and LithoParameter inputs.

        Args:
            input_dict: key needs to match the keys in self.trainable_parameters
                values are the new values to assign to these parameters.
        """
        param_dict = {}
        layerstack_param_dict = {}
        litho_param_dict = {}
        for key, value in input_dict.items():
            if type(self.trainable_parameters[key]) is NamedParameter:
                param_dict[key] = value
            elif type(self.trainable_parameters[key]) is LayerStackThickness:
                layerstack_param_dict[key] = value
            elif type(self.trainable_parameters[key]) is LithoParameter:
                litho_param_dict[key] = value
        return param_dict, layerstack_param_dict, litho_param_dict

    def perturb_layerstack(self, layerstack_param_dict):
        """Returns a temporary LayerStack with a new thickness value for the (current) LayerStackThickness objects in layerstack_param_dict.

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

    def perturb_geometry(self, current_component, litho_param_dict):
        """Returns a temporary Component on which all the morphological operations contained in the current litho_param_dict have been applied.

        Args:
            current_component: the current component, with params_dict already applied
            litho_param_dict: key needs to match a key in self.trainable_parameters having for value a LithoParameter object
                                    value is the input to the LithoParameter transformation attribute
        """
        for key, value in litho_param_dict.items():
            lithoParameter_obj = self.trainable_parameters[key]
            current_component = lithoParameter_obj.get_transformation(
                current_component, value
            )
        return current_component

    """
    QUEUING
    """

    def define_output_vector_labels(self):
        """Uses number of component ports, number of modes solved for, and port_symmetries to define smallest output vector."""
        output_vector_labels_iter = product(
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

    def arange_inputs(self, type="arange"):
        """Prepares input vectors for spanning simulation space.

        Arguments:
            type: str, arange or corners. Defines the iterator function to use with parameter objects.

        Returns:
            ranges_dict: dict, with keys parameter names and values arrays of containing all unique parameter values.
        """
        if type == "arange":
            ranges_dict = {
                name: parameter.arange()
                for name, parameter in self.trainable_parameters.items()
            }
        elif type == "corners":
            ranges_dict = {
                name: parameter.corners()
                for name, parameter in self.trainable_parameters.items()
            }
        else:
            raise ValueError("Type should be arange or corners.")

        return ranges_dict

    def get_output_from_inputs(self, labels, values, remote_function):
        """Get one output vector from a set of inputs.

        How to map parameters to simulation inputs depends on the target simulator.

        Arguments:
            labels: keys of the parameters
            values: values of the parameters
            remote_function: ray remote function object to use for the simulation

        Returns:
            remote function ID for delayed execution
            the remote function returns new_inputs, output_vectors
            new_inputs is an array containing derived inputs (e.g. wavelength in FDTD broadband simulation)
            output_vectors is a vector of reals representing the output to model (neff, s-params, etc.)
        """
        return NotImplementedError

    def get_all_inputs_outputs(self, type="arange"):
        """Get all outputs given all sets of inputs.

        get_output_from_inputs and remote_function are defined in the child class.
        """
        # Define possible parameter values
        ranges_dict = self.arange_inputs(type=type)
        # For all combinations of parameter values
        input_ids = list(product(*ranges_dict.values()))
        output_ids = [
            self.get_output_from_inputs(
                ranges_dict.keys(), values, self.remote_function
            )
            for values in product(*ranges_dict.values())
        ]
        # Execute the jobs
        results = ray.get(output_ids)

        # Parse the outputs into input and output vectors
        input_vectors = []
        output_vectors = []
        for input_example, output_example in zip(
            input_ids, results
        ):  # TODO no for loops!
            input_vector = list(input_example)
            input_vector.extend(output_example[0])
            input_vectors.append(input_vector)
            output_vectors.append(output_example[1])

        return (
            jnp.array(input_vectors),
            jnp.array(output_vectors),
        )

    """
    MODELS
    """

    def set_nd_nd_interp(self):
        """Returns ND-ND interpolator.

        Returns:
            self.inference: [callable giving an output_vector given an input_vector]
            list is of length 2*output_vector length, first output_vector entries are real, second imaginary
        """
        input_vectors, output_vectors = self.get_all_inputs_outputs()
        self.inference = nd_nd_interpolation(input_vectors, output_vectors)

    def set_mlp_interp(self):
        """Returns multilayer perceptron interpolator.

        Returns:
            self.inference: [callable giving an output_vector given an input_vector]
            list is of length 2*output_vector length, first output_vector entries are real, second imaginary
        """
        input_vectors, output_vectors = self.get_all_inputs_outputs()
        self.inference = mlp_regression(input_vectors, output_vectors)

    def input_dict_to_input_vector(self, input_dict):
        """Convert an input_dict with trainable and non-trainable parameters to an input vector on which inference can be performed.

        Find faster way!
        """
        return_array = [
            value
            for key, value in input_dict.items()
            if key in self.trainable_parameters.keys()
        ]
        return jnp.array(return_array)

    def validate(self, num_samples=1):
        """Validates the model by calculating random points within the interpolation range, and comparing to predictions."""
        calculated_outputs_ids = []
        inferred_outputs = []
        validation_inputs = []
        for _index in range(num_samples):
            validation_inputs_local = self.get_random_dict()
            validation_inputs.append(validation_inputs_local)
            calculated_outputs_ids.append(
                self.get_output_from_inputs(
                    validation_inputs_local.keys(),
                    validation_inputs_local.values(),
                    self.remote_function,
                )
            )
            inferred_outputs_local = [
                self.inference[mode](validation_inputs_local.values())
                for mode in range(self.num_modes)
            ]
            inferred_outputs.append(inferred_outputs_local)

        # Execute the jobs
        calculated_outputs_results = ray.get(calculated_outputs_ids)

        # Parse the outputs into input and output vectors
        calculated_outputs = [
            calculated_output_vector[1]
            for calculated_output_vector in calculated_outputs_results
        ]
        return validation_inputs, calculated_outputs, inferred_outputs

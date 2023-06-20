from pathlib import Path

import ray

from gdsfactory.config import sparameters_path
from gdsfactory.pdk import get_layer_stack
from gdsfactory.read import import_gds
from gdsfactory.simulation.gmeep import write_sparameters_meep
from gdsfactory.simulation.sax.build_model import Model


@ray.remote(num_cpus=1)
def remote_output_from_inputs(**kwargs):
    input_component = import_gds(kwargs["input_component_file"], read_metadata=True)
    output_vector_labels = kwargs["output_vector_labels"]
    del kwargs["input_component_file"]
    del kwargs["output_vector_labels"]

    sp = write_sparameters_meep(input_component, **kwargs)

    new_inputs = []
    output_vectors = []
    for wavelength_index, wavelength in enumerate(sp["wavelengths"]):
        new_inputs.append(wavelength)
        output_vectors.append(
            [sp[output_key][wavelength_index] for output_key in output_vector_labels]
        )

    return new_inputs, output_vectors


class MeepFDTDModel(Model):
    def __init__(self, **kwargs) -> None:
        """Generic model inferred from MEEP 2.5D FDTD calculations."""
        super().__init__(**kwargs)

        self.output_vector_labels = self.define_output_vector_labels()

        self.temp_dir = kwargs.get("tempdir") or Path(sparameters_path) / "temp_ray"
        self.temp_file_str = kwargs.get("tempfile") or "temp_ray"

        self.temp_dir.mkdir(exist_ok=True, parents=True)

        self.remote_function = remote_output_from_inputs.options(
            num_cpus=self.num_cpus_per_task,  # num_gpus=self.num_gpus_per_task
        )

        return None

    # def get_Sparameters(self, input_dict, output_vector_labels):
    #     """For FDTD, results are directly S-parameters."""
    #     sp = self.get_results(self, input_dict)
    #     output_vector = [sp[output_key] for output_key in self.output_vector_labels]
    #     output_vectors.append(output_vector)
    #     return sp

    def get_output_from_inputs(self, labels, values, remote_function):
        """Get one output vector from a set of inputs.

        How to map parameters to simulation inputs depends on the target simulator.

        Arguments:
            labels: keys of the parameters
            values: values of the parameters
            remote_function: ray remote function object to use for the simulation

        Returns:
            remote function ID for delayed execution
        """
        # Prepare this specific input vector
        input_dict = dict(zip(labels, [float(value) for value in values]))
        # Parse input vector according to parameter type
        param_dict, layerstack_param_dict, litho_param_dict = self.parse_input_dict(
            input_dict
        )
        input_component = self.perturb_geometry(
            self.trainable_component(param_dict), litho_param_dict
        )
        wavelengths = self.non_trainable_parameters["wavelength"]

        sim_settings = self.simulation_settings["sim_settings"]
        sim_settings["wavelength_start"] = wavelengths.min_value
        sim_settings["wavelength_stop"] = wavelengths.max_value
        sim_settings["wavelength_points"] = wavelengths.count()

        # We cannot serialize components, so save it as gds and import instead
        value_str = [str(value) for value in values]
        current_file = self.temp_dir / "_".join(list(value_str))
        input_component_file = current_file.with_suffix(".gds")
        input_component.write_gds(input_component_file, with_metadata=True)

        # Define function input given parameter values and transformed layerstack/component
        function_input = dict(
            input_component_file=input_component_file,
            cores=self.num_cpus_per_task,
            output_vector_labels=self.output_vector_labels,
        )
        function_input.update(sim_settings)
        # Assign the task to a worker
        return remote_function.remote(**function_input)


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.simulation.sax.parameter import LithoParameter, NamedParameter
    from gdsfactory.technology import LayerStack

    c = gf.components.coupler_full(
        coupling_length=0.1, dx=10.0, dy=5.0, gap=0.5, dw=0.0, cross_section="strip"
    )
    c.show(show_ports=True)

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack().layers[k]
            for k in (
                "slab90",
                "core",
                "box",
                "clad",
            )
        }
    )

    def trainable_coupler(parameters):
        return gf.components.coupler_full(
            coupling_length=parameters["coupling_length"],
            gap=parameters["gap"],
            dw=0.0,
        )

    port_symmetries_coupler = {
        "o1@0,o1@0": ["o2@0,o2@0", "o3@0,o3@0", "o4@0,o4@0"],
        "o2@0,o1@0": ["o1@0,o2@0", "o3@0,o4@0", "o4@0,o3@0"],
        "o3@0,o1@0": ["o1@0,o3@0", "o2@0,o4@0", "o4@0,o2@0"],
        "o4@0,o1@0": ["o1@0,o4@0", "o2@0,o3@0", "o3@0,o2@0"],
    }

    sim_settings = dict(
        resolution=10,
        xmargin=1.0,
        ymargin=1.0,
        is_3d=False,
        port_source_names=["o1"],
        port_symmetries=port_symmetries_coupler,
        run=True,
        overwrite=True,
        layer_stack=filtered_layerstack,
        z=0.1,
    )

    coupler_model = MeepFDTDModel(
        trainable_component=trainable_coupler,
        layerstack=filtered_layerstack,
        simulation_settings={
            "sim_settings": sim_settings,
        },
        trainable_parameters={
            "coupling_length": NamedParameter(
                min_value=5, max_value=5, nominal_value=5, step=10
            ),
            "gap": NamedParameter(
                min_value=0.2, max_value=0.2, nominal_value=0.2, step=0.3
            ),
            "erosion_dilation": LithoParameter(
                min_value=-0.2,
                max_value=0.2,
                nominal_value=0.0,
                step=0.4,
                layername="core",
            ),
        },
        non_trainable_parameters={
            "wavelength": NamedParameter(
                min_value=1.54, max_value=1.56, nominal_value=1.55, step=0.01
            ),
        },
        num_modes=1,
        num_cpus_per_task=1,
    )

    input_vectors, output_vectors = coupler_model.get_all_inputs_outputs(type="arange")
    # interpolator = coupler_model.set_nd_nd_interp()

    # import jax.numpy as jnp

    # params = jnp.stack(
    #     jnp.broadcast_arrays(
    #         jnp.asarray([6.0, 6.0, 6.0]),
    #         jnp.asarray([1.0, 1.0, 1.0]),
    #         jnp.asarray([1.55, 1.5, 1.55]),
    #         jnp.asarray([0.22, 0.22, 0.19]),
    #     ),
    #     0,
    # )

    # print(interpolator["o1@0,o2@0"](params))

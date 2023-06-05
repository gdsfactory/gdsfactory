import jax.numpy as jnp
import numpy as np
import ray
from sax.utils import reciprocal

from gdsfactory.pdk import get_layer_stack
from gdsfactory.simulation.fem.mode_solver import compute_cross_section_modes
from gdsfactory.simulation.sax.build_model import Model


@ray.remote(num_cpus=1)
def remote_output_from_inputs(
    cross_section,
    layerstack,
    wavelength,
    num_modes,
    order,
    radius,
    mesh_filename,
    resolutions,
    overwrite,
    with_cache,
):
    """Specific simulator, outside the class for stateless remote execution.

    Returns:
        new_inputs: numbers from the simulation that we would like to use as model inputs (e.g. wavelength in FDTD)
        output_vectors: vectors of reals capturing the output of the simulation
    """
    lams, basis, xs = compute_cross_section_modes(
        cross_section=cross_section,
        layerstack=layerstack,
        wl=wavelength,
        num_modes=num_modes,
        order=order,
        radius=radius,
        mesh_filename="mesh.msh",
        resolutions=resolutions,
        overwrite=overwrite,
        with_cache=True,
    )

    # Vector of reals
    real_neffs = np.real(lams)
    imag_neffs = np.imag(lams)

    return [], np.hstack((real_neffs, imag_neffs))


class FemwellWaveguideModel(Model):
    def __init__(self, **kwargs) -> None:
        """Waveguide model inferred from Femwell mode simula tion."""
        super().__init__(**kwargs)

        # remote function
        self.remote_function = remote_output_from_inputs.options(
            num_cpus=self.num_cpus_per_task,  # num_gpus=self.num_gpus_per_task
        )

        # results vector size
        self.size_results = self.num_modes

        return None

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
        # Apply required transformations depending on parameter type
        input_crosssection = self.trainable_component(param_dict).info["cross_section"]
        input_layerstack = self.perturb_layerstack(layerstack_param_dict)
        # Define function input given parameter values and transformed layerstack/component
        function_input = dict(
            cross_section=input_crosssection,
            layerstack=input_layerstack,
            wavelength=input_dict["wavelength"],
            num_modes=self.num_modes,
            order=self.simulation_settings["order"],
            radius=self.simulation_settings["radius"],
            mesh_filename="mesh.msh",
            resolutions=self.simulation_settings["resolutions"],
            overwrite=self.simulation_settings["overwrite"],
            with_cache=True,
        )
        return remote_function.remote(**function_input)

    def sdict(self, input_dict):
        """Returns S-parameters SDict from component using interpolated neff and length."""
        # Convert input dict to numeric (find better way to do this)
        input_numeric = self.input_dict_to_input_vector(input_dict)

        real_neffs = jnp.array(
            [self.inference[mode](input_numeric) for mode in range(self.num_modes)]
        )
        # imag_neffs = jnp.array(
        #     [
        #         self.inference[mode](input_numeric)
        #         for mode in range(self.num_modes, 2 * self.num_modes)
        #     ]
        # )  # currently not used
        phase = (
            2 * jnp.pi * real_neffs * input_dict["length"] / input_dict["wavelength"]
        )
        amplitude = jnp.asarray(
            10 ** (-input_dict["loss"] * input_dict["length"] / 20), dtype=complex
        )
        transmission = amplitude * jnp.exp(1j * phase)

        sp = {
            (f"o1@{mode}", f"o2@{mode}"): transmission[mode]
            for mode in range(self.num_modes)
        }
        return reciprocal(sp)


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.cross_section import rib
    from gdsfactory.simulation.sax.parameter import LayerStackThickness, NamedParameter
    from gdsfactory.technology import LayerStack

    c = gf.components.straight(
        cross_section=rib(width=2),
        length=10,
    )
    c.show()

    layerstack = get_layer_stack()

    filtered_layerstack = LayerStack(
        layers={
            k: layerstack.layers[k]
            for k in (
                "slab90",
                "core",
                "box",
                "clad",
            )
        }
    )

    def trainable_straight_rib(parameters):
        return gf.components.straight(cross_section=rib(width=parameters["width"]))

    rib_waveguide_model = FemwellWaveguideModel(
        trainable_component=trainable_straight_rib,
        layerstack=filtered_layerstack,
        simulation_settings={
            "resolutions": {
                "core": {"resolution": 0.02, "distance": 2},
                "clad": {"resolution": 0.2, "distance": 1},
                "box": {"resolution": 0.2, "distance": 1},
                "slab90": {"resolution": 0.05, "distance": 1},
            },
            "overwrite": False,
            "order": 1,
            "radius": jnp.inf,
        },
        trainable_parameters={
            "width": NamedParameter(
                min_value=0.3, max_value=1.0, nominal_value=0.5, step=0.1
            ),
            "wavelength": NamedParameter(
                min_value=1.545, max_value=1.555, nominal_value=1.55, step=0.005
            ),
            "core_thickness": LayerStackThickness(
                layerstack=filtered_layerstack,
                min_value=0.19,
                max_value=0.25,
                nominal_value=0.22,
                layername="core",
                step=0.3,
            ),
        },
        non_trainable_parameters={
            "length": NamedParameter(nominal_value=10),
            "loss": NamedParameter(nominal_value=1),
        },
        num_modes=4,
    )

    # Sweep corners
    # input_vectors, output_vectors = rib_waveguide_model.get_model_input_output(
    #     type="arange"
    # )

    # Sweep steps
    # input_vectors, output_vectors = rib_waveguide_model.get_all_inputs_outputs()
    interpolator = rib_waveguide_model.set_nd_nd_interp()
    # interpolator = rib_waveguide_model.set_mlp_interp()

    params = jnp.stack(
        jnp.broadcast_arrays(
            jnp.asarray([0.3, 0.55, 1.0]),
            jnp.asarray([1.55, 1.55, 1.55]),
            jnp.asarray([0.22, 0.22, 0.22]),
        ),
        0,
    )

    params_arr = [0.5, 1.55, 0.22, 10]

    params_dict = {
        "width": 0.5,
        "wavelength": 1.55,
        "core_thickness": 0.22,
        "length": 10,
        "loss": 1,
    }

    print(rib_waveguide_model.sdict(params_dict))

    import matplotlib.pyplot as plt

    widths = jnp.linspace(0.3, 1.0, 100)
    plt.plot(widths)

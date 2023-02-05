from gdsfactory.pdk import _ACTIVE_PDK, get_layer_stack
from gdsfactory.simulation.sax.build_model import Model
from gdsfactory.simulation.gmeep import write_sparameters_meep

_ACTIVE_PDK.materials_index.update(sin=2)


class MeepFDTDModel(Model):
    def __init__(self, **kwargs) -> None:
        """Generic model inferred from MEEP 2.5D FDTD calculations."""
        super().__init__(**kwargs)
        return None

    # def get_Sparameters(self, input_dict, output_vector_labels):
    #     """For FDTD, results are directly S-parameters."""
    #     sp = self.get_results(self, input_dict)
    #     output_vector = [sp[output_key] for output_key in self.output_vector_labels]
    #     output_vectors.append(output_vector)
    #     return sp

    def get_results(self, input_dict):
        """Setup and run a simulation with one set of inputs."""
        param_dict, layerstack_param_dict = self.parse_input_dict(input_dict)
        input_component = self.component(param_dict)
        # input_layerstack = self.perturb_layerstack(layerstack_param_dict)

        wavelengths = self.non_trainable_parameters["wavelength"]

        sim_settings = self.simulation_settings["sim_settings"]
        sim_settings["wavelength_start"] = wavelengths.min_value
        sim_settings["wavelength_stop"] = wavelengths.max_value
        sim_settings["wavelength_points"] = wavelengths.count()

        # TODO: 2.5D setup
        # In this case multiple wavelengths probably cannot be extracted from a single simulation, or need dispersive material
        sp = write_sparameters_meep(input_component, **sim_settings)
        # Reformat with wavelengths as inputs
        new_inputs = []
        output_vectors = []
        for wavelength_index, wavelength in enumerate(sp["wavelengths"]):
            new_inputs.append(wavelength)
            output_vectors.append(
                [
                    sp[output_key][wavelength_index]
                    for output_key in self.output_vector_labels
                ]
            )

        return new_inputs, output_vectors


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.simulation.sax.parameter import NamedParameter
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
        xmargin=1.0,
        ymargin=1.0,
        is_3d=False,
        port_source_names=["o1"],
        port_symmetries=port_symmetries_coupler,
        run=True,
        overwrite=False,
        layer_stack=filtered_layerstack,
        z=0.1,
    )

    coupler_model = MeepFDTDModel(
        component=trainable_coupler,
        layerstack=filtered_layerstack,
        simulation_settings={
            "sim_settings": sim_settings,
        },
        trainable_parameters={
            "coupling_length": NamedParameter(
                min_value=0.1, max_value=10.1, nominal_value=5, step=10
            ),
            "gap": NamedParameter(
                min_value=0.2, max_value=0.5, nominal_value=0.2, step=0.3
            ),
        },
        non_trainable_parameters={
            "wavelength": NamedParameter(
                min_value=1.54, max_value=1.56, nominal_value=1.55, step=0.01
            ),
        },
        num_modes=1,
    )

    # input_vectors, output_vectors = strip_strip_taper_model.get_data()
    interpolator = coupler_model.set_nd_nd_interp()

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

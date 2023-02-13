from gdsfactory.pdk import _ACTIVE_PDK, get_layer_stack
from gdsfactory.simulation.eme.meow_eme import MEOW
from gdsfactory.simulation.sax.build_model import Model

_ACTIVE_PDK.materials_index.update(sin=2)


class MeowEMEModel(Model):
    def __init__(self, **kwargs) -> None:
        """Generic model inferred from MEOW EME calculations."""
        super().__init__(**kwargs)
        return None

    def get_Sparameters(self, input_dict, output_vector_labels):
        """For EME, results are directly S-parameters."""
        return self.get_results(self, input_dict)

    def get_results(self, input_dict):
        """Setup and run a simulation with one set of inputs."""
        param_dict, layerstack_param_dict = self.parse_input_dict(input_dict)
        input_component = self.component(param_dict)
        input_layerstack = self.perturb_layerstack(layerstack_param_dict)

        eme = MEOW(
            component=input_component,
            layerstack=input_layerstack,
            wavelength=float(input_dict["wavelength"])
            if "wavelength" in input_dict
            else 1.55,
            num_modes=int(self.simulation_settings["num_eme_modes"])
            if "num_eme_modes" in input_dict
            else 4,
            cell_length=float(self.simulation_settings["cell_length"])
            if "cell_length" in input_dict
            else 0.5,
            resolution_x=int(self.simulation_settings["mode_res"])
            if "mode_res" in input_dict
            else 100,
            resolution_y=int(self.simulation_settings["mode_res"])
            if "mode_res" in input_dict
            else 100,
            spacing_x=float(self.simulation_settings["spacing_x"])
            if "spacing_x" in input_dict
            else 1,
            spacing_y=float(self.simulation_settings["spacing_y"])
            if "spacing_y" in input_dict
            else 1,
            overwrite=self.simulation_settings["overwrite"]
            if "overwrite" in input_dict
            else False,
        )
        return eme.compute_sparameters()


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.cross_section import rib, strip
    from gdsfactory.simulation.sax.parameter import LayerStackThickness, NamedParameter
    from gdsfactory.technology import LayerStack

    c = gf.components.taper_cross_section_linear(
        cross_section1=rib(width=2), cross_section2=rib(width=0.5)
    )
    c.show()

    # layerstack = gf.tech.get_layer_stack_generic()

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

    def trainable_strip_strip_taper(parameters):
        return gf.components.taper_cross_section_linear(
            length=parameters["length"],
            cross_section1=strip(width=0.5),
            cross_section2=strip(width=parameters["width2"]),
        )

    strip_strip_taper_model = MeowEMEModel(
        component=trainable_strip_strip_taper,
        layerstack=filtered_layerstack,
        simulation_settings={
            "mode_res": 100,
            "cell_length": 0.5,
            "num_eme_modes": 4,
            "spacing_x": 1,
            "spacing_y": -3,
            "overwrite": False,
        },
        trainable_parameters={
            "length": NamedParameter(min_value=5, max_value=6, nominal_value=6, step=1),
            "width2": NamedParameter(
                min_value=1.0, max_value=1.0, nominal_value=1.0, step=1.0
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
        num_modes=2,
    )

    # input_vectors, output_vectors = strip_strip_taper_model.get_data()
    interpolator = strip_strip_taper_model.get_nd_nd_interp()

    import jax.numpy as jnp

    params = jnp.stack(
        jnp.broadcast_arrays(
            jnp.asarray([6.0, 6.0, 6.0]),
            jnp.asarray([1.0, 1.0, 1.0]),
            jnp.asarray([1.55, 1.5, 1.55]),
            jnp.asarray([0.22, 0.22, 0.19]),
        ),
        0,
    )

    print(interpolator["o1@0,o2@0"](params))

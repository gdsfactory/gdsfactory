# # Variability analysis
#
# You can study the effect of variability on device performance using the same methods to iterate component parameters to build models.
#
# ## Lithographic parameters
#
# Not all variability can be captured by simply changing the Component or LayerStack input parameters.
#
# `LithoParameter` parameters have a parametrizable `transformation` attribute that you can use to modify the Component geometry prior to simulation in more complex ways than simply changing its calling arguments. The parameter has methods that return a temporary component given an initial component and a transformation type.
#
# Here are the transformations we support so far:
#
# 1. Dilation and erosion
# 2. Corner rounding
# 3. Offsets
# 4. Corner rounding
# 5. Corner analysis

# +
import gdsfactory as gf

c = gf.Component("myComponent")
poly1a = c.add_polygon(
    [[2.8, 3], [5, 3], [5, 0.8]],
    layer="WG",
)
poly1b = c.add_polygon(
    [
        [2, 0],
        [2, 2],
        [4, 2],
        [4, 0],
    ],
    layer="WG",
)
poly1c = c.add_polygon(
    [
        [0, 0.5],
        [0, 1.5],
        [3, 1.5],
        [3, 0.5],
    ],
    layer="WG",
)
poly2 = c.add_polygon(
    [
        [0, 0],
        [5, 0],
        [5, 3],
        [0, 3],
    ],
    layer="SLAB90",
)
poly3 = c.add_polygon(
    [
        [2.5, -2],
        [3.5, -2],
        [3.5, -0.1],
        [2.5, -0.1],
    ],
    layer="WG",
)
c.add_port(name="o1", center=(0, 1), width=1, orientation=0, layer=1)
c.add_port(name="o2", center=(3, -2), width=1, orientation=90, layer=1)
c.plot()
# -

# ### Dilation and erosion
#
# A `LithoParameter` of `type = "layer_dilation_erosion"` parametrizes a layerwise growing (positive value) or shrinking (negative value) of the geometry. Note that the ports are properly resized when they are on the transformed layer:

# +
from gdsfactory.simulation.sax.parameter import LithoParameter

param = LithoParameter(layername="core")
eroded_c = param.layer_dilation_erosion(c, 0.2)
eroded_c
# -

param = LithoParameter(layername="core")
eroded_c = param.layer_dilation_erosion(c, -0.3)
eroded_c

param = LithoParameter(layername="slab90")
eroded_c = param.layer_dilation_erosion(c, 0.2)
eroded_c

# ### Offsets
#
# Lithography can sometimes laterally offset layers w.r.t. to one another.
# This is captured by layerwise `type = "layer_x_offset"` and  `type = "layer_x_offset"`.
# Note that ports are also translated:

param = LithoParameter(layername="core")
offset_c = param.layer_x_offset(c, 0.5)
offset_c

param = LithoParameter(layername="core")
offset_c = param.layer_y_offset(c, -0.5)
offset_c

# ## Corner rounding
#
# The erosion and dilation above is done with "worst case" sharp corners.
# An erosion --> dilation --> erosion sequence, accessible with `type = "layer_round_corners"` can be done to parametrize corner rounding.
# For ports, here parts of the geometry overlapping with ports are patched to prevent the ports from being off the layer.

param = LithoParameter(layername="core")
smooth_c = param.layer_round_corners(c, 0.1)
smooth_c

param = LithoParameter(layername="core")
smooth_c = param.layer_round_corners(c, 0.4)
smooth_c

# ## Corner analysis
#
# For convenience, the model builder can also iterate over only the `min`, `max`, and `nominal` values of all trainable_parameters by using the `types=corners` instead of the default `types=arange` argument of `Model.get_model_input_output(type="corners")`.
#
# ## Directional coupler example
#
# Consider a directional coupler component which is modeled through a generic `MeepFDTDModel`. The only difference between this and the `FemwellWaveguideModel` from last notebook is how the simulation is defined: everything else involving iteration over parameters, multiprocessing, and model fitting, is identical. This makes model building easily extensible to new simulators.
#
# Here, we are only interested in variability analysis of the geometry, and so we create a trainable coupler with fixed length and gap:

# +
import gdsfactory as gf
from gdsfactory.simulation.sax.parameter import NamedParameter
from gdsfactory.technology import LayerStack
from gdsfactory.pdk import get_layer_stack


# gdsfactory layerstack
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


# trainable component function, choosing which parameters to fix and which to consider for the model
def trainable_coupler(parameters):
    return gf.components.coupler_full(
        coupling_length=10,
        gap=0.3,
        dw=0.0,
    )


c = trainable_coupler({})
c.plot()
# -

# When defining the model, we add the LithoParameter `erosion_magnitude`. For all models, a `TransformParameter` which if set, will offset the provided component prior to simulation, emulating erosion (when <1), nominal behaviour (when 1) and dilation (when >1). This morphological transformation is currently global; more advanced spatially-correlated filters are an obvious next step.

# +
from gdsfactory.simulation.sax.meep_FDTD_model import MeepFDTDModel

# Simulation settings
port_symmetries_coupler = {
    "o1@0,o1@0": ["o2@0,o2@0", "o3@0,o3@0", "o4@0,o4@0"],
    "o2@0,o1@0": ["o1@0,o2@0", "o3@0,o4@0", "o4@0,o3@0"],
    "o3@0,o1@0": ["o1@0,o3@0", "o2@0,o4@0", "o4@0,o2@0"],
    "o4@0,o1@0": ["o1@0,o4@0", "o2@0,o3@0", "o3@0,o2@0"],
}

sim_settings = dict(
    resolution=30,
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
    trainable_component=trainable_coupler,
    layerstack=filtered_layerstack,
    simulation_settings={
        "sim_settings": sim_settings,
    },
    trainable_parameters={
        "dilation_magnitude": LithoParameter(
            type="layer_dilation_erosion",
            layername="core",
            min_value=-0.05,
            max_value=0.05,
            nominal_value=0.0,
            step=0.05,
        ),
    },
    non_trainable_parameters={
        "wavelength": NamedParameter(
            min_value=1.54, max_value=1.56, nominal_value=1.55, step=0.01
        ),
    },
    num_modes=1,
)

# +
# input_vectors, output_vectors = coupler_model.get_model_input_output(type="corners")
# -

# We can analyze the output vectors as a function of input vectors to study variability (TODO).
#
# Since such a change of morphology can also be approximated with a change in gap and waveguide width of the original component, we can compare the results to a model of the component with these as swept NamedParameters (TODO).

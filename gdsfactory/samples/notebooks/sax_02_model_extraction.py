# # Automated model extraction
#
# You can use gdsfactory simulation plugins to build SDict models for circuit simulations.
#
# The parent `Model` class contains common logic for model building such as input-output vector definition from a set of input parameters, as well as fitting of the input-output vector relationships (for instance, through ND-ND interpolation and feedforward neural nets).  It further interfaces with [Ray](https://www.ray.io/) to distribute the required computations seamlessly from laptop, to cluster, to cloud.
#
# The children subclasses inherit all of this machinery, but further define solver- or component-specific information such as:
#
# - `outputs_from_inputs` method: how the input vectors (`Component`, `LayerStack`, or lithographic transformation arguments) are mapped to output vectors (this could directly be the S-parameters, or some solver results used to generate S-parameters like effective index)
# - `sdict` method: how the output vectors are mapped to S-parameter dictionaries for circuit simulation (this could directly be the result of `output_from_input`, or some downstream calculation using the output vectors with some extra Component parameters whose effect on the S-parameters is known and does not require training)
#
# For instance, consider a `straight` component in the generic LayerStack

# +
import jax.numpy as jnp

from gdsfactory.pdk import get_layer_stack

import gdsfactory as gf
from gdsfactory.cross_section import rib
from gdsfactory.simulation.sax.parameter import LayerStackThickness, NamedParameter
from gdsfactory.technology import LayerStack
from gdsfactory.generic_tech import get_generic_pdk
from loguru import logger

logger.remove()

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

c = gf.components.straight(
    cross_section=rib(width=2),
    length=10,
)
c.plot()

# +
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
# -

# We first wrap this component into a function taking for argument only a dictionary, the keys of which are used to parametrize the Component arguments we are interested in varying. Below, for instance, we force the component straight to have a `rib` cross-section, whose width can be varied.
#


def trainable_straight_rib(parameters):
    return gf.components.straight(cross_section=rib(width=parameters["width"]))


# ## Instantiating Models

# Next we can instantiate the `Model` proper. Here, we use the children class `FemwellWaveguideModel`. Its `outputs_from_inputs` method returns the effective index from the input geometry, and its `sdict` function uses the input geometry, length, and loss to return the S-parameters for the corresponding straight waveguide:

# +
from gdsfactory.simulation.sax.femwell_waveguide_model import FemwellWaveguideModel

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
            min_value=0.4, max_value=0.6, nominal_value=0.5, step=0.05
        ),
        "wavelength": NamedParameter(
            min_value=1.545, max_value=1.555, nominal_value=1.55, step=0.005
        ),
        "core_thickness": LayerStackThickness(
            layerstack=filtered_layerstack,
            min_value=0.21,
            max_value=0.23,
            nominal_value=0.22,
            layername="core",
            step=0.1,
        ),
    },
    non_trainable_parameters={
        "length": NamedParameter(nominal_value=10),
        "loss": NamedParameter(nominal_value=1),
    },
    num_modes=4,
)
# -

# Note the dictionary parameters:
#
# (1) the entries of `simulation_settings` are used by the model builder to parametrize the simulator,
#
# (2) the entries of `trainable_parameters` are used to define the simulation space that maps inputs to outputs and which requires interpolation, and
#
# (3) the entries of `non_trainable_parameters` are required to calculate the S-parameters, but do not appear in the simulator (their effect can be added after intermediate results have been interpolated).
#
#
# We also provide arguments to launch or connect to a Ray cluster to distribute the computations. `address` is the IP of the cluster (defaults to finding a local running instance, or launching one), `dashboard_port` is the local IP to connect to monitor the tasks, `num_cpus` is the total number of CPUs to allocate the cluster (defaults to autoscaling), `num_cpus_per_task` is the number of CPUs each raylet gets by default.
#
#
# ## Training models
#
# The Model object can generate input and output vectors requiring modelling from these dicts:

input_vectors, output_vectors = rib_waveguide_model.get_all_inputs_outputs()

# From above, we expect the input vector to have a number of rows equal to the set of trainable parameter points, here len(widths) x len(core_thickness) x len(wavelength) = 15, and a number of columns equal to the number of trainable parameters (3):

# +
import numpy as np

print(np.shape(input_vectors))
print(input_vectors[0])
# -

# The output (here, the effective indices) will have #input_vector rows, and #modes columns:

print(output_vectors[0])
print(np.shape(output_vectors))

# Typically we are not interested in these vectors per say, but in some interpolation model between them. One way is to perform ND-ND interpolation:

rib_waveguide_model.set_nd_nd_interp()

# The populates the model with an interpolator
#
# ## Model inference
#
# These can then be used to construct the S-parameters within the trainable_parameter range:

# +
params_dict = {
    "width": 0.5,
    "wavelength": 1.55,
    "core_thickness": 0.22,
    "length": 10,
    "loss": 1,
}

print(rib_waveguide_model.sdict(params_dict))
# -

# These can also be called as arrays:

# +
params_dict = {
    "width": jnp.array([0.5, 0.3, 0.65]),
    "wavelength": jnp.array([1.55, 1.547, 1.55]),
    "core_thickness": jnp.array([0.22, 0.22, 0.21]),
    "length": jnp.ones(3) * 10,
    "loss": jnp.ones(3) * 1,
}

print(rib_waveguide_model.sdict(params_dict))
# -

# ## Model validation

# We can validate the intermediate input-output relationships by comparing the predictions to new simulations within the trainable parameter space:

validation_inputs, calculated_outputs, inferred_outputs = rib_waveguide_model.validate(
    num_samples=1
)

validation_inputs

input_vectors

output_vectors

# While the trend seems reasonable, the model above could benefit from more examples or better simulation parameter tuning.

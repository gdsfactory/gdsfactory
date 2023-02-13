from typing import List

import jax
import jax.numpy as jnp


def nd_interpolation(_grid, output_vector):
    """Return JAX N-D interpolator given a M-D input and 1-D output vector."""
    _data = jnp.asarray(output_vector).reshape(len(_grid[i]) for i in range(len(_grid)))

    # @jax.jit
    def _get_coordinate(arr1d: jnp.ndarray, value: jnp.ndarray):
        return jnp.interp(value, arr1d, jnp.arange(arr1d.shape[0]))

    # @jax.jit
    def _get_coordinates(arrs1d: List[jnp.ndarray], values: jnp.ndarray):
        return jnp.array([_get_coordinate(a, v) for a, v in zip(arrs1d, values)])

    # @jax.jit
    def interp_output(params):
        coords = _get_coordinates(_grid, params)
        return jax.scipy.ndimage.map_coordinates(_data, coords, 1, mode="nearest")

    return interp_output


def nd_nd_interpolation(input_vectors, output_vectors):
    """Return JAX N-D interpolator given a N-D input and M-D output vector."""
    _grid = [jnp.sort(jnp.unique(input_vector)) for input_vector in input_vectors.T]
    return [
        nd_interpolation(_grid, jnp.asarray(output_vector))
        for output_vector in output_vectors.T
    ]


# def nd_nd_interpolation(input_vectors, output_labels, output_vectors):
#     """Return JAX N-D interpolator given a N-D input and M-D output vector."""
#     _grid = [jnp.sort(jnp.unique(input_vector)) for input_vector in input_vectors.T]
#     return {
#         output_label: nd_interpolation(_grid, jnp.asarray(output_vector))
#         for output_label, output_vector in zip(output_labels, output_vectors.T)
#     }

# flake8: noqa
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jran


def mlp_regression(
    input_vectors,
    output_vectors,
    training_test_split: float = 0.8,
    num_layers: int = 3,
    num_neurons_per_layer: int = 5,
    learning_rate: float = 1e-2,
    num_epochs: int = 100,
    batch_size: int = 3,
    seed: int = 0,
    save: str = "weights",
):
    """Fit multilayer perceptron inference model for vectors [self.num_inputs] --> [self.num_outputs].
    Regression task using SeLu activation on all hidden layers. Adapted from https://gist.github.com/aphearin/0da99f715906715e1d7d6b004c2dbb73.
    TODO:
        (1) more flexibility to parametrize the training (adaptive learning rate, optimizer choice, etc.),
        (2) more flexibility in network topologies, activations
        (3) network hyperparameter tuning
        (4) try low-data online learning
    Arguments:
        input_vectors: vector of inputs
        output_vectors: vector of outputs (examples)
        training_test_split: % training vs test data
        num_layers: number of layers in the neural net.
        num_neurons_per_layer: number of neurons per layer.
        learning_rate: for training.
        num_epochs: for training.
        batch_size: for training.
        seed: for weight initialization
        save: where to save the model weights.
    """

    # Split into training and test data
    x = np.random.rand(jnp.shape(input_vectors)[0])
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = (
        indices[: int(training_test_split * len(x))],
        indices[int(training_test_split * len(x)) :],
    )
    training_inputs, test_inputs = input_vectors[training_idx], input_vectors[test_idx]
    training_outputs, test_outputs = (
        output_vectors[training_idx],
        output_vectors[test_idx],
    )

    def feedforward_prediction(params, abscissa):
        """Each neuron is just the activation function applied to y = w*x + b, except for the final layer, when no activation function is used.
        Parameters
        ----------
        params : list
            Parameters of the network, with one list element per layer.
            See notes below on network initialization.
        abscissa : ndarray
            Array of shape (batch_size, n_features)
        Returns
        -------
        preds : ndarray
            Array of shape (batch_size, n_targets)
        """
        activations = abscissa

        #  Loop over every dense layer except the last
        for w, b in params[:-1]:
            outputs = jnp.dot(w, activations) + b  # apply affine transformation
            activations = jax.nn.selu(outputs)  #  apply nonlinear activation

        #  Now for the final layer
        w_final, b_final = params[-1]
        final_outputs = jnp.dot(w_final, activations) + b_final
        return final_outputs  # Final layer is just w*x + b with no activation

    def get_random_layer_params(m, n, ran_key, scale=0.01):
        """Helper function to randomly initialize.
        weights and biases using the JAX-defined randoms.
        """
        w_key, b_key = jran.split(ran_key)
        ran_weights = scale * jran.normal(w_key, (n, m))
        ran_biases = scale * jran.normal(b_key, (n,))
        return ran_weights, ran_biases

    def get_init_network_params(sizes, ran_key):
        """Initialize all layers for a fully-connected neural network."""
        keys = jran.split(ran_key, len(sizes))
        return [
            get_random_layer_params(m, n, k)
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)
        ]

    def get_network_layer_sizes(n_features, n_targets, n_layers, n_neurons_per_layer):
        dense_layer_sizes = [n_neurons_per_layer] * n_layers
        layer_sizes = [n_features, *dense_layer_sizes, n_targets]
        return layer_sizes

    ran_key = jran.PRNGKey(seed)

    num_features, num_targets = (
        jnp.shape(training_inputs)[1],
        jnp.shape(training_outputs)[1],
    )

    layer_sizes = get_network_layer_sizes(
        num_features, num_targets, num_layers, num_neurons_per_layer
    )

    init_params = get_init_network_params(layer_sizes, ran_key)

    batched_prediction = jax.vmap(feedforward_prediction, in_axes=(None, 0))

    # @jax.jit
    def mse_loss(params, abscissa, targets):
        preds = batched_prediction(params, abscissa)
        diff = preds - targets
        return jnp.sum(diff * diff) / preds.shape[0]

    # @jax.jit
    def update(params, x, y, learning_rate):
        grads = jax.grad(mse_loss)(params, x, y)
        return [
            (w - learning_rate * dw, b - learning_rate * db)
            for (w, b), (dw, db) in zip(params, grads)
        ]

    params = init_params
    losses = []
    for epoch in range(num_epochs):  # noqa: B007
        # Sample batch size test and training
        training_indices = np.random.choice(
            range(len(training_inputs)), batch_size, replace=False
        )
        testing_indices = np.random.choice(
            range(len(test_inputs)), batch_size, replace=False
        )
        current_loss = mse_loss(
            params, test_inputs[testing_indices], test_outputs[testing_indices]
        )
        losses.append(current_loss)
        params = update(
            params,
            training_inputs[training_indices],
            training_outputs[training_indices],
            learning_rate,
        )

    # Format into per-mode callable
    return [
        lambda input_vector: feedforward_prediction(params, input_vector)[i]
        for i in range(num_targets)
    ]

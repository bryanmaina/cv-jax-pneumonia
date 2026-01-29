import jax
import jax.numpy as jnp
import optax


def weighted_cross_entropy(
    logits: jax.Array, labels: jax.Array, weights: jax.Array
) -> jax.Array:
    """
    Computes weigted softmax cross entropy.

    This function applies a class-specific weight to eash sample's loss before calculating the mean acros the batch.

    Args:
        logits: Unormalized log probabilities with shape (batch_size, num_classes)
        labels: Integer array of ground truth class indeices iwth sape (batch_size, )
        weights: Array of weights for each class with shape (num_classes, )

    Returns:
        The scalar mean weighted cross-entropy loss.
    """
    unweighted_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    sample_weights = jnp.take(weights, labels)
    weighted_loss = unweighted_loss * sample_weights

    return jnp.mean(weighted_loss)

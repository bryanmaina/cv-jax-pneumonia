import jax.numpy as jnp
import optax

from src.losses import weighted_cross_entropy


def test_weighted_cross_entropy():
    # logits: (batch_size, num_classes)
    # labels: (batch_size,)
    logits = jnp.array([[10.0, -10.0], [-10.0, 10.0]])  # Very certain predictions
    labels = jnp.array([0, 1])

    # Weights: class 0 has weight 1.0, class 1 has weight 2.0
    weights = jnp.array([1.0, 2.0])

    # Loss should be low since predictions match labels
    loss = weighted_cross_entropy(logits, labels, weights)
    assert loss < 0.1

    # Inverse lables to get high loss
    bad_labels = jnp.array([1, 0])
    loss_high = weighted_cross_entropy(logits, bad_labels, weights)

    # Prediction for class 1 (wrong) should be weighted by 2.0
    # Prediction for class 0 (wrong) should be weighted by 1.0
    # Let's compare with unweighted
    unweighted_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, bad_labels
    ).mean()

    # Roughly: (2.0 * loss_for_sample_0 + 1.0 * loss_for_sample_1) / 2
    # In unweighted: (loss_for_sample_0 + loss_for_sample_1) / 2
    # Since logits are symetric, loss_for_sample_0 == loss_for_sample_1
    # Weighted loss should be ~1.5 * unweighted_loss
    assert jnp.allclose(loss_high, 1.5 * unweighted_loss, rtol=1e-2)

from tensorflow import reduce_mean, Tensor, maximum, reduce_sum, math, sqrt, square, float32, convert_to_tensor
from tensorflow.keras.losses import binary_crossentropy


def wasserstein_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculates the Wasserstein loss for a batch of predicted and true values.

    The Wasserstein loss function is often used in training generative adversarial networks (GANs)
    to measure the distance between the distribution of generated data and real data. This loss
    function helps in stabilizing the training process of GANs by providing a more meaningful
    and smooth gradient signal.

    Args:
        y_true (Tensor): The ground truth values, usually -1 or 1 indicating real or fake samples.
        y_pred (Tensor): The predicted values from the discriminator.

    Returns:
        Tensor: The computed Wasserstein loss as a single scalar tensor.
    """
    return reduce_mean(y_true * y_pred)


def binary_cross_entropy_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculates the binary cross entropy loss, commonly used for training GANs.

    This function measures the performance of a classification model whose output is a
    probability value between 0 and 1. It penalizes the probability based on the distance
    from the actual label.

    Args:
        y_true (Tensor): The true labels (real or fake).
        y_pred (Tensor): The predicted probabilities.

    Returns:
        Tensor: The computed binary cross-entropy loss.
    """
    return binary_crossentropy(y_true, y_pred)


def hinge_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Hinge loss for training GANs. It is particularly useful for "GANs with hinge loss" which
    has been shown to improve the quality of generated images.

    Args:
        y_true (Tensor): The true labels, where real samples have labels -1 or 1.
        y_pred (Tensor): The discriminator's prediction.

    Returns:
        Tensor: The computed hinge loss.
    """
    return reduce_mean(maximum(0., 1. - y_true * y_pred))


def least_squares_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Least squares loss function for GANs, provides penalties in a squared manner which
    often leads to more stable training by mitigating the issue of vanishing gradients.

    Args:
        y_true (Tensor): The ground truth labels.
        y_pred (Tensor): The predicted values from the discriminator.

    Returns:
        Tensor: The computed least squares loss.
    """
    return reduce_mean((y_true - y_pred) ** 2)


def kl_divergence_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculates the Kullback-Leibler divergence loss for a batch of predictions. This loss
    measures how one probability distribution diverges from a second expected probability distribution.

    Args:
        y_true (Tensor): The true distribution (real samples).
        y_pred (Tensor): The predicted distribution (generated samples).

    Returns:
        Tensor: The computed KL divergence loss.
    """
    return reduce_sum(y_true * math.log(y_true / (y_pred + 1e-8)), axis=1)


def pearson_correlation_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Pearson correlation loss for GANs, which attempts to maximize the Pearson correlation
    coefficient between the predicted and true values. This loss helps ensure that the predictions
    closely follow the trends in the real data.

    Args:
        y_true (Tensor): The true labels or values.
        y_pred (Tensor): The predicted outputs from the model.

    Returns:
        Tensor: The negative of the Pearson correlation coefficient as the loss value.
    """
    y_true_mean = reduce_mean(y_true, axis=0)
    y_pred_mean = reduce_mean(y_pred, axis=0)
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean
    correlation = reduce_sum(y_true_centered * y_pred_centered) / (
            sqrt(reduce_sum(square(y_true_centered))) * sqrt(reduce_sum(square(y_pred_centered))) + 1e-8)
    return convert_to_tensor(-correlation, dtype=float32)

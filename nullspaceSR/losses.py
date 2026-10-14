from torch import Tensor


def coefficient_image_mse(predicted: Tensor, target: Tensor, *,
                          channels: int, scale: int) -> Tensor:
    """Coefficient SSE / HR pixel count (including channels).

    With an orthonormal decoder and LR=A(HR), image MSE equals truncation
    image MSE plus this loss. Normalization is comparable across ranks.
    """
    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("Coefficients must have matching BCHW shapes")
    if channels < 1 or scale < 2 or predicted.shape[1] % channels:
        raise ValueError("Invalid channel/scale configuration")
    b, _, h, w = predicted.shape
    return (predicted - target).square().sum() / (b * channels * scale**2 * h * w)

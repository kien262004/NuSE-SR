from torch import Tensor, nn

from ..operators import AveragePoolSR
from ..pca import BlockNullSpacePCA


class NuSESR(nn.Module):
    """SR predictor with an optional fixed PCA null-space decoder plugin."""

    def __init__(self, predictor: nn.Module, decoder: BlockNullSpacePCA | None = None) -> None:
        super().__init__()
        self.predictor = predictor
        self.decoder = decoder
        self.channels = decoder.channels if decoder is not None else predictor.channels
        self.scale = decoder.scale if decoder is not None else predictor.scale
        self.operator = AveragePoolSR(self.scale) if decoder is not None else None

    @property
    def uses_null_space(self) -> bool:
        return self.decoder is not None

    @property
    def loss_name(self) -> str:
        return "train_coefficient_image_mse" if self.uses_null_space else "train_image_mse"

    def predict_coefficients(self, lr: Tensor) -> Tensor:
        if not self.uses_null_space:
            raise RuntimeError("Coefficient prediction requires the PCA null-space plugin")
        if lr.ndim != 4 or lr.shape[1] != self.decoder.channels:
            raise ValueError(f"Expected BCHW LR with {self.decoder.channels} channels")
        coeffs = self.predictor(lr)
        expected = (lr.shape[0], self.decoder.channels * self.decoder.rank, *lr.shape[-2:])
        if tuple(coeffs.shape) != expected:
            raise ValueError(f"Predictor output must have shape {expected}, got {tuple(coeffs.shape)}")
        return coeffs

    def predict_null(self, lr: Tensor) -> Tensor:
        """Return only the synthesized null-space detail."""
        return self.decoder(self.predict_coefficients(lr))

    def forward(self, lr: Tensor) -> Tensor:
        if not self.uses_null_space:
            return self.predictor(lr)
        return self.operator.pseudoinverse(lr) + self.predict_null(lr)


# Compatibility name; decoder is optional for image-space predictors.
RNDModel = NuSESR

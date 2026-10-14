from torch import Tensor, nn

from nullspaceSR.operators import AveragePoolSR
from nullspaceSR.pca import BlockNullSpacePCA


class NuSESR(nn.Module):
    """LR -> coefficient maps -> fixed null decoder -> consistent SR.

    Every decoded detail is already in the block-average null space.
    No image-space projection is applied in forward.
    """

    def __init__(self, predictor: nn.Module, decoder: BlockNullSpacePCA) -> None:
        super().__init__()
        self.predictor = predictor
        self.decoder = decoder
        self.operator = AveragePoolSR(decoder.scale)

    def predict_coefficients(self, lr: Tensor) -> Tensor:
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
        return self.operator.pseudoinverse(lr) + self.predict_null(lr)


# Compatibility name: constructor now accepts (predictor, PCA decoder).
RNDModel = NuSESR

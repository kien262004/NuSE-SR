from torch import Tensor, nn


class SmallCoefficientCNN(nn.Module):
    """LR-domain backbone; head emits C*k coefficient maps."""

    def __init__(self, channels: int = 3, rank: int = 8, width: int = 32) -> None:
        super().__init__()
        if min(channels, rank, width) < 1:
            raise ValueError("channels, rank and width must be positive")
        self.layers = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, channels * rank, 3, padding=1),
        )

    def forward(self, lr: Tensor) -> Tensor:
        return self.layers(lr)

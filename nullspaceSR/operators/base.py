from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class MeasurementOperator(nn.Module, ABC):
    """Measurement A and its chosen pseudoinverse A-dagger."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply A to a candidate signal/image."""

    @abstractmethod
    def pseudoinverse(self, y: Tensor) -> Tensor:
        """Map a measurement to a signal-space solution."""

"""Synthetic demonstration; not a DIV2K training/evaluation result."""

import torch
from nullspaceSR import (
    AveragePoolSR, BlockNullSpacePCA, NuSESR,
    NuSEBackbone, coefficient_image_mse,
)


def main() -> None:
    torch.manual_seed(2026)
    scale, rank, channels = 4, 8, 3
    # Replace TRAIN samples with a loader yielding floating HR tensors.
    train_hr = [torch.rand(2, channels, 32, 32) for _ in range(3)]
    decoder = BlockNullSpacePCA.fit(train_hr, scale=scale, rank=rank)
    model = NuSESR(NuSEBackbone(channels, rank, scale=scale, eigenvalues=decoder.eigenvalues), decoder)
    operator = AveragePoolSR(scale)
    hr = torch.rand(2, channels, 32, 48)  # Separate rectangular sample
    lr = operator(hr)
    target = decoder.encode_hr(hr)
    coeffs = model.predict_coefficients(lr)
    detail = decoder(coeffs)
    output = operator.pseudoinverse(lr) + detail
    oracle = operator.pseudoinverse(lr) + decoder(target)
    prediction_loss = coefficient_image_mse(coeffs, target, channels=channels, scale=scale)
    truncation = (oracle - hr).square().mean()
    image_loss = (output - hr).square().mean()
    torch.testing.assert_close(operator(detail), torch.zeros_like(lr), atol=1e-6, rtol=0)
    torch.testing.assert_close(operator(output), lr, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(image_loss, truncation + prediction_loss)
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=1e-3)
    optimizer.zero_grad()
    prediction_loss.backward()
    optimizer.step()
    torch.testing.assert_close(operator(model(lr)), lr, atol=1e-6, rtol=1e-6)
    print(f"LR {tuple(lr.shape)} -> coefficients {tuple(coeffs.shape)} -> SR {tuple(output.shape)}")
    print(f"max consistency error: {(operator(output) - lr).abs().max().item():.3e}")
    print(f"image MSE: {image_loss.item():.6f}; truncation + prediction: {(truncation + prediction_loss).item():.6f}")
    print("One supervised coefficient training step completed.")


if __name__ == "__main__":
    main()

"""Kiểm c3 không cần dữ liệu: lúc khởi tạo bằng NuSEE, loss = Σα·MSE ảnh, R học được và vẫn trực chuẩn trong null
space, lr hiệu dụng của R, gộp R khi suy luận (tham số như e80 / d8), nạp lại checkpoint.

Chạy: python -m unittest experiments/c3_pca_mixing/test_pca_mixing.py   (từ gốc repo)
"""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from experiments.testing import fake_stats, load_experiment, randomize  # noqa: E402
from nuse_e.config import DESIGN  # noqa: E402
from nuse_e.models.nuse_e import NuSEE  # noqa: E402
from nuse_e.models.registry import load_run, make_nuse  # noqa: E402
from nuse_e.nullspace.architecture import architecture_from_stats, loss_alpha  # noqa: E402
from nuse_e.operators import degrade  # noqa: E402

EXPERIMENT = load_experiment(HERE)


def arch_of(stats, variant):
    return architecture_from_stats(stats, **{**DESIGN, **EXPERIMENT.VARIANTS[variant]})


def image_loss(model, lr, hr):
    alpha = loss_alpha(model.alpha, len(model.budgets))
    return sum(a * F.mse_loss(model(lr, k), hr) for k, a in zip(model.budgets, alpha))


class PCAMixingTest(unittest.TestCase):
    def setUp(self):
        self.stats = fake_stats()
        self.hr = torch.rand(2, 3, 32, 40, dtype=torch.float64)
        self.lr = degrade(self.hr, "average_pool")

    def model(self, variant):
        return make_nuse(self.stats, arch_of(self.stats, variant)).double()

    def test_same_architecture_and_inference_params_as_reference(self):
        for mix, ref in (("e80mix", "e80"), ("d8mix", "d8")):
            a, b = arch_of(self.stats, mix), arch_of(self.stats, ref)
            self.assertEqual({k: v for k, v in a.items() if k not in ("impl", "options")}, b)
            mixed, plain = make_nuse(self.stats, a).fuse(), NuSEE(self.stats, b).fuse()
            self.assertEqual([mixed.params_for(k) for k in mixed.budgets], [plain.params_for(k) for k in plain.budgets])
            self.assertEqual(sum(p.numel() for p in mixed.parameters()), sum(p.numel() for p in plain.parameters()))

    def test_initial_model_equals_nusee_with_same_weights(self):
        mixed = self.model("d8mix").eval()
        randomize([mixed.stem, mixed.stages, mixed.heads], std=0.1)
        plain = NuSEE(self.stats, arch_of(self.stats, "d8")).double().eval()
        plain.load_state_dict({k: v for k, v in mixed.state_dict().items() if k in plain.state_dict()}, strict=False)
        plain.decoder.basis.copy_(mixed.decoder.source_basis)
        with torch.no_grad():
            for k in plain.budgets:
                self.assertLess((mixed(self.lr, k) - plain(self.lr, k)).abs().max().item(), 1e-12, f"K={k}")

    def test_loss_equals_image_mse_and_reaches_rotation(self):
        for variant in ("e80mix", "d8mix"):
            model = self.model(variant)
            randomize([model], std=0.1)
            loss = model.loss(self.lr, self.hr)                               # NuSEE.loss tính ở float32
            exact = image_loss(model, self.lr, self.hr)
            self.assertLess(abs(loss.item() - exact.item()) / exact.item(), 1e-5, variant)
            loss.backward()
            self.assertGreater(model.decoder.delta.grad.abs().sum().item(), 0, variant)

    def test_rotation_stays_orthonormal_in_null_space(self):
        model = self.model("e80mix")
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(5):
            opt.zero_grad()
            model.loss(self.lr, self.hr).backward()
            opt.step()
        u = model.decoder.basis.detach()
        self.assertGreater((u - model.decoder.source_basis).abs().max().item(), 1e-4)
        eye = torch.eye(u.shape[-1], dtype=u.dtype).expand(3, -1, -1)      # U lưu float32 trong NuSEE: sai số ~1e-8
        self.assertLess((torch.einsum("cpk,cpj->ckj", u, u) - eye).abs().max().item(), 1e-6)
        self.assertLess(u.sum(1).abs().max().item(), 1e-6)                    # 1^T U' = 0: trong null space
        with torch.no_grad():
            for k in model.budgets:
                self.assertLess((degrade(model(self.lr, k), "average_pool") - self.lr).abs().max().item(), 1e-6)

    def test_effective_learning_rate_of_rotation_is_scaled(self):
        model = self.model("e80mix")
        lr = 1e-3
        opt = torch.optim.Adam(model.decoder.parameters(), lr=lr)
        model.loss(self.lr, self.hr).backward()
        opt.step()
        step = (model.decoder.scale * model.decoder.delta).abs().max().item()   # thay đổi của I + s·Δ
        self.assertAlmostEqual(step, EXPERIMENT.MIXING["basis_lr_scale"] * lr, delta=1e-7)

    def test_fuse_and_checkpoint_round_trip(self):
        arch = arch_of(self.stats, "e80mix")
        model = make_nuse(self.stats, arch)
        randomize([model], std=0.1)
        model.eval()
        with torch.no_grad():
            want = [model(self.lr.float(), k).clone() for k in model.budgets]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best.pt"
            torch.save({"nuse_e": {"model": "NuSE-E/e80mix"}, "stats": self.stats, "arch": arch,
                        "model": model.state_dict()}, path)
            loaded, _ = load_run(path)
        self.assertEqual(type(loaded.decoder).__name__, "FixedDecoder")
        with torch.no_grad():
            for w, k in zip(want, loaded.budgets):
                self.assertLess((loaded(self.lr.float(), k) - w).abs().max().item(), 1e-5, f"K={k}")


if __name__ == "__main__":
    unittest.main()

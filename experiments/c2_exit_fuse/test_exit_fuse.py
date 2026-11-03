"""Kiểm c2 không cần dữ liệu: lúc khởi tạo bằng NuSEE, exit nhỏ không đọc phần sau, tham số đúng công thức.

Chạy: python -m unittest experiments/c2_exit_fuse/test_exit_fuse.py   (từ gốc repo)
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

import torch  # noqa: E402

from experiments.testing import e80_arch, fake_stats, load_experiment, randomize  # noqa: E402
from nuse_e.config import PARAM_CAP  # noqa: E402
from nuse_e.models.blocks import ResBlock  # noqa: E402
from nuse_e.models.nuse_e import NuSEE  # noqa: E402
from nuse_e.models.registry import make_nuse  # noqa: E402
from nuse_e.operators import degrade  # noqa: E402

IMPL = load_experiment(HERE).IMPL


class ExitFuseTest(unittest.TestCase):
    def setUp(self):
        self.stats = fake_stats()
        self.y = degrade(torch.rand(1, 3, 40, 56, dtype=torch.float64), "average_pool")

    def test_initial_model_equals_nusee_with_same_weights(self):
        fused = make_nuse(self.stats, e80_arch((2, 1, 1), 8, impl=IMPL)).double().eval()
        randomize([fused.stem, fused.stages, fused.heads])            # mixer giữ khởi tạo đơn vị
        plain = NuSEE(self.stats, e80_arch((2, 1, 1), 8)).double().eval()
        plain.load_state_dict({k: v for k, v in fused.state_dict().items() if not k.startswith("mixers.")})
        with torch.no_grad():
            for k in plain.budgets:
                self.assertLess((fused(self.y, k) - plain(self.y, k)).abs().max().item(), 1e-12, f"K={k}")

    def test_small_exit_ignores_later_stages_heads_and_mixers(self):
        model = make_nuse(self.stats, e80_arch((2, 1, 1), 8, impl=IMPL)).double().eval()
        randomize([model])
        with torch.no_grad():
            for g, k in enumerate(model.budgets[:-1]):
                before = model(self.y, k).clone()
                randomize([*model.stages[g + 1:], *model.heads[g + 1:], *model.mixers[g + 1:]], seed=g + 1)
                self.assertTrue(torch.equal(before, model(self.y, k)), f"K={k}")

    def test_params_match_formula_and_cap(self):
        blk = ResBlock.inference_params(32, "plain")
        mixers = [0, 64 * 32 + 32, 64 * 32 + 32 + 96 * 32 + 32]          # Conv1x1 (2W → W) ở exit 2, (3W → W) ở exit 3
        for blocks, kmax_params in (((5, 1, 1), 137187), ((5, 2, 1), 155683)):
            model = make_nuse(self.stats, e80_arch(blocks, impl=IMPL))
            plain = NuSEE(self.stats, e80_arch(blocks))
            got = [model.params_for(k) for k in model.budgets]
            self.assertEqual(got, [plain.params_for(k) + m for k, m in zip(plain.budgets, mixers)])
            self.assertEqual(got[-1], kmax_params)
            self.assertEqual(got[-1], sum(p.numel() for p in model.parameters()))
            self.assertLessEqual(got[-1], PARAM_CAP)
            self.assertTrue(all(b - a >= blk for a, b in zip(got, got[1:])))   # đơn điệu, mỗi exit thêm >= 1 block

    def test_training_step_reaches_mixers(self):
        model = make_nuse(self.stats, e80_arch((2, 1, 1), 8, impl=IMPL))
        hr = torch.rand(2, 3, 32, 32)
        loss = model.loss(degrade(hr, "average_pool"), hr)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(m.weight.grad is not None for m in model.mixers[1:]))


if __name__ == "__main__":
    unittest.main()

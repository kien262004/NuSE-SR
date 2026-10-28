"""Dòng lệnh: python -m nuse_e.pipeline <bước> [<bước> ...] [tuỳ chọn]. Xem nuse_e/pipeline/__init__.py."""
import argparse
import time

from . import __doc__ as PIPELINE_DOC
from .checks import step_screen, step_test
from .context import Ctx
from .evaluation import step_cost, step_evaluate
from .figures import step_curves, step_figures
from .prepare import step_setup, step_stats
from .report import step_report
from .selection import step_select
from .training import step_train

STEPS = {"setup": step_setup, "stats": step_stats, "test": step_test, "screen": step_screen, "train": step_train,
         "curves": step_curves, "select": step_select, "evaluate": step_evaluate, "cost": step_cost,
         "report": step_report, "figures": step_figures}


def main(argv=None):
    parser = argparse.ArgumentParser(description=PIPELINE_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=[*STEPS, "all"], help="một hoặc nhiều bước, chạy theo thứ tự đã ghi")
    parser.add_argument("--out", default="outputs/nuse_e", help="thư mục output (mặc định outputs/nuse_e)")
    parser.add_argument("--data-root", help="thư mục để tự tìm DIV2K_train_HR / DIV2K_valid_HR")
    parser.add_argument("--train-hr", help="thư mục DIV2K_train_HR")
    parser.add_argument("--val-hr", help="thư mục DIV2K_valid_HR")
    parser.add_argument("--old-runs", help="thư mục chứa checkpoint cũ (spanv2_x4/best.pt, disp_nuse_x4_k8/best.pt, ...)")
    parser.add_argument("--benchmarks", help="thư mục chứa Set5, Set14, B100, Urban100, Manga109 (ảnh HR)")
    parser.add_argument("--cache", help="thư mục cache ảnh (mặc định <out>/cache)")
    parser.add_argument("--resume-from", help="bước setup: chép stats/ và runs/ từ output phiên trước")
    parser.add_argument("--variants", help="chỉ dùng các biến thể này, ví dụ e80,d8 (mặc định: mọi biến thể trong config.py)")
    parser.add_argument("--device", help="cuda hoặc cpu (mặc định: cuda nếu có)")
    parser.add_argument("--gpus", help="bước train: danh sách GPU, ví dụ 0,1 (mỗi GPU một run)")
    parser.add_argument("--smoke", action="store_true", help="chạy thử cực ngắn để kiểm code")
    args = parser.parse_args(argv)
    ctx = Ctx(args)
    steps = list(STEPS) if "all" in args.steps else args.steps
    for name in steps:
        print(f"\n==================== {name} ====================", flush=True)
        t0 = time.time()
        STEPS[name](ctx)
        print(f"[{name}] xong sau {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

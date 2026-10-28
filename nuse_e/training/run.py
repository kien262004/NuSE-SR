"""Train một run trong một tiến trình riêng (một GPU).

Dùng: python -m nuse_e.training.run <run_dir>/config.json
Bước `train` của pipeline gọi lệnh này cho từng run; có thể gọi tay để chạy lại một run.
"""
import json
import sys
import time
from pathlib import Path

from .trainer import train


def main(config_path):
    cfg = json.loads(Path(config_path).read_text())
    log_path = Path(cfg["run_dir"]) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a") as stream:
            stream.write(line + "\n")

    log(f"start {cfg['model']} / {cfg['method']} on {cfg['device']}")
    train(cfg, log)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1])

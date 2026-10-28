"""Cache ảnh train: giải mã một lần vào một file uint8 phẳng, crop đọc thẳng từ memmap."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

def _decode_flat(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes(), image.size[1], image.size[0]


def build_flat_cache(files, path, workers=4):
    """Decode every training image once into <path> (raw HWC uint8, back to back) + <path>.json index."""
    from concurrent.futures import ProcessPoolExecutor
    path, index_path = Path(path), Path(str(path) + ".json")
    files = [Path(f) for f in files]
    if index_path.exists() and path.exists():
        index = json.loads(index_path.read_text())
        if [e["name"] for e in index] == [f.stem for f in files]:
            return path
    index, offset = [], 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out, ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for f, (raw, h, w) in zip(files, pool.map(_decode_flat, [str(f) for f in files], chunksize=4)):
            out.write(raw)
            index.append({"name": f.stem, "offset": offset, "h": h, "w": w})
            offset += len(raw)
    index_path.write_text(json.dumps(index))
    return path


class FlatImages:
    def __init__(self, path):
        self.index = json.loads(Path(str(path) + ".json").read_text())
        self.data = np.memmap(str(path), dtype=np.uint8, mode="r")

    def __len__(self):
        return len(self.index)

    def size(self, i):
        e = self.index[i]
        return e["h"], e["w"]

    def crop(self, i, top, left, c):
        e = self.index[i]
        view = self.data[e["offset"]:e["offset"] + e["h"] * e["w"] * 3].reshape(e["h"], e["w"], 3)
        return torch.from_numpy(np.ascontiguousarray(view[top:top + c, left:left + c])).permute(2, 0, 1)

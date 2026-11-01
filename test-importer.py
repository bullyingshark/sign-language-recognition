import tempfile
from pathlib import Path

import numpy as np

from slr.config import Config
from slr.data.importers.base import BaseImporter, RawSample


class FakeImporter(BaseImporter):
    def samples(self):
        rng = np.random.default_rng(0)
        for i in range(2):
            pose = rng.random((47, 33, 3)).astype(np.float32)
            pose[:, 11] = [0.6, 0.5, 0.0]    # shoulders, so normalization works
            pose[:, 12] = [0.4, 0.5, 0.0]
            yield RawSample("hello", "ivan_k", f"s_{i}",
                            np.zeros((47, 21, 3), np.float32),     # left hand never detected
                            rng.random((47, 21, 3)).astype(np.float32),
                            pose)


with tempfile.TemporaryDirectory() as tmp:
    imp = FakeImporter(Config(data_dir=Path(tmp)))
    print("saved:", imp.run())
    print("second run (resume):", imp.run())
    files = sorted(Path(tmp).glob("*/*.npy"))
    print([f.relative_to(tmp).as_posix() for f in files])
    arr = np.load(files[0])
    print(arr.shape, "| left hand all zeros:", not arr[:, :63].any())
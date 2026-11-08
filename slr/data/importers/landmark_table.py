"""Importer for long-format landmark tables (Kaggle asl-signs style CSV)."""
from pathlib import Path, PurePosixPath
from typing import Iterator

import numpy as np
import pandas as pd

from slr.config import Config
from slr.data.importers.base import BaseImporter, RawSample
from slr.keypoints.features import FeatureBuilder


class LandmarkTableImporter(BaseImporter):
    """Imports a long-format landmark CSV (one row per landmark per frame).

    Expected columns: sequence_id, sign, path, frame, type, landmark_index, x, y, z.
    The signer id is taken from `path` (".../<signer>/<sequence>.parquet").
    """

    # integer codes are much cheaper in memory than the "left_hand"/... strings
    _TYPE_CODES = {"left_hand": 0, "right_hand": 1, "pose": 2}

    def __init__(self, config: Config, csv_path: Path, chunksize: int = 1_000_000) -> None:
        super().__init__(config)
        self._csv = Path(csv_path)
        self._chunksize = chunksize

    def samples(self) -> Iterator[RawSample]:
        table, meta = self._load()
        for seq_id, group in table.groupby("sequence_id", sort=False):
            yield self._build_sample(seq_id, group, meta.loc[seq_id])

    def _load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read the CSV in chunks. Returns (landmarks without face, per-sequence metadata)."""
        columns = ["sequence_id", "sign", "path", "frame", "type", "landmark_index", "x", "y", "z"]
        parts, metas = [], []
        for chunk in pd.read_csv(self._csv, usecols=columns, chunksize=self._chunksize):
            metas.append(chunk[["sequence_id", "sign", "path"]].drop_duplicates())
            chunk = chunk[chunk["type"] != "face"]                       # the face is not used
            chunk = chunk.assign(type=chunk["type"].map(self._TYPE_CODES).astype("int8"))
            parts.append(chunk[["sequence_id", "frame", "type", "landmark_index", "x", "y", "z"]])
        meta = pd.concat(metas).drop_duplicates("sequence_id").set_index("sequence_id")
        table = pd.concat(parts, ignore_index=True)
        return table, meta

    def _build_sample(self, seq_id: int, group: pd.DataFrame, info: pd.Series) -> RawSample:
        """Rows of one sequence -> RawSample with (frames, points, 3) arrays."""
        frames = np.sort(group["frame"].unique())
        frame_idx = np.searchsorted(frames, group["frame"].to_numpy())   # frame number -> 0..n-1
        types = group["type"].to_numpy()
        points = group["landmark_index"].to_numpy()
        coords = np.nan_to_num(group[["x", "y", "z"]].to_numpy(dtype=np.float32))  # NaN -> 0 = "not detected"

        def part(type_code: int, n_points: int) -> np.ndarray:
            arr = np.zeros((len(frames), n_points, 3), dtype=np.float32)
            mask = types == type_code
            arr[frame_idx[mask], points[mask]] = coords[mask]
            return arr

        return RawSample(
            label=str(info["sign"]),
            signer=PurePosixPath(info["path"]).parts[-2],        # "train_landmark_files/<signer>/<id>.parquet"
            sample_id=str(seq_id),
            left=part(self._TYPE_CODES["left_hand"], FeatureBuilder.HAND_POINTS),
            right=part(self._TYPE_CODES["right_hand"], FeatureBuilder.HAND_POINTS),
            pose=part(self._TYPE_CODES["pose"], FeatureBuilder.POSE_POINTS),
        )
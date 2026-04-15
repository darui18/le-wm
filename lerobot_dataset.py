from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class LeRobotDatasetConfig:
    repo_id: str
    split: str = "train"
    cache_dir: str | None = None
    num_steps: int = 4
    frameskip: int = 1
    episode_key: str = "episode_index"
    frame_key: str = "frame_index"
    key_mapping: dict[str, str] | None = None


class LeRobotSequenceDataset(Dataset):
    """Sequence sampler for LeRobot/HF datasets.

    It converts frame-level trajectories into fixed-length sequences compatible
    with LeWM training.
    """

    def __init__(self, config: LeRobotDatasetConfig, transform=None):
        super().__init__()
        self.config = config
        self.transform = transform

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "`datasets` is required for LeRobot format. Install with `uv pip install datasets`."
            ) from exc

        self.dataset = load_dataset(
            path=config.repo_id,
            split=config.split,
            cache_dir=config.cache_dir,
        )

        self.key_mapping = config.key_mapping or {
            "pixels": "observation.images.top",
            "action": "action",
            "proprio": "observation.state",
            "state": "observation.state",
        }

        self.column_names = list(self.key_mapping.keys())

        episode_idx = np.asarray(self.dataset[config.episode_key])
        frame_idx = np.asarray(self.dataset[config.frame_key])

        self._valid_starts = self._build_valid_starts(episode_idx, frame_idx)

    def _build_valid_starts(self, episode_idx: np.ndarray, frame_idx: np.ndarray) -> list[int]:
        valid_starts: list[int] = []
        n = len(episode_idx)
        stride = self.config.frameskip
        req_last_offset = (self.config.num_steps - 1) * stride

        i = 0
        while i < n:
            ep = episode_idx[i]
            j = i
            while j < n and episode_idx[j] == ep:
                j += 1

            ep_frames = frame_idx[i:j]
            ep_len = len(ep_frames)
            max_start = ep_len - req_last_offset
            if max_start > 0:
                valid_starts.extend(range(i, i + max_start))
            i = j

        return valid_starts

    def _extract_value(self, row: dict[str, Any], mapped_key: str) -> Any:
        value: Any = row
        for part in mapped_key.split("."):
            value = value[part]
        return value

    def _to_numpy(self, value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def __len__(self) -> int:
        return len(self._valid_starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self._valid_starts[idx]
        steps = [start + t * self.config.frameskip for t in range(self.config.num_steps)]

        sample: dict[str, torch.Tensor] = {}
        for target_key, source_key in self.key_mapping.items():
            seq = []
            for row_idx in steps:
                row = self.dataset[int(row_idx)]
                value = self._extract_value(row, source_key)
                seq.append(self._to_numpy(value))

            seq_arr = np.stack(seq)
            sample[target_key] = torch.from_numpy(seq_arr)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_col_data(self, key: str) -> np.ndarray:
        source_key = self.key_mapping[key]
        out = []
        for i in range(len(self.dataset)):
            row = self.dataset[i]
            value = self._extract_value(row, source_key)
            out.append(self._to_numpy(value))
        return np.asarray(out)

    def get_dim(self, key: str) -> int:
        col_data = self.get_col_data(key)
        return int(col_data.shape[-1])

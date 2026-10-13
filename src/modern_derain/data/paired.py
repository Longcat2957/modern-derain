"""Torch conversion lives only in the supervised training input pipeline."""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import build_augmentation, get_rng_state, restore_rng_state
from .images import read_rgb, validate_pair
from .sources import Sample


class PairedDataset(Dataset):
    def __init__(self, samples: list[Sample], crop_size: int = 256, seed: int = 0):
        if not samples or any(sample.clean is None for sample in samples):
            raise ValueError("Supervised training requires rainy/clean pairs")
        self.samples = list(samples)
        self.transform = build_augmentation(crop_size, seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        rainy, clean = read_rgb(sample.rainy), read_rgb(sample.clean)
        validate_pair(rainy, clean)
        result = self.transform(image=rainy, clean=clean)
        return tuple(
            torch.from_numpy(
                np.ascontiguousarray(result[key].transpose(2, 0, 1), dtype=np.float32) / 255.0
            )
            for key in ("image", "clean")
        )

    def get_augmentation_state(self) -> list[dict]:
        return get_rng_state(self.transform)

    def restore_augmentation_state(self, state: list[dict]) -> None:
        restore_rng_state(self.transform, state)


def worker_init_fn(worker_id: int) -> None:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

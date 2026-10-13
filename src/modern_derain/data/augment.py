"""One shared uint8 crop, preserving the upstream inclusive coordinate range."""

import copy
import random

import albumentations as A
import numpy as np
from albumentations.core.transform_params import SampledParams


class InclusiveCrop(A.RandomCrop):
    def sample_parameters(self, params, data, targets, sampling):
        height, width = targets.require_aligned_spatial_shape(2)
        if height < self.height or width < self.width:
            raise ValueError(
                f"Pair is smaller than crop {self.height}x{self.width}: {height}x{width}"
            )
        top = sampling.py_random.randint(0, height - self.height)
        left = sampling.py_random.randint(0, width - self.width)
        return SampledParams(
            params={
                "crop_coords": (left, top, left + self.width, top + self.height),
                "pad_params": None,
            }
        )


def build_augmentation(crop_size: int, seed: int) -> A.Compose:
    if type(crop_size) is not int or crop_size < 16 or crop_size % 16:
        raise ValueError("crop_size must be at least 16 and divisible by 16")
    return A.Compose(
        [InclusiveCrop(height=crop_size, width=crop_size, p=1)],
        additional_targets={"clean": "image"},
        seed=seed,
    )


def get_rng_state(compose: A.Compose) -> list[dict]:
    states = []
    for transform in [compose, *compose.transforms]:
        states.append(
            {
                "python": transform.py_random.getstate(),
                "numpy": copy.deepcopy(transform.random_generator.bit_generator.state),
            }
        )
    return states


def restore_rng_state(compose: A.Compose, states: list[dict]) -> None:
    transforms = [compose, *compose.transforms]
    if len(states) != len(transforms):
        raise ValueError("Augmentation RNG state does not match the configured pipeline")
    for transform, state in zip(transforms, states, strict=True):
        py_random = random.Random()
        py_random.setstate(state["python"])
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["numpy"])
        transform.set_random_state(rng, py_random)

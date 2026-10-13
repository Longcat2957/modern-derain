"""EfficientDeRain imagewise metrics on recovered BGR float arrays."""

import math

import numpy as np


def _validate(pred: np.ndarray, target: np.ndarray) -> None:
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError("Metrics require matching HWC three-channel images")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Metrics require finite images")


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    _validate(pred, target)
    pred, target = pred.astype(np.float32), target.astype(np.float32)
    mse = float(np.mean((pred - target) ** 2))
    return 100.0 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    _validate(pred, target)
    if min(pred.shape[:2]) < 7:
        raise ValueError("Official SSIM requires image height and width >= 7")
    return float(
        structural_similarity(
            target.astype(np.float64),
            pred.astype(np.float64),
            channel_axis=-1,
            data_range=255,
            win_size=7,
            gaussian_weights=False,
            use_sample_covariance=True,
            K1=0.01,
            K2=0.03,
        )
    )

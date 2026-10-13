import math

import numpy as np
import pytest
from skimage.metrics import structural_similarity

from modern_derain.evaluation.metrics import psnr, ssim


def test_official_psnr_zero_and_whole_image_mse():
    target = np.zeros((9, 11, 3), np.float32)
    pred = np.full_like(target, 10)
    assert psnr(target, target) == 100
    assert psnr(pred, target) == pytest.approx(20 * math.log10(25.5))
    pred[..., 0] = 20
    assert psnr(pred, target) == pytest.approx(20 * math.log10(255 / math.sqrt(200)))


def test_ssim_float64_channelwise_017_protocol():
    rng = np.random.default_rng(123)
    target = rng.uniform(0, 255, (19, 23, 3)).astype(np.float32)
    pred = np.clip(target + rng.normal(0, 10, target.shape), 0, 255).astype(np.float32)
    expected = (
        sum(
            structural_similarity(
                target[..., i].astype(np.float64),
                pred[..., i].astype(np.float64),
                data_range=255,
                win_size=7,
                gaussian_weights=False,
                use_sample_covariance=True,
                K1=0.01,
                K2=0.03,
            )
            for i in range(3)
        )
        / 3
    )
    assert ssim(pred, target) == expected
    assert ssim(pred, target) == pytest.approx(0.9906500829043292, abs=1e-14)
    assert ssim(target, target) == 1
    with pytest.raises(ValueError, match="height"):
        ssim(target[:5], target[:5])
    with pytest.raises(ValueError, match="matching"):
        psnr(target[:5], target)

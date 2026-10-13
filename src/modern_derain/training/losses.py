"""Active EfficientDeRain training loss; distinct from evaluation SSIM."""

from math import exp

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SSIM(nn.Module):
    """11×11 Gaussian SSIM with zero padding and population moments."""

    def __init__(self):
        super().__init__()
        gaussian = torch.tensor(
            [exp(-((position - 5) ** 2) / (2 * 1.5**2)) for position in range(11)]
        )
        gaussian = gaussian / gaussian.sum()
        window = gaussian[:, None].mm(gaussian[None, :])[None, None]
        self.register_buffer("window", window, persistent=False)

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        if first.shape != second.shape or first.ndim != 4:
            raise ValueError("training SSIM requires matching NCHW images")
        with torch.autocast(device_type=first.device.type, enabled=False):
            if first.dtype in (torch.float16, torch.bfloat16):
                first = first.float()
            if second.dtype in (torch.float16, torch.bfloat16):
                second = second.float()
            channels = first.shape[1]
            window = self.window.to(first).expand(channels, 1, 11, 11).contiguous()
            mean_first = F.conv2d(first, window, padding=5, groups=channels)
            mean_second = F.conv2d(second, window, padding=5, groups=channels)
            square_first, square_second = mean_first.pow(2), mean_second.pow(2)
            product = mean_first * mean_second
            variance_first = (
                F.conv2d(first * first, window, padding=5, groups=channels) - square_first
            )
            variance_second = (
                F.conv2d(second * second, window, padding=5, groups=channels) - square_second
            )
            covariance = F.conv2d(first * second, window, padding=5, groups=channels) - product
            score = ((2 * product + 0.01**2) * (2 * covariance + 0.03**2)) / (
                (square_first + square_second + 0.01**2)
                * (variance_first + variance_second + 0.03**2)
            )
            return score.mean()


class DerainLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM()

    def forward(self, prediction: Tensor, clean: Tensor) -> Tensor:
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            if prediction.dtype in (torch.float16, torch.bfloat16):
                prediction = prediction.float()
            if clean.dtype in (torch.float16, torch.bfloat16):
                clean = clean.float()
            return F.l1_loss(prediction, clean) - 0.2 * self.ssim(clean, prediction)

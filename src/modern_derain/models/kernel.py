"""RGB dynamic filtering with the pinned EfficientDeRain core ordering."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class KernelConv(nn.Module):
    """One 3×3 core shared by four pixel-wise dilation filters."""

    def __init__(self, kernel_size=(3,), sep_conv=False, core_bias=False):
        super().__init__()
        if list(kernel_size) != [3] or sep_conv or core_bias:
            raise ValueError(
                "v3 Rain1400 requires kernel_size=[3], sep_conv=False, core_bias=False"
            )

    def forward(self, frames: Tensor, core: Tensor, white_level=1.0, rate=1) -> Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("KernelConv expects RGB NCHW frames")
        batch, _, height, width = frames.shape
        if core.shape != (batch, 27, height, width):
            raise ValueError("KernelConv core must have shape [B,27,H,W]")
        if rate not in (1, 2, 3, 4) or white_level <= 0:
            raise ValueError("dilation must be 1..4 and white_level must be positive")
        # FP32 accumulation keeps dynamic kernels finite under CUDA autocast.
        if frames.dtype in (torch.float16, torch.bfloat16):
            frames = frames.float()
        if core.dtype in (torch.float16, torch.bfloat16):
            core = core.float()
        frames = frames.reshape(batch, 3, 1, height, width)
        weights = core.reshape(batch, 3, 9, 1, height, width)
        padded = F.pad(frames, (rate, rate, rate, rate))
        neighborhoods = torch.stack(
            [
                padded[..., row * rate : row * rate + height, col * rate : col * rate + width]
                for row in range(3)
                for col in range(3)
            ],
            dim=2,
        )
        prediction = (weights * neighborhoods).sum(dim=2)
        prediction = torch.stack((prediction,), dim=0).mean(dim=0)
        return prediction.squeeze(2) / white_level

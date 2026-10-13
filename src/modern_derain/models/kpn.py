"""Checkpoint-compatible v3 Rain1400 KPN, rebuilt using standard PyTorch."""

import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .kernel import KernelConv

UPSTREAM_COMMIT = "465759da2428a4263eeee6ab464a0135fca2900c"
UPSTREAM_CHECKPOINT_SHA256 = "b8aaa9b5cd01177ca22a5344813923f91f298515be6382cf8654799cab5fa676"
MODEL_CONFIG = {
    "color": True,
    "burst_length": 1,
    "blind_est": True,
    "kernel_size": [3],
    "sep_conv": False,
    "channel_att": False,
    "spatial_att": False,
    "upMode": "bilinear",
    "core_bias": False,
}


class Basic(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        layers = []
        for channels in (in_channels, out_channels, out_channels):
            layers.extend((nn.Conv2d(channels, out_channels, 3, padding=1), nn.ReLU()))
        self.conv1 = nn.Sequential(*layers)

    def forward(self, image: Tensor) -> Tensor:
        return self.conv1(image)


class KPN(nn.Module):
    """Single RGB input; the CNN and dynamic-filter inputs retain separate roles."""

    def __init__(
        self,
        color=True,
        burst_length=1,
        blind_est=True,
        kernel_size=(3,),
        sep_conv=False,
        channel_att=False,
        spatial_att=False,
        upMode="bilinear",
        core_bias=False,
    ):
        super().__init__()
        supplied = {
            "color": color,
            "burst_length": burst_length,
            "blind_est": blind_est,
            "kernel_size": list(kernel_size),
            "sep_conv": sep_conv,
            "channel_att": channel_att,
            "spatial_att": spatial_att,
            "upMode": upMode,
            "core_bias": core_bias,
        }
        if supplied != MODEL_CONFIG:
            raise ValueError("only the official v3 Rain1400 KPN configuration is supported")
        for index, (inputs, outputs) in enumerate(
            (
                (3, 64),
                (64, 128),
                (128, 256),
                (256, 512),
                (512, 512),
                (1024, 512),
                (768, 256),
                (384, 27),
            ),
            start=1,
        ):
            setattr(self, f"conv{index}", Basic(inputs, outputs))
        self.outc = nn.Conv2d(27, 27, 1)
        self.kernel_pred = KernelConv()
        self.conv_final = nn.Conv2d(12, 3, 3, padding=1)

    @staticmethod
    def _upsample(image: Tensor) -> Tensor:
        return F.interpolate(
            image, scale_factor=2, mode="bilinear", align_corners=False, antialias=False
        )

    def forward(self, rainy: Tensor) -> Tensor:
        return self.forward_pair(rainy, rainy, white_level=1.0)

    def forward_pair(self, data_with_est: Tensor, data: Tensor, white_level=1.0) -> Tensor:
        if data.ndim != 4 or data.shape[1] != 3 or data_with_est.shape != data.shape:
            raise ValueError("KPN expects matching RGB NCHW inputs")
        if min(data.shape[-2:]) < 16 or any(size % 16 for size in data.shape[-2:]):
            raise ValueError("KPN height/width must be positive multiples of 16, minimum 16")
        first = self.conv1(data_with_est)
        second = self.conv2(F.avg_pool2d(first, 2, 2))
        third = self.conv3(F.avg_pool2d(second, 2, 2))
        fourth = self.conv4(F.avg_pool2d(third, 2, 2))
        fifth = self.conv5(F.avg_pool2d(fourth, 2, 2))
        sixth = self.conv6(torch.cat((fourth, self._upsample(fifth)), dim=1))
        seventh = self.conv7(torch.cat((third, self._upsample(sixth)), dim=1))
        eighth = self.conv8(torch.cat((second, self._upsample(seventh)), dim=1))
        core = self.outc(self._upsample(eighth))
        filtered = [self.kernel_pred(data, core, white_level, rate) for rate in (1, 2, 3, 4)]
        combined = filtered[0]
        for image in filtered[1:]:
            combined = torch.cat((combined, image), dim=1)
        return self.conv_final(combined)


def load_checkpoint(path, config_model=None, preprocessing=None, device="cpu"):
    """Load the observed raw upstream dict or our envelope; never remap keys."""
    from .assets import resolve_checkpoint

    path = resolve_checkpoint(Path(path))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("checkpoint must contain a nonempty state dictionary or envelope")
    if all(isinstance(value, Tensor) for value in payload.values()):
        payload = {"kind": "fp32", "state_dict": payload, "provenance": {"format": "raw-upstream"}}
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("checkpoint envelope is missing state_dict")
    metadata = payload.get("config", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint config metadata must be an object")
    saved_model = metadata.get("model", MODEL_CONFIG)
    if (
        saved_model != MODEL_CONFIG
        or any(type(saved_model[key]) is not type(value) for key, value in MODEL_CONFIG.items())
        or any(type(value) is not int for value in saved_model["kernel_size"])
        or (config_model is not None and saved_model != config_model)
    ):
        raise ValueError(
            "checkpoint model configuration differs from the requested v3 Rain1400 model"
        )
    saved_preprocessing = metadata.get("preprocessing", payload.get("preprocessing"))
    if saved_preprocessing is not None and preprocessing is not None:
        expected = preprocessing
        actual = saved_preprocessing
        if isinstance(expected, str) and isinstance(actual, dict):
            actual = actual.get("protocol")
        if actual != expected:
            raise ValueError("checkpoint preprocessing differs from the requested protocol")
    model = KPN(**saved_model)
    if payload.get("kind", "fp32") in ("ptq", "qat"):
        if "modelopt_state" not in payload:
            raise ValueError("quantized checkpoint lacks ModelOpt structure and quantizer state")
        import modelopt.torch.opt as mto
        from modelopt.torch.quantization.nn import TensorQuantizer

        from .quantized import INT8_BACKEND, register_int8_backend

        register_int8_backend()

        with redirect_stdout(sys.stderr):
            model = mto.restore_from_modelopt_state(model, modelopt_state=payload["modelopt_state"])
        model.load_state_dict(payload["state_dict"], strict=True)
        enabled = 0
        for name, quantizer in model.named_modules():
            if not isinstance(quantizer, TensorQuantizer) or not quantizer.is_enabled:
                continue
            enabled += 1
            if quantizer.backend != INT8_BACKEND:
                raise ValueError(
                    f"{name} uses an incompatible INT8 arithmetic policy; recalibrate from FP32"
                )
            scale = quantizer.amax
            if quantizer.num_bits != 8 or quantizer.unsigned:
                raise ValueError(f"{name} is not symmetric signed INT8")
            if (
                (name.endswith("weight_quantizer") and quantizer.axis != 0)
                or (name.endswith("input_quantizer") and quantizer.axis is not None)
                or not name.endswith(("weight_quantizer", "input_quantizer"))
            ):
                raise ValueError(f"{name} violates the Conv-only INT8 axis policy")
            if scale is None or not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError(f"{name} lacks a valid calibrated scale")
        if not enabled:
            raise ValueError("quantized checkpoint has no enabled calibrated quantizers")
    else:
        if payload.get("kind", "fp32") != "fp32":
            raise ValueError("unknown checkpoint kind")
        if any(
            not isinstance(value, Tensor) or value.dtype != torch.float32
            for value in payload["state_dict"].values()
        ):
            raise ValueError(
                "FP32 checkpoint tensors must be torch.float32; implicit casts are refused"
            )
        model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), payload


def package_checkpoint(source, destination, preprocessing):
    """Wrap original FP32 tensors without changing names, values or precision."""
    from modern_derain.artifacts import sha256

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("checkpoint packaging destination must be a new file")
    if sha256(source) != UPSTREAM_CHECKPOINT_SHA256:
        raise ValueError("source does not match the verified official v3 Rain1400 checkpoint")
    state = torch.load(source, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(value, Tensor) and value.dtype == torch.float32 for value in state.values()
        )
    ):
        raise ValueError("packaging requires the original raw FP32 state dictionary")
    original = KPN()
    original.load_state_dict(state, strict=True)
    payload = {
        "format_version": 1,
        "kind": "fp32",
        "state_dict": state,
        "config": {"model": MODEL_CONFIG, "preprocessing": preprocessing},
        "provenance": {
            "upstream_commit": UPSTREAM_COMMIT,
            "source_url": "http://www.xujuefei.com/models_effderain.zip",
            "retrieved_url": "https://www.xujuefei.com/models_effderain.zip",
            "archive_member": "models/v3_rain1400/v3_rain1400.pth",
            "source_sha256": sha256(source),
            "source_size": source.stat().st_size,
            "conversion": "FP32 raw state_dict envelope only; every tensor unchanged",
            "weight_license": "no separate checkpoint license verified",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            torch.save(payload, output)
        restored = torch.load(temporary, map_location="cpu", weights_only=True)
        for key, value in state.items():
            actual = restored["state_dict"][key]
            if actual.dtype != value.dtype or not torch.equal(value, actual):
                raise ValueError(f"packaged checkpoint changed tensor {key}")
        converted = KPN()
        converted.load_state_dict(restored["state_dict"], strict=True)
        original.eval()
        converted.eval()
        sample = torch.linspace(0, 1, 3 * 16 * 32).reshape(1, 3, 16, 32)
        with torch.inference_mode():
            before, after = original(sample), converted(sample)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return payload

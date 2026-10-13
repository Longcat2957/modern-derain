"""Small JSON configuration contract; paths are relative to the caller's CWD."""

import copy
import json
import math
from pathlib import Path

DEFAULT_CHECKPOINT = "weights/efficientderain-rain1400-v3.pt"
DEFAULT_PREPROCESSING = {
    "protocol": "official-ceil16",
    "color": "RGB",
    "range": [0.0, 1.0],
    "interpolation": "opencv-linear",
    "recovery": "clip-truncate-uint8-bgr-resize",
}
DEFAULTS = {
    "model": {
        "color": True,
        "burst_length": 1,
        "blind_est": True,
        "kernel_size": [3],
        "sep_conv": False,
        "channel_att": False,
        "spatial_att": False,
        "upMode": "bilinear",
        "core_bias": False,
    },
    "data": {
        "path": "data/paired",
        "layout": "paired",
        "train": "train",
        "val": "val",
        "test": "test",
        "calibration": "calibration",
    },
    "training": {
        "checkpoint": DEFAULT_CHECKPOINT,
        "resume": None,
        "device": "cuda",
        "epochs": 250,
        "batch_size": 16,
        "lr": 0.0002,
        "betas": [0.5, 0.999],
        "weight_decay": 0.0,
        "decay_start_epoch": 50,
        "crop_size": 256,
        "seed": 1234,
        "num_workers": 0,
        "persistent_workers": False,
        "drop_last": False,
        "amp": False,
        "output_dir": "artifacts/train",
    },
    "quantization": {
        "steps": 0,
        "calibration_samples": 32,
        "exclude_modules": ["outc", "conv_final"],
    },
    "export": {
        "checkpoint": DEFAULT_CHECKPOINT,
        "shape": None,
        "opset": 18,
        "output": "artifacts/export/model.onnx",
        "precision": "fp32",
        "device": "cpu",
    },
    "inference": {
        "checkpoint": DEFAULT_CHECKPOINT,
        "device": "cuda",
        "protocol": "official-ceil16",
        "input": None,
        "output": "artifacts/infer",
    },
    "evaluation": {
        "checkpoint": DEFAULT_CHECKPOINT,
        "device": "cuda",
        "protocol": "official-ceil16",
        "split": "val",
        "output": "artifacts/eval/results.json",
        "max_psnr_drop_db": 0.3,
    },
    "benchmark": {"shape": None, "target_fps": None, "target_p95_ms": None, "power_mode": None},
}


def deep_merge(base: dict, update: dict, *, skip_none: bool = False) -> dict:
    result = copy.deepcopy(base)
    if not isinstance(update, dict):
        raise ValueError("Configuration must be a JSON object")
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {key}")
        if skip_none and value is None:
            continue
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            result[key] = deep_merge(base[key], value, skip_none=skip_none)
        else:
            result[key] = value
    return result


def _path(value, name, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")


def _number(value, name, minimum, *, integer=False):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be {'an integer' if integer else 'a number'} >= {minimum}")


def validate_shape(shape, *, required=False):
    if shape is None and not required:
        return
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(type(v) is not int for v in shape)
        or shape[:2] != [1, 3]
        or any(v < 16 or v % 16 for v in shape[2:])
    ):
        raise ValueError("shape must be [1, 3, H, W], with H/W >= 16 and divisible by 16")


def validate(config: dict):
    deep_merge(DEFAULTS, config)
    if (
        config["model"] != DEFAULTS["model"]
        or any(
            type(config["model"][key]) is not type(value)
            for key, value in DEFAULTS["model"].items()
        )
        or any(type(value) is not int for value in config["model"]["kernel_size"])
    ):
        raise ValueError("Only the official v3 Rain1400 KPN model configuration is supported")
    data = config["data"]
    for key in ("path", "train", "val", "test", "calibration"):
        _path(data[key], f"data.{key}")
    if data["layout"] not in ("paired", "rain1400"):
        raise ValueError("data.layout must be paired or rain1400")
    training = config["training"]
    for section in ("training", "export", "inference", "evaluation"):
        _path(
            config[section]["checkpoint"], f"{section}.checkpoint", nullable=section == "training"
        )
        if config[section]["device"] not in ("cpu", "cuda"):
            raise ValueError(f"{section}.device must be cpu or cuda")
    _path(training["resume"], "training.resume", nullable=True)
    _path(training["output_dir"], "training.output_dir")
    if training["checkpoint"] is not None and training["resume"] is not None:
        raise ValueError("--checkpoint and --resume are mutually exclusive")
    if training["checkpoint"] is None and training["resume"] is None:
        raise ValueError("training requires a checkpoint or resume path")
    for key in ("epochs", "batch_size", "crop_size"):
        _number(training[key], f"training.{key}", 1, integer=True)
    if training["crop_size"] < 16 or training["crop_size"] % 16:
        raise ValueError("training.crop_size must be >= 16 and divisible by 16")
    for key in ("seed", "num_workers", "decay_start_epoch"):
        _number(training[key], f"training.{key}", 0, integer=True)
    for key in ("amp", "drop_last", "persistent_workers"):
        if type(training[key]) is not bool:
            raise ValueError(f"training.{key} must be boolean")
    if training["persistent_workers"] and not training["num_workers"]:
        raise ValueError("persistent_workers requires num_workers > 0")
    _number(training["lr"], "training.lr", 0)
    if training["lr"] == 0:
        raise ValueError("training.lr must be > 0")
    _number(training["weight_decay"], "training.weight_decay", 0)
    betas = training["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("training.betas must contain two numbers")
    for value in betas:
        _number(value, "training.betas", 0)
        if value >= 1:
            raise ValueError("training.betas must be < 1")
    quantization = config["quantization"]
    _number(quantization["steps"], "quantization.steps", 0, integer=True)
    _number(
        quantization["calibration_samples"], "quantization.calibration_samples", 1, integer=True
    )
    exclusions = quantization["exclude_modules"]
    if not isinstance(exclusions, list) or any(
        v not in ("outc", "conv8", "conv_final") for v in exclusions
    ):
        raise ValueError("quantization.exclude_modules may contain outc, conv8, conv_final")
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("quantization.exclude_modules must be unique")
    export = config["export"]
    validate_shape(export["shape"])
    _number(export["opset"], "export.opset", 13, integer=True)
    _path(export["output"], "export.output")
    if export["precision"] not in ("fp32", "int8"):
        raise ValueError("export.precision must be fp32 or int8")
    _path(config["inference"]["input"], "inference.input", nullable=True)
    for section in ("inference", "evaluation"):
        _path(config[section]["output"], f"{section}.output")
        if config[section]["protocol"] not in ("official-ceil16", "fixed"):
            raise ValueError(f"{section}.protocol must be official-ceil16 or fixed")
        if config[section]["protocol"] == "fixed":
            validate_shape(export["shape"], required=True)
    if config["evaluation"]["split"] not in ("val", "test"):
        raise ValueError("evaluation.split must be val or test")
    _number(config["evaluation"]["max_psnr_drop_db"], "evaluation.max_psnr_drop_db", 0)
    validate_shape(config["benchmark"]["shape"])
    for key in ("target_fps", "target_p95_ms"):
        value = config["benchmark"][key]
        if value is not None:
            _number(value, f"benchmark.{key}", 0)
            if value == 0:
                raise ValueError(f"benchmark.{key} must be > 0")
    power = config["benchmark"]["power_mode"]
    if power is not None and (not isinstance(power, str) or not power.strip()):
        raise ValueError("benchmark.power_mode must be a nonempty string or null")


def load_config(path: Path | None = None, overrides: dict | None = None) -> dict:
    user = {}
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            user = json.load(handle)
    config = deep_merge(DEFAULTS, user)
    # A single training config can also select the initial inference/export weight.
    initial = user.get("training", {}).get("checkpoint")
    if initial is not None:
        for section in ("export", "inference", "evaluation"):
            if "checkpoint" not in user.get(section, {}):
                config[section]["checkpoint"] = initial
    overrides = overrides or {}
    config = deep_merge(config, overrides, skip_none=True)
    cli_training = {
        key: value for key, value in overrides.get("training", {}).items() if value is not None
    }
    supplied = {**user.get("training", {}), **cli_training}
    if (cli_training.get("resume") is not None and "checkpoint" not in cli_training) or (
        supplied.get("resume") is not None and "checkpoint" not in supplied
    ):
        config["training"]["checkpoint"] = None
    validate(config)
    return config


def resolve_split(config: dict, name: str) -> Path:
    if name not in ("train", "val", "test", "calibration"):
        raise ValueError(f"Unknown split: {name}")
    path = Path(config["data"][name])
    return path if path.is_absolute() else Path(config["data"]["path"]) / path

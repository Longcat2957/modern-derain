"""ModelOpt static INT8 CNN calibration; pixel kernels remain floating point."""

import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy

import torch

from modern_derain.artifacts import create_run_dir, run_metadata, sha256, write_json
from modern_derain.config import DEFAULT_PREPROCESSING, resolve_split
from modern_derain.data.images import prepare_rgb, read_rgb
from modern_derain.data.sources import (
    check_leakage,
    scan_calibration,
    scan_pairs,
    snapshot,
    training_snapshots,
    validate_split_sources,
)
from modern_derain.models.quantized import INT8_BACKEND, register_int8_backend
from modern_derain.training.distributed import DistributedContext, distributed_session
from modern_derain.training.loop import save_checkpoint, train_model


def quantizer_summary(model) -> list[dict]:
    from modelopt.torch.quantization.nn import TensorQuantizer

    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        enabled = module.is_enabled
        amax = module.amax
        if enabled:
            if module.backend != INT8_BACKEND:
                raise ValueError(f"{name} does not use the saved ONNX Q/DQ arithmetic policy")
            if module.num_bits != 8 or module.unsigned:
                raise ValueError(f"{name} is not symmetric signed INT8")
            if (
                (name.endswith("weight_quantizer") and module.axis != 0)
                or (name.endswith("input_quantizer") and module.axis is not None)
                or not name.endswith(("weight_quantizer", "input_quantizer"))
            ):
                raise ValueError(f"{name} violates the Conv-only INT8 axis policy")
            if amax is None or not torch.isfinite(amax).all() or not (amax > 0).all():
                raise ValueError(f"{name} has missing/nonfinite/nonpositive calibration scale")
        rows.append(
            {
                "name": name,
                "enabled": enabled,
                "num_bits": module.num_bits,
                "axis": module.axis,
                "unsigned": module.unsigned,
                "backend": module.backend,
                "amax_shape": list(amax.shape) if amax is not None else None,
                "amax_min": float(amax.min()) if amax is not None else None,
                "amax_max": float(amax.max()) if amax is not None else None,
            }
        )
    if not any(row["enabled"] for row in rows):
        raise ValueError("The model contains no calibrated enabled quantizers")
    return rows


def calibrate_model(model, forward_loop, exclude_modules: list[str]):
    import modelopt.torch.quantization as mtq

    register_int8_backend()

    names = dict(model.named_modules())
    missing = [name for name in exclude_modules if name not in names]
    if missing:
        raise ValueError(f"Unknown quantization exclusions: {missing}")
    recipe = deepcopy(mtq.INT8_DEFAULT_CFG)
    recipe["quant_cfg"].append({"quantizer_name": "*", "enable": False})
    quantized_convolutions = []
    for name, module in names.items():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        if any(name == exclude or name.startswith(exclude + ".") for exclude in exclude_modules):
            continue
        quantized_convolutions.append(name)
        recipe["quant_cfg"].extend(
            [
                {
                    "quantizer_name": f"{name}.input_quantizer",
                    "enable": True,
                    "cfg": {
                        "num_bits": 8,
                        "axis": None,
                        "unsigned": False,
                        "type": "static",
                        "backend": INT8_BACKEND,
                        "narrow_range": False,
                    },
                },
                {
                    "quantizer_name": f"{name}.weight_quantizer",
                    "enable": True,
                    "cfg": {
                        "num_bits": 8,
                        "axis": 0,
                        "unsigned": False,
                        "type": "static",
                        "backend": INT8_BACKEND,
                        "narrow_range": False,
                    },
                },
            ]
        )
    if not quantized_convolutions:
        raise ValueError("No Conv2d layers remain for W8A8 calibration")
    with redirect_stdout(sys.stderr):
        model = mtq.quantize(model.eval(), recipe, forward_loop)
    from modelopt.torch.quantization.nn import TensorQuantizer

    for module in model.modules():
        if isinstance(module, TensorQuantizer):
            module.disable_calib()
    summary = quantizer_summary(model)
    return model, {
        "base": "INT8_DEFAULT_CFG",
        "numeric_backend": INT8_BACKEND,
        "numeric_formula": "round(input/float32(float64(amax)/127)), clip[-128,127], multiply(scale)",
        "arithmetic": "FP32 quantization, original-dtype output for AMP",
        "gradient": "STE when normalized input in [-128,127], fixed scales",
        "signed_range": [-128, 127],
        "weight": "signed-static-per-channel-axis-0",
        "activation": "signed-static-per-tensor",
        "scale_policy": "fixed-after-calibration",
        "quantized_convolutions": quantized_convolutions,
        "excluded_modules": exclude_modules,
        "exclusion_reason": "Conservative floating kernel head/image fusion; quality comparison pending",
        "floating_dynamic_filter": "FP32 multiply/reduce",
        "quantizers": summary,
    }


def restore_quantized_model(model, payload: dict):
    import modelopt.torch.opt as mto

    register_int8_backend()

    if not payload.get("modelopt_state"):
        raise ValueError("Quantized checkpoint is missing ModelOpt architecture state")
    with redirect_stdout(sys.stderr):
        model = mto.restore_from_modelopt_state(model, modelopt_state=payload["modelopt_state"])
    model.load_state_dict(payload["state_dict"], strict=True)
    quantizer_summary(model)
    return model


def run_qat(config: dict) -> dict | None:
    with distributed_session(config["training"]["device"]) as context:
        if context.is_primary:
            return _run_qat(config, context)
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            return _run_qat(config, context)


def _run_qat(config: dict, context: DistributedContext) -> dict | None:
    import modelopt.torch.opt as mto

    from modern_derain.models.assets import resolve_checkpoint
    from modern_derain.models.kpn import load_checkpoint

    training, quantization = config["training"], config["quantization"]
    shape = config["export"]["shape"]
    if shape is None:
        raise ValueError("Calibration/QAT requires export.shape=[1,3,H,W]")
    path = context.primary_call(
        lambda: resolve_checkpoint(training["resume"] or training["checkpoint"])
    )
    model, payload = load_checkpoint(
        path, config_model=config["model"], preprocessing=DEFAULT_PREPROCESSING
    )
    device = context.device
    model = model.to(device)
    inherited_training = training_snapshots(payload)
    if "modelopt_state" in payload:
        calibration = payload.get("calibration")
        if not calibration:
            raise ValueError("Quantized checkpoint is missing calibration provenance")
        validate_split_sources(calibration["data_snapshot"], "calibration")
        if (
            calibration["shape"] != shape
            or calibration["policy"]["excluded_modules"] != quantization["exclude_modules"]
            or calibration["policy"].get("numeric_backend") != INT8_BACKEND
        ):
            raise ValueError(
                "Saved quantization shape/policy differs from the current configuration"
            )
    else:
        if training["resume"]:
            raise ValueError("QAT --resume requires a QAT training checkpoint")
        samples = scan_calibration(resolve_split(config, "calibration"), config["data"]["layout"])
        validate_split_sources(samples, "calibration")
        samples = samples[: quantization["calibration_samples"]]
        calibration_snapshot = snapshot(samples)
        for saved in inherited_training:
            check_leakage(saved, calibration_snapshot)
        heldout_snapshots = {
            split: saved
            for split, saved in payload.get("heldout_snapshots", {}).items()
            if split in {"val", "test"}
        }
        for split, saved in heldout_snapshots.items():
            validate_split_sources(saved, split)
        for split in ("val", "test"):
            folder = resolve_split(config, split)
            if folder.exists():
                heldout_samples = scan_pairs(folder, config["data"]["layout"])
                validate_split_sources(heldout_samples, split)
                heldout_snapshots[split] = snapshot(heldout_samples)
        check_leakage(calibration_snapshot, *heldout_snapshots.values())
        for saved in inherited_training:
            check_leakage(saved, *heldout_snapshots.values())

        def forward_loop(calibration_model):
            with torch.no_grad():
                for sample in samples:
                    image, _ = prepare_rgb(read_rgb(sample.rainy), protocol="fixed", shape=shape)
                    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
                    if context.any_true(not bool(torch.isfinite(calibration_model(tensor)).all())):
                        raise ValueError(f"Nonfinite calibration output for {sample.id}")

        model, policy = calibrate_model(model, forward_loop, quantization["exclude_modules"])
        calibration = {
            "data_snapshot": calibration_snapshot,
            "samples": len(samples),
            "shape": shape,
            "preprocessing": {**DEFAULT_PREPROCESSING, "protocol": "fixed"},
            "policy": policy,
            "tensorrt_target_validation": "not performed",
        }
        payload = {**payload, "calibration": calibration, "heldout_snapshots": heldout_snapshots}
    for saved in inherited_training:
        check_leakage(saved, calibration["data_snapshot"])
    if quantization["steps"] > 0:
        return train_model(
            model,
            config,
            payload,
            kind="qat",
            modelopt_state_fn=mto.modelopt_state,
            max_steps=quantization["steps"],
        )
    if training["resume"]:
        raise ValueError(
            "QAT --resume requires steps>0; use --checkpoint for calibration-only export"
        )

    def save_ptq():
        run_dir = create_run_dir(config, "qat")
        checkpoint = run_dir / "ptq.pt"
        result_payload = {
            "format_version": 1,
            "kind": "ptq",
            "state_dict": model.state_dict(),
            "modelopt_state": mto.modelopt_state(model),
            "config": {
                "model": config["model"],
                "preprocessing": DEFAULT_PREPROCESSING,
                "quantization": deepcopy(quantization),
            },
            "calibration": calibration,
            "provenance": {
                "initial": payload.get("provenance", {}),
                "checkpoint": str(path),
                "checkpoint_sha256": sha256(path),
            },
        }
        if payload.get("data_snapshot") is not None:
            result_payload["data_snapshot"] = payload["data_snapshot"]
        if payload.get("inherited_training_snapshots") is not None:
            result_payload["inherited_training_snapshots"] = payload["inherited_training_snapshots"]
        if payload.get("heldout_snapshots") is not None:
            result_payload["heldout_snapshots"] = payload["heldout_snapshots"]
        save_checkpoint(checkpoint, result_payload)
        metadata = run_metadata(config, path)
        metadata["environment"]["torch_runtime"] = {
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
        }
        metadata["environment"]["distributed"] = {
            "backend": "nccl" if context.distributed else None,
            "world_size": context.world_size,
            "calibration": "identical subset independently consumed on every rank",
        }
        metadata["calibration"] = calibration
        write_json(run_dir / "metadata.json", metadata)
        result = {
            "command": "qat",
            "kind": "ptq",
            "steps": 0,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "calibration_samples": calibration["samples"],
            "artifacts": str(run_dir),
        }
        write_json(run_dir / "result.json", result)
        return result

    result = context.primary_call(save_ptq)
    return result if context.is_primary else None

"""PyTorch backend connection for image inference and official quality scores."""

from pathlib import Path

import torch

from modern_derain.artifacts import run_metadata, write_json
from modern_derain.config import DEFAULT_PREPROCESSING, resolve_split, validate_shape
from modern_derain.data.images import prepare_rgb, read_rgb, recover, validate_pair, write_recovered
from modern_derain.data.sources import (
    Sample,
    check_leakage,
    image_files,
    scan_pairs,
    snapshot,
    training_snapshots,
    validate_split_sources,
)
from modern_derain.models.assets import resolve_checkpoint
from modern_derain.models.kpn import load_checkpoint


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu for a CPU run")
    return device


def _predict(model, rainy, device):
    tensor = torch.from_numpy(rainy).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
    if output.shape != tensor.shape:
        raise ValueError(f"Model output shape differs from RGB input: {output.shape}")
    return output[0].float().cpu().numpy()


def _load_model(config, settings, checkpoint, device):
    validate_shape(config["export"]["shape"], required=settings["protocol"] == "fixed")
    model, payload = load_checkpoint(checkpoint, config["model"], DEFAULT_PREPROCESSING, device)
    if payload.get("kind", "fp32") in ("ptq", "qat"):
        calibrated_shape = payload.get("calibration", {}).get("shape")
        if settings["protocol"] != "fixed" or calibrated_shape != config["export"]["shape"]:
            raise ValueError(
                "Quantized inference/evaluation requires protocol=fixed and the calibrated export.shape"
            )
    return model, payload


def run_infer(
    config: dict, input_path: Path | None, output_path: Path | None, checkpoint: Path | None
) -> dict:
    settings = config["inference"]
    if input_path is None and settings["input"] is None:
        raise ValueError("Inference requires --input or inference.input")
    input_path = Path(input_path if input_path is not None else settings["input"])
    output_path = Path(output_path if output_path is not None else settings["output"])
    checkpoint = Path(checkpoint if checkpoint is not None else settings["checkpoint"])
    if input_path.is_file():
        files = [input_path]
        input_root = input_path.parent
    else:
        files = image_files(input_path)
        input_root = input_path
    if output_path.resolve() == input_path.resolve() or (
        input_path.is_dir()
        and (
            output_path.resolve().is_relative_to(input_path.resolve())
            or input_path.resolve().is_relative_to(output_path.resolve())
        )
    ):
        raise ValueError("Inference output must be separate from the input folder")
    single_output = output_path.suffix.lower() == ".png"
    if single_output and len(files) != 1:
        raise ValueError("A single output PNG requires one input image")
    targets = [
        output_path
        if single_output
        else output_path / p.relative_to(input_root).with_suffix(".png")
        for p in files
    ]
    metadata_path = (
        output_path.with_suffix(".json") if single_output else output_path / "results.json"
    )
    if len(set(targets)) != len(targets):
        raise ValueError("Input filenames would create duplicate output PNGs")
    if metadata_path.exists() or any(p.exists() for p in targets):
        raise ValueError("Inference output already exists; choose a new artifact path")
    protected = {p.resolve() for p in files} | {checkpoint.resolve()}
    if metadata_path.resolve() in protected or any(p.resolve() in protected for p in targets):
        raise ValueError("Inference output would overwrite an input or checkpoint")
    device = _device(settings["device"])
    checkpoint = resolve_checkpoint(checkpoint)
    model, payload = _load_model(config, settings, checkpoint, device)
    model.eval()
    records = []
    for rainy_path, target in zip(files, targets, strict=True):
        rainy, original = prepare_rgb(
            read_rgb(rainy_path), settings["protocol"], config["export"]["shape"]
        )
        restored = recover(_predict(model, rainy, device), original)
        write_recovered(target, restored)
        records.append(
            {
                "id": rainy_path.relative_to(input_root).as_posix(),
                "output": str(target.resolve()),
                "original_size": list(original),
                "input_shape": [1, *rainy.shape],
            }
        )
    samples = [
        Sample(p.relative_to(input_root).as_posix(), p, None, p.relative_to(input_root).as_posix())
        for p in files
    ]
    result = run_metadata(config, checkpoint)
    result["environment"]["torch_runtime"] = {
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
    }
    result.update(
        {
            "backend": "pytorch",
            "kind": payload.get("kind", "fp32"),
            "preprocessing": settings["protocol"],
            "data_snapshot": snapshot(samples),
            "checkpoint_preprocessing": payload.get("config", {}).get("preprocessing"),
            "images": records,
            "count": len(records),
            "saved_image_policy": "PNG uint8 truncation after official floating recovery",
        }
    )
    write_json(metadata_path, result)
    return {"count": len(records), "output": str(output_path), "metadata": str(metadata_path)}


def run_eval(config: dict, checkpoint: Path | None, split: str | None = None) -> dict:
    from .metrics import psnr, ssim

    settings = config["evaluation"]
    output = Path(settings["output"])
    checkpoint = Path(checkpoint if checkpoint is not None else settings["checkpoint"])
    split = split if split is not None else settings["split"]
    if output.exists():
        raise ValueError(f"Evaluation output already exists: {output}")
    samples = scan_pairs(resolve_split(config, split), config["data"]["layout"])
    validate_split_sources(samples, split)
    protected = {checkpoint.resolve()} | {p.resolve() for s in samples for p in (s.rainy, s.clean)}
    if output.resolve() in protected:
        raise ValueError("Evaluation output would overwrite an input, clean image or checkpoint")
    data_snapshot = snapshot(samples)
    device = _device(settings["device"])
    checkpoint = resolve_checkpoint(checkpoint)
    model, payload = _load_model(config, settings, checkpoint, device)
    for saved in training_snapshots(payload):
        check_leakage(saved, data_snapshot)
    calibration = payload.get("calibration", {}).get("data_snapshot")
    if calibration:
        validate_split_sources(calibration, "calibration")
        check_leakage(calibration, data_snapshot)
    model.eval()
    scores = []
    for sample in samples:
        rainy_image, clean_image = read_rgb(sample.rainy), read_rgb(sample.clean)
        validate_pair(rainy_image, clean_image)
        rainy, original = prepare_rgb(rainy_image, settings["protocol"], config["export"]["shape"])
        clean, _ = prepare_rgb(clean_image, settings["protocol"], config["export"]["shape"])
        pred_recovered = recover(_predict(model, rainy, device), original)
        clean_recovered = recover(clean, original)
        scores.append(
            {
                "id": sample.id,
                "psnr": psnr(pred_recovered, clean_recovered),
                "ssim": ssim(pred_recovered, clean_recovered),
                "original_size": list(original),
                "input_shape": [1, *rainy.shape],
            }
        )
    result = run_metadata(config, checkpoint)
    result["environment"]["torch_runtime"] = {
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
    }
    result.update(
        {
            "backend": "pytorch",
            "kind": payload.get("kind", "fp32"),
            "split": split,
            "checkpoint_preprocessing": payload.get("config", {}).get("preprocessing"),
            "preprocessing": settings["protocol"],
            "metric_protocol": {
                "recovery": "clip-truncate-uint8-bgr-resize",
                "psnr_peak": 255,
                "psnr_zero_mse": 100,
                "ssim_dtype": "float64",
                "ssim_win_size": 7,
                "aggregation": "arithmetic mean of per-image scores",
            },
            "data_snapshot": data_snapshot,
            "samples": scores,
            "count": len(scores),
            "unique_sources": len({s.source_id for s in samples}),
            "psnr": sum(s["psnr"] for s in scores) / len(scores),
            "ssim": sum(s["ssim"] for s in scores) / len(scores),
        }
    )
    write_json(output, result)
    return {
        "count": result["count"],
        "psnr": result["psnr"],
        "ssim": result["ssim"],
        "output": str(output),
    }

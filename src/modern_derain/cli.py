"""Typed command dispatch; optional SDKs are imported only when a command runs."""

import importlib
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from modern_derain.config import load_config

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console(stderr=True)
Config = Annotated[Path | None, typer.Option(exists=True, file_okay=True, dir_okay=False)]
Data = Annotated[Path | None, typer.Option(exists=True, file_okay=False, dir_okay=True)]
Checkpoint = Annotated[Path | None, typer.Option(dir_okay=False)]


@app.callback()
def root():
    """EfficientDeRain v3 Rain1400: train, quantize, export and restore RGB images."""


def _strings(values):
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def _run(module, function, profile, config, overrides, *args):
    try:
        settings = load_config(config, overrides)
        # Exporter/SDK messages are human logs; stdout is a single result JSON.
        with redirect_stdout(sys.stderr):
            implementation = getattr(importlib.import_module(module), function)
            result = implementation(settings, *args)
        typer.echo(json.dumps(result, allow_nan=False))
    except ModuleNotFoundError as error:
        if error.name and (
            error.name.startswith("modelopt") or error.name in ("requests", "huggingface_hub")
        ):
            profile = "--extra qat" + (" --extra export" if "deployment" in module else "")
            if function == "run_eval":
                profile += " --extra metrics"
        console.print(
            f"Missing dependency {error.name}. Run uv run --locked {profile} modern-derain ...",
            markup=False,
        )
        raise typer.Exit(1) from error
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        console.print(str(error), markup=False)
        raise typer.Exit(1) from error


@app.command()
def train(
    config: Config = None,
    data: Data = None,
    checkpoint: Checkpoint = None,
    resume: Checkpoint = None,
    epochs: Annotated[int | None, typer.Option(min=1)] = None,
    batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    lr: Annotated[float | None, typer.Option(min=0)] = None,
    crop_size: Annotated[int | None, typer.Option(min=16)] = None,
    seed: Annotated[int | None, typer.Option(min=0)] = None,
    num_workers: Annotated[int | None, typer.Option(min=0)] = None,
    amp: Annotated[bool | None, typer.Option("--amp/--no-amp")] = None,
    device: Annotated[str | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = None,
):
    """Fine-tune FP32 KPN on paired training folders."""
    options = _strings(
        {
            "checkpoint": checkpoint,
            "resume": resume,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "crop_size": crop_size,
            "seed": seed,
            "num_workers": num_workers,
            "amp": amp,
            "device": device,
            "output_dir": output,
        }
    )
    _run(
        "modern_derain.training.loop",
        "run_train",
        "--extra train",
        config,
        {"data": _strings({"path": data}), "training": options},
    )


@app.command()
def qat(
    config: Config = None,
    data: Data = None,
    checkpoint: Checkpoint = None,
    resume: Checkpoint = None,
    steps: Annotated[int | None, typer.Option(min=0)] = None,
    calibration_samples: Annotated[int | None, typer.Option(min=1)] = None,
    batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    num_workers: Annotated[int | None, typer.Option(min=0)] = None,
    amp: Annotated[bool | None, typer.Option("--amp/--no-amp")] = None,
    device: Annotated[str | None, typer.Option()] = None,
    height: Annotated[int | None, typer.Option(min=16)] = None,
    width: Annotated[int | None, typer.Option(min=16)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
):
    """Calibrate CNN INT8 quantizers; --steps 0 saves PTQ without supervised training."""
    export = _shape(height, width)
    options = _strings(
        {
            "checkpoint": checkpoint,
            "resume": resume,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "amp": amp,
            "device": device,
            "output_dir": output,
        }
    )
    _run(
        "modern_derain.training.quantize",
        "run_qat",
        "--extra qat",
        config,
        {
            "data": _strings({"path": data}),
            "training": options,
            "quantization": {"steps": steps, "calibration_samples": calibration_samples},
            "export": export,
        },
    )


def _shape(height, width):
    if (height is None) != (width is None):
        raise typer.BadParameter("--height and --width must be supplied together")
    return {} if height is None else {"shape": [1, 3, height, width]}


@app.command(name="export")
def export_model(
    config: Config = None,
    checkpoint: Checkpoint = None,
    height: Annotated[int | None, typer.Option(min=16)] = None,
    width: Annotated[int | None, typer.Option(min=16)] = None,
    quantized: Annotated[bool | None, typer.Option("--quantized/--no-quantized")] = None,
    opset: Annotated[int | None, typer.Option(min=13)] = None,
    device: Annotated[str | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = None,
):
    """Export fixed batch-1 RGB FP32 or Q/DQ ONNX for target-side TensorRT build."""
    options = _strings(
        {"checkpoint": checkpoint, "opset": opset, "device": device, "output": output}
    )
    options.update(_shape(height, width))
    if quantized is not None:
        options["precision"] = "int8" if quantized else "fp32"
    _run(
        "modern_derain.deployment.export",
        "run_export",
        "--extra export",
        config,
        {"export": options},
    )


@app.command()
def infer(
    config: Config = None,
    checkpoint: Checkpoint = None,
    input: Annotated[Path | None, typer.Option(exists=True)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
    device: Annotated[str | None, typer.Option()] = None,
    protocol: Annotated[str | None, typer.Option()] = None,
):
    """Restore an image or folder, preserving relative image paths."""
    options = _strings(
        {
            "checkpoint": checkpoint,
            "input": input,
            "output": output,
            "device": device,
            "protocol": protocol,
        }
    )
    _run(
        "modern_derain.evaluation.quality",
        "run_infer",
        "--extra train",
        config,
        {"inference": options},
        input,
        output,
        checkpoint,
    )


@app.command(name="eval")
def evaluate(
    config: Config = None,
    data: Data = None,
    checkpoint: Checkpoint = None,
    split: Annotated[str | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = None,
    device: Annotated[str | None, typer.Option()] = None,
    protocol: Annotated[str | None, typer.Option()] = None,
):
    """Evaluate paired validation/test folders using official imagewise PSNR/SSIM."""
    options = _strings(
        {
            "checkpoint": checkpoint,
            "split": split,
            "output": output,
            "device": device,
            "protocol": protocol,
        }
    )
    _run(
        "modern_derain.evaluation.quality",
        "run_eval",
        "--extra train --extra metrics",
        config,
        {"data": _strings({"path": data}), "evaluation": options},
        checkpoint,
        split,
    )


def main():
    app()

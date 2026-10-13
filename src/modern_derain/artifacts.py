"""Run output and environment metadata, without importing a GPU SDK."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

UPSTREAM_COMMIT = "465759da2428a4263eeee6ab464a0135fca2900c"


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path | str, data) -> Path:
    path = Path(path)
    contents = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(contents)
    return path


def environment() -> dict:
    versions = {}
    for name in (
        "modern-derain",
        "numpy",
        "opencv-python-headless",
        "typer",
        "rich",
        "torch",
        "albumentationsx",
        "nvidia-modelopt",
        "onnx",
        "onnxscript",
        "scikit-image",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        result["nvidia_smi"] = process.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = None
    result["jetpack_l4t"] = None
    result["tensorrt"] = None
    return result


def run_metadata(config: dict, checkpoint: Path | str | None = None) -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "config": config,
        "project_revision": revision,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint": None if checkpoint is None else str(Path(checkpoint).resolve()),
        "checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
        "environment": environment(),
        "fingerprint_policy": "absolute path, file size and mtime_ns; not a content hash",
    }


def create_run_dir(config: dict, command: str) -> Path:
    path = Path(config["training"]["output_dir"])
    path.mkdir(parents=True, exist_ok=False)
    return path

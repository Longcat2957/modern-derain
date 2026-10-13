"""Verified checkpoint assets; no downloads during import or CLI help."""

import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

from modern_derain.artifacts import sha256

DEFAULT_CHECKPOINT = Path("weights/efficientderain-rain1400-v3.pt")
# Set only after a versioned asset has actually been published in this project's Release.
DEFAULT_RELEASE_URL = None
DEFAULT_SHA256 = "b7be892d6483e7e41b65f143ed1d6fc199cdaece8229f4a540e897a80abeeab6"
DEFAULT_SIZE = 109600519


def _verify(path: Path, expected_hash: str, expected_size: int) -> None:
    if path.stat().st_size != expected_size or sha256(path) != expected_hash:
        raise ValueError(f"checkpoint size/SHA-256 verification failed: {path}")


def download_verified(path: Path, url: str, expected_hash: str, expected_size: int) -> Path:
    """Install verified bytes atomically and never overwrite an existing file."""
    path = Path(path)
    if path.exists():
        _verify(path, expected_hash, expected_size)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, urlopen(url, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        _verify(temporary, expected_hash, expected_size)
        # An exclusive atomic link also protects against a concurrent installation.
        try:
            os.link(temporary, path)
        except FileExistsError:
            _verify(path, expected_hash, expected_size)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def resolve_checkpoint(path) -> Path:
    path = Path(path)
    is_default = path.resolve() == DEFAULT_CHECKPOINT.resolve()
    if path.is_file():
        if is_default and DEFAULT_SHA256 is not None:
            _verify(path, DEFAULT_SHA256, DEFAULT_SIZE)
        return path
    if not is_default:
        raise FileNotFoundError(f"user checkpoint does not exist: {path}")
    if DEFAULT_RELEASE_URL is None:
        raise FileNotFoundError(
            "default checkpoint is unavailable: its verified project Release URL has not been "
            "published; use an existing --checkpoint (see weights/README.md)"
        )
    return download_verified(path, DEFAULT_RELEASE_URL, DEFAULT_SHA256, DEFAULT_SIZE)

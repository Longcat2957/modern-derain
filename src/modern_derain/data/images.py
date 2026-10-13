"""NumPy/OpenCV RGB preprocessing; independent of PyTorch and SDKs."""

from pathlib import Path

import cv2
import numpy as np


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an 8-bit three-channel image: {path}")
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def validate_pair(rainy: np.ndarray, clean: np.ndarray) -> None:
    if rainy.shape != clean.shape:
        raise ValueError(f"Rainy/clean image shapes differ: {rainy.shape} versus {clean.shape}")


def prepare_rgb(
    rgb: np.ndarray, protocol: str = "official-ceil16", shape: list[int] | None = None
) -> tuple[np.ndarray, tuple[int, int]]:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected nonempty uint8 HWC RGB input")
    height, width = rgb.shape[:2]
    if protocol == "official-ceil16":
        target_h, target_w = (height + 15) // 16 * 16, (width + 15) // 16 * 16
    elif protocol == "fixed":
        if (
            shape is None
            or len(shape) != 4
            or shape[:2] != [1, 3]
            or any(type(n) is not int for n in shape)
            or min(shape[2:]) < 16
            or any(n % 16 for n in shape[2:])
        ):
            raise ValueError(
                "Fixed RGB input requires shape [1,3,H,W] with H/W >=16 and divisible by 16"
            )
        target_h, target_w = shape[2:]
    else:
        raise ValueError(f"Unknown preprocessing protocol: {protocol}")
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32) / 255.0, (
        height,
        width,
    )


def recover(chw: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """Official metric recovery: truncation before floating BGR resize."""
    if chw.ndim != 3 or chw.shape[0] != 3 or not np.isfinite(chw).all():
        raise ValueError("Expected finite three-channel CHW model output")
    height, width = original_size
    if min(height, width) < 1:
        raise ValueError("Recovery size must be positive")
    quantized = np.clip(chw.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(quantized[..., ::-1], dtype=np.float32)
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)


def write_recovered(path: Path, bgr: np.ndarray) -> None:
    path = Path(path)
    if path.exists():
        raise ValueError(f"Refusing to overwrite image: {path}")
    if path.suffix.lower() != ".png":
        raise ValueError("Restored output images must use lossless .png")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", bgr.astype(np.uint8))
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())

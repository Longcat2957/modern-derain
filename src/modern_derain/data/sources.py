"""Discover folders once and preserve the exact sample list used by a run."""

import re
import zlib
from dataclasses import dataclass
from pathlib import Path

from .rain1400_identity import GROUND_TRUTH, rainy_identity

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RAIN1400_SOURCE = "fu-cvpr2017-rain1400"


@dataclass(frozen=True)
class Sample:
    id: str
    rainy: Path
    clean: Path | None
    source_id: str


def image_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Image folder does not exist: {root}")
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"No images found in {root}")
    names, targets = set(), set()
    for path in files:
        name = path.relative_to(root).as_posix().casefold()
        target = path.resolve()
        if name in names or target in targets:
            raise ValueError(f"Duplicate or ambiguous image sample: {path}")
        names.add(name)
        targets.add(target)
    return files


def _source_id(relative: Path) -> str:
    # A nested folder represents one capture sequence; root files are independent images.
    return relative.parent.as_posix() if relative.parent != Path(".") else relative.stem


def _rain1400_id(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9]\d*)_([1-9]\d*)\.jpg", path.name)
    if not match:
        raise ValueError(f"Unverified Rain1400 filename: {path}; expected <clean_id>_<rain_id>.jpg")
    clean_id, rain_id = map(int, match.groups())
    if clean_id not in GROUND_TRUTH or not 1 <= rain_id <= 14:
        raise ValueError(f"Unknown official Rain1400 source/rain ID: {path}")
    return clean_id, rain_id


def _verify_identity(path: Path, identity: tuple[int, int], role: str) -> None:
    expected_size, expected_crc = identity
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"Unverified Rain1400 {role}: {path}; use author-hosted original JPGs, "
            "or select paired layout for a separately verified dataset"
        )
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    if crc != expected_crc:
        raise ValueError(f"Rain1400 {role} identity differs from the official archive: {path}")


def scan_pairs(root: Path, layout: str = "paired") -> list[Sample]:
    root = Path(root)
    if layout not in {"paired", "rain1400"}:
        raise ValueError(f"Unknown data layout: {layout}")
    rainy_name, clean_name = (
        ("rainy", "clean") if layout == "paired" else ("rainy_image", "ground_truth")
    )
    other_names = {"rainy_image", "ground_truth"} if layout == "paired" else {"rainy", "clean"}
    if any((root / name).exists() for name in other_names):
        raise ValueError(f"Ambiguous dataset layout in {root}; choose one folder layout")
    rainy_root, clean_root = root / rainy_name, root / clean_name
    rainy_files, clean_files = image_files(rainy_root), image_files(clean_root)
    clean_lookup = {path.relative_to(clean_root).as_posix(): path for path in clean_files}
    samples, checked, rain_ids = [], set(), set()
    for rainy in rainy_files:
        relative = rainy.relative_to(rainy_root)
        if layout == "paired":
            key = relative.as_posix()
            source_id = _source_id(relative)
        else:
            if relative.parent != Path("."):
                raise ValueError(f"Unverified nested Rain1400 structure: {rainy}")
            clean_id, rain_id = _rain1400_id(rainy)
            if (clean_id, rain_id) in rain_ids:
                raise ValueError(f"Duplicate Rain1400 rain ID: {rainy}")
            rain_ids.add((clean_id, rain_id))
            key = f"{clean_id}.jpg"
            source_id = f"{RAIN1400_SOURCE}:{clean_id}"
        if key not in clean_lookup:
            raise ValueError(f"Missing clean counterpart for {rainy}: {clean_root / key}")
        clean = clean_lookup[key]
        if clean.resolve() == rainy.resolve():
            raise ValueError(f"Rainy and clean are the same file: {rainy}")
        if layout == "rain1400" and clean not in checked:
            _verify_identity(clean, GROUND_TRUTH[clean_id], "ground truth")
            checked.add(clean)
        samples.append(Sample(relative.as_posix(), rainy, clean, source_id))
    used_clean = {sample.clean for sample in samples}
    if used_clean != set(clean_files):
        raise ValueError(f"Clean folder contains unmatched images: {clean_root}")
    if layout == "rain1400":
        ids = {int(sample.source_id.rsplit(":", 1)[1]) for sample in samples}
        if any(i <= 900 for i in ids) and any(i > 900 for i in ids):
            raise ValueError(
                "Official Rain1400 training and testing sources must remain in separate splits"
            )
    return samples


def scan_calibration(root: Path, layout: str = "paired") -> list[Sample]:
    """Read only the explicitly selected rainy folder, never a different split."""
    root = Path(root)
    if layout not in {"paired", "rain1400"}:
        raise ValueError(f"Unknown data layout: {layout}")
    if (root / "rainy").exists() and (root / "rainy_image").exists():
        raise ValueError(f"Ambiguous calibration folder structure: {root}")
    if (root / "rainy").is_dir():
        rainy_root = root / "rainy"
        official_rain1400 = False
    elif (root / "rainy_image").is_dir():
        if layout != "rain1400":
            raise ValueError("rainy_image calibration requires explicit data.layout=rain1400")
        rainy_root = root / "rainy_image"
        official_rain1400 = True
    else:
        if any((root / name).exists() for name in ("clean", "ground_truth")):
            raise ValueError(f"Expected rainy-only or paired/rainy calibration folder: {root}")
        rainy_root = root
        official_rain1400 = False
    files = image_files(rainy_root)
    if layout == "rain1400" and rainy_root == root:
        numeric = [p for p in files if p.parent == root and re.fullmatch(r"\d+_\d+\.jpg", p.name)]
        if numeric:
            if len(numeric) != len(files):
                raise ValueError(
                    "Ambiguous Rain1400 calibration filenames; place generic drone images under rainy/"
                )
            official_rain1400 = True
    samples = []
    for path in files:
        relative = path.relative_to(rainy_root)
        if official_rain1400:
            if relative.parent != Path("."):
                raise ValueError(f"Unverified nested Rain1400 calibration structure: {path}")
            clean_id, rain_id = _rain1400_id(path)
            try:
                _verify_identity(path, rainy_identity(clean_id, rain_id), "rainy image")
            except ValueError as error:
                raise ValueError(
                    f"{error}; place generic drone images under rainy/ to use sequence identity"
                ) from error
            source_id = f"{RAIN1400_SOURCE}:{clean_id}"
        else:
            source_id = _source_id(relative)
        samples.append(Sample(relative.as_posix(), path, None, source_id))
    validate_split_sources(samples, "calibration")
    return samples


def validate_split_sources(samples: list[Sample] | list[dict], split: str) -> None:
    """Preserve the verified archive's official test sources in their original role."""
    if split not in {"train", "val", "test", "calibration"}:
        raise ValueError(f"Unknown split: {split}")
    prefix = f"{RAIN1400_SOURCE}:"
    for sample in samples:
        source = sample["source_id"] if isinstance(sample, dict) else sample.source_id
        if source.startswith(prefix):
            try:
                source_id = int(source[len(prefix) :])
            except ValueError as error:
                raise ValueError(f"Invalid verified Rain1400 source ID: {source}") from error
            valid = 901 <= source_id <= 1000 if split == "test" else 1 <= source_id <= 900
            if not valid:
                raise ValueError(
                    f"Original Rain1400 source {source_id} is not allowed in {split}; "
                    "IDs 1–900 are training sources and 901–1000 are final test sources"
                )


def training_snapshots(payload: dict) -> list[list[dict]]:
    """Keep every dataset that influenced inherited weights separate from resume input."""
    inherited = payload.get("inherited_training_snapshots", [])
    if not isinstance(inherited, list):
        raise ValueError("Checkpoint inherited_training_snapshots must be a list")
    records = []
    for saved in [*inherited, payload.get("data_snapshot", [])]:
        if not isinstance(saved, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("source_id"), str)
            for item in saved
        ):
            raise ValueError("Checkpoint training snapshot must contain source identity records")
        if saved and saved not in records:
            validate_split_sources(saved, "train")
            records.append(saved)
    return records


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def snapshot(samples: list[Sample]) -> list[dict]:
    """Stat fingerprints detect path/size/mtime changes, not every content edit."""
    return [
        {
            "id": s.id,
            "source_id": s.source_id,
            "rainy": _fingerprint(s.rainy),
            "clean": _fingerprint(s.clean) if s.clean is not None else None,
        }
        for s in samples
    ]


def check_leakage(*snapshots: list[dict]) -> None:
    source_ids, files = set(), set()
    for records in snapshots:
        current_sources = {item["source_id"] for item in records}
        current_files = {
            item[role]["path"]
            for item in records
            for role in ("rainy", "clean")
            if item.get(role) is not None
        }
        overlap = source_ids & current_sources
        if overlap or files & current_files:
            raise ValueError(f"Source leakage between splits: {sorted(overlap)[:5]}")
        source_ids.update(current_sources)
        files.update(current_files)

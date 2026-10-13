from pathlib import Path

import pytest

from modern_derain.artifacts import sha256
from modern_derain.models import assets, kpn


def test_download_checks_hash_installs_once_and_cleans_failure(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"small-checkpoint-fixture")
    destination = tmp_path / "weights" / "model.pt"
    expected = sha256(source)
    assert (
        assets.download_verified(destination, source.as_uri(), expected, source.stat().st_size)
        == destination
    )
    assert destination.read_bytes() == source.read_bytes()
    assert (
        assets.download_verified(destination, "missing://unused", expected, source.stat().st_size)
        == destination
    )
    destination.write_bytes(b"existing-different-content")
    with pytest.raises(ValueError, match="verification"):
        assets.download_verified(destination, source.as_uri(), expected, source.stat().st_size)
    assert destination.read_bytes() == b"existing-different-content"
    bad_destination = destination.with_name("bad.pt")
    with pytest.raises(ValueError, match="verification"):
        assets.download_verified(bad_destination, source.as_uri(), "0" * 64, source.stat().st_size)
    assert not bad_destination.exists()
    assert list(destination.parent.glob(".bad.pt.*")) == []


def test_resolver_never_falls_back_for_missing_user_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(assets, "DEFAULT_RELEASE_URL", None)
    with pytest.raises(FileNotFoundError, match="user checkpoint"):
        assets.resolve_checkpoint(Path("user.pt"))
    with pytest.raises(FileNotFoundError, match="Release URL"):
        assets.resolve_checkpoint(assets.DEFAULT_CHECKPOINT)


def test_packaging_exclusively_preserves_tensor_values(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(kpn, "KPN", lambda: torch.nn.Conv2d(3, 3, 1))
    source = tmp_path / "raw.pth"
    state = kpn.KPN().state_dict()
    torch.save(state, source)
    monkeypatch.setattr(kpn, "UPSTREAM_CHECKPOINT_SHA256", sha256(source))
    destination = tmp_path / "packaged.pt"
    payload = kpn.package_checkpoint(source, destination, {"protocol": "official-ceil16"})
    assert all(torch.equal(value, payload["state_dict"][key]) for key, value in state.items())
    assert list(tmp_path.glob(".packaged.pt.*")) == []
    with pytest.raises(ValueError, match="new file"):
        kpn.package_checkpoint(source, destination, {})
    source.write_bytes(b"wrong official checkpoint")
    with pytest.raises(ValueError, match="does not match"):
        kpn.package_checkpoint(source, tmp_path / "bad.pt", {})

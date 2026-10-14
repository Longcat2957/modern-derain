import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from modern_derain.config import DEFAULTS
from modern_derain.data.images import read_rgb
from modern_derain.training.losses import DerainLoss

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs"),
]
WORKER = Path(__file__).parent / "fixtures" / "distributed_training_worker.py"


def run_distributed(tmp_path, scenario, *, random_crop=False):
    (tmp_path / "observations").mkdir()
    rng = np.random.default_rng(42)
    for side in ("rainy", "clean"):
        (tmp_path / "data" / "train" / side).mkdir(parents=True)
    intensities = (20, 60, 200, 240, 120) if scenario == "gradient" else (20, 60, 200, 240)
    for index, intensity in enumerate(intensities):
        size = 20 if random_crop else 16
        rainy = rng.integers(80, 180, (size, size, 3), dtype=np.uint8)
        clean = np.full_like(rainy, intensity)
        for side, image in (("rainy", rainy), ("clean", clean)):
            assert cv2.imwrite(str(tmp_path / "data" / "train" / side / f"{index}.png"), image)
    torch.save({"provenance": {}}, tmp_path / "initial.pt")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    # Clearing inherited ranks also makes this regression safe inside a launcher shell.
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(WORKER),
            scenario,
            str(tmp_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return [
        json.loads((tmp_path / "observations" / f"rank-{rank}.json").read_text())
        for rank in range(2)
    ]


def test_ddp_reduces_global_batch_gradient_and_saves_plain_weights(tmp_path):
    observations = run_distributed(tmp_path, "gradient")
    consumed = sorted(observations[0]["sampler_indices"][0] + observations[1]["sampler_indices"][0])
    assert len(consumed) == len(set(consumed)) == 4
    rainy, clean = (
        torch.stack(
            [
                torch.from_numpy(read_rgb(path).transpose(2, 0, 1).copy()).float() / 255
                for path in (
                    tmp_path / "data" / "train" / side / f"{index}.png" for index in consumed
                )
            ]
        )
        for side in ("rainy", "clean")
    )
    scale = torch.nn.Parameter(torch.tensor([0.75, 1.0, 1.25]).view(1, 3, 1, 1))
    optimizer = torch.optim.Adam(
        [scale], lr=DEFAULTS["training"]["lr"], betas=tuple(DEFAULTS["training"]["betas"])
    )
    DerainLoss()(rainy * scale, clean).backward()
    expected_gradient = scale.grad.clone()
    optimizer.step()
    for rank in range(2):
        actual = torch.load(tmp_path / "observations" / f"rank-{rank}.pt", weights_only=True)
        torch.testing.assert_close(actual["gradient"], expected_gradient, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(actual["scale"], scale, rtol=0, atol=2e-7)
    assert observations[0]["complete"] == observations[1]["complete"]
    assert observations[0]["is_primary_result"] and not observations[1]["is_primary_result"]
    assert observations[0]["device"] == "cuda:0" and observations[1]["device"] == "cuda:1"
    assert set(observations[0]["sampler_indices"][0]).isdisjoint(
        observations[1]["sampler_indices"][0]
    )
    checkpoints = list((tmp_path / "complete").glob("*.pt"))
    assert len(checkpoints) == 1
    payload = torch.load(checkpoints[0], weights_only=True)
    assert set(payload["state_dict"]) == {"scale"}
    assert payload["training_state"]["contract"]["distributed"]["world_size"] == 2
    assert payload["training_state"]["contract"]["sampler"] == "distributed-random"
    assert len(payload["training_state"]["rng_by_rank"]) == 2
    metadata = json.loads((tmp_path / "complete" / "metadata.json").read_text())
    assert metadata["sampler"]["excluded_sampler_tail"] == 1
    assert (tmp_path / "complete" / "result.json").is_file()


def test_ddp_epoch_resume_replays_each_rank_crop_and_sampler(tmp_path):
    observations = run_distributed(tmp_path, "resume", random_crop=True)
    complete = torch.load(observations[0]["complete"]["checkpoint"], weights_only=True)
    resumed = torch.load(observations[0]["resumed"]["checkpoint"], weights_only=True)
    torch.testing.assert_close(
        complete["state_dict"]["scale"], resumed["state_dict"]["scale"], rtol=0, atol=0
    )
    assert observations[0]["resumed"]["losses"] == observations[0]["complete"]["losses"][1:]
    for observation in observations:
        assert observation["sampler_epochs"] == [0, 1, 1]
        assert observation["sampler_indices"][1] == observation["sampler_indices"][2]
        assert "contract" in observation["contract_error"].lower()
    assert observations[0]["sampler_indices"][0] != observations[0]["sampler_indices"][1]
    states = complete["training_state"]["rng_by_rank"]
    assert not torch.equal(states[0]["loader"], states[1]["loader"])


@pytest.mark.parametrize("scenario", ["loss", "gradient_nan"])
def test_nonfinite_on_one_rank_stops_all_ranks_without_hanging(tmp_path, scenario):
    observations = run_distributed(tmp_path, scenario)
    assert all("Nonfinite" in observation["error"] for observation in observations)
    assert not list((tmp_path / "complete").glob("*.pt"))

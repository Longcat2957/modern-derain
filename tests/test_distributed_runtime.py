import pytest
import torch

from modern_derain.training.distributed import current_context, distributed_session


@pytest.fixture(autouse=True)
def clear_launcher_environment(monkeypatch):
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_single_cpu_session_preserves_execution_without_cuda_setup(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Single CPU execution must not initialize CUDA or a process group")

    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    with distributed_session("cpu") as context:
        assert context == current_context("cpu")
        assert context.device == torch.device("cpu")
        assert context.is_primary and not context.distributed
        assert context.primary_call(lambda: 42) == 42
    assert not torch.distributed.is_initialized()


@pytest.mark.parametrize(
    ("environment", "device", "message"),
    [
        ({"WORLD_SIZE": "invalid"}, "cpu", "WORLD_SIZE must be an integer"),
        ({"WORLD_SIZE": "0"}, "cpu", "WORLD_SIZE must be >= 1"),
        ({"WORLD_SIZE": "2"}, "cuda", "valid RANK and LOCAL_RANK"),
        (
            {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"},
            "cpu",
            "Distributed training requires CUDA",
        ),
    ],
)
def test_invalid_launcher_fails_before_starting_process_group(
    monkeypatch, environment, device, message
):
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid launch settings must fail before waiting for other ranks")

    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    with pytest.raises(ValueError, match=message):
        with distributed_session(device):
            pass

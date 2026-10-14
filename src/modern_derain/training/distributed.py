"""Single-node/multi-node torchrun setup for one CUDA device per training rank."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


def _environment_integer(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error


def _device(name: str, *, local_rank: int | None = None) -> torch.device:
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only cpu and cuda devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select device='cpu' for CPU execution")
    if local_rank is not None:
        if device.type != "cuda":
            raise ValueError("Distributed training requires CUDA and one GPU per torchrun process")
        if device.index is not None and device.index != local_rank:
            raise ValueError("Use device='cuda' with torchrun; LOCAL_RANK selects each process GPU")
        device = torch.device("cuda", local_rank)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable")
    return device


@dataclass(frozen=True)
class DistributedContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def any_true(self, value: bool) -> bool:
        """Let every rank fail together before a following backward collective."""
        if not self.distributed:
            return bool(value)
        flag = torch.tensor(int(value), device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def sum_values(self, values: list[float]) -> list[float]:
        if not self.distributed:
            return list(values)
        tensor = torch.tensor(values, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()

    def gather_objects(self, value) -> list | None:
        """Collect rank-local RNG state for an epoch-boundary checkpoint."""
        if not self.distributed:
            return [value]
        gathered = [None] * self.world_size if self.is_primary else None
        dist.gather_object(value, gathered, dst=0)
        return gathered

    def broadcast_object(self, value):
        if not self.distributed:
            return value
        objects = [value if self.is_primary else None]
        dist.broadcast_object_list(objects, src=0, device=self.device)
        return objects[0]

    def primary_call(self, function):
        """Run filesystem work once and share its result or failure with all ranks."""
        if not self.distributed:
            return function()
        outcome = None
        if self.is_primary:
            try:
                outcome = {"ok": True, "value": function()}
            except Exception as error:
                outcome = {
                    "ok": False,
                    "type": type(error).__name__,
                    "message": str(error),
                }
        outcome = self.broadcast_object(outcome)
        if not outcome["ok"]:
            error_type = ValueError if outcome["type"] == "ValueError" else RuntimeError
            raise error_type(f"Rank 0 {outcome['type']}: {outcome['message']}")
        return outcome["value"]

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier(device_ids=[self.local_rank])


def current_context(device_name: str) -> DistributedContext:
    """Use the active session, or retain the ordinary single-process behavior."""
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size == 1:
            return DistributedContext(_device(device_name), rank=dist.get_rank())
        if torch.device(device_name).type != "cuda":
            raise ValueError("Distributed training requires CUDA and one GPU per torchrun process")
        if dist.get_backend() != "nccl":
            raise ValueError("Distributed CUDA training requires the NCCL backend")
        local_rank = _environment_integer("LOCAL_RANK", -1)
        if local_rank < 0:
            local_rank = torch.cuda.current_device()
        return DistributedContext(
            _device(device_name, local_rank=local_rank),
            rank=dist.get_rank(),
            world_size=world_size,
            local_rank=local_rank,
        )
    if _environment_integer("WORLD_SIZE", 1) > 1:
        raise ValueError("torchrun training must enter distributed_session before train_model")
    return DistributedContext(_device(device_name))


@contextmanager
def distributed_session(device_name: str):
    """Initialize NCCL from torchrun's environment and always release owned state."""
    world_size = _environment_integer("WORLD_SIZE", 1)
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be >= 1")
    if dist.is_available() and dist.is_initialized():
        yield current_context(device_name)
        return
    if world_size == 1:
        yield DistributedContext(_device(device_name))
        return
    if not dist.is_available() or not dist.is_nccl_available():
        raise ValueError("Distributed CUDA training requires PyTorch NCCL support")
    rank = _environment_integer("RANK", -1)
    local_rank = _environment_integer("LOCAL_RANK", -1)
    if not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("torchrun must provide valid RANK and LOCAL_RANK values")
    device = _device(device_name, local_rank=local_rank)
    torch.cuda.set_device(device)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=5),
            device_id=device,
        )
        yield DistributedContext(device, rank, world_size, local_rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

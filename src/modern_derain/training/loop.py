"""Original Adam/loss schedule with explicit epoch-boundary replay."""

import os
import random
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from modern_derain.artifacts import create_run_dir, run_metadata, sha256, write_json
from modern_derain.config import DEFAULT_PREPROCESSING, resolve_split
from modern_derain.data.paired import PairedDataset, worker_init_fn
from modern_derain.data.sources import (
    check_leakage,
    scan_calibration,
    scan_pairs,
    snapshot,
    training_snapshots,
    validate_split_sources,
)
from modern_derain.training.distributed import current_context, distributed_session
from modern_derain.training.losses import DerainLoss


def select_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select device='cpu' for CPU execution")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Only cpu and cuda devices are supported")
    return device


def save_checkpoint(path: Path, payload: dict) -> None:
    """Publish a complete checkpoint without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        # A hard link makes publication atomic and fails if the destination exists.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rng_state(dataset, generator: torch.Generator, context=None) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() and not (context and context.distributed)
            else []
        ),
        "local_cuda": (
            torch.cuda.get_rng_state(context.device) if context and context.distributed else None
        ),
        "augmentation": dataset.get_augmentation_state(),
        "loader": generator.get_state(),
    }


def _restore_rng(state: dict, dataset, generator: torch.Generator, context=None) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if state.get("local_cuda") is not None:
        if context is None or not context.distributed:
            raise ValueError("Distributed RNG state requires the saved distributed topology")
        torch.cuda.set_rng_state(state["local_cuda"], context.device)
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the saved CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])
    dataset.restore_augmentation_state(state["augmentation"])
    generator.set_state(state["loader"])


def training_contract(config: dict, kind: str, max_steps: int | None) -> dict:
    training = {
        key: value
        for key, value in config["training"].items()
        if key not in {"checkpoint", "resume", "output_dir"}
    }
    contract = {
        "model": config["model"],
        "preprocessing": DEFAULT_PREPROCESSING,
        "training": training,
        "kind": kind,
        "max_steps": max_steps,
        "augmentation": {
            "policy": "upstream-common-random-crop",
            "crop_size": training["crop_size"],
        },
        "sampler": "random",
        "quantization": config["quantization"] if kind == "qat" else None,
    }
    context = current_context(training["device"])
    if context.distributed:
        contract["sampler"] = "distributed-random"
        contract["distributed"] = {
            "backend": "nccl",
            "world_size": context.world_size,
            "batch_size_per_rank": training["batch_size"],
            "global_batch_size": training["batch_size"] * context.world_size,
            "sampler_drop_last": True,
        }
    return contract


def _adjust_lr(optimizer, training: dict, epoch: int) -> None:
    # Upstream updates after each optimizer step using the one-based current epoch.
    if epoch >= training["decay_start_epoch"]:
        denominator = training["epochs"] - training["decay_start_epoch"]
        multiplier = (training["epochs"] - epoch) / denominator if denominator else 0.0
        for group in optimizer.param_groups:
            group["lr"] = training["lr"] * multiplier


def train_model(
    model,
    config: dict,
    initial_payload: dict,
    *,
    kind: str = "fp32",
    modelopt_state_fn=None,
    max_steps: int | None = None,
) -> dict | None:
    """Train a prepared model; quantization is owned by the caller."""
    with distributed_session(config["training"]["device"]) as context:
        return _train_model(
            model,
            config,
            initial_payload,
            context,
            kind=kind,
            modelopt_state_fn=modelopt_state_fn,
            max_steps=max_steps,
        )


def _train_model(
    model,
    config,
    initial_payload,
    context,
    *,
    kind,
    modelopt_state_fn,
    max_steps,
) -> dict | None:
    training = config["training"]
    device = context.device
    if training["amp"] and device.type != "cuda":
        raise ValueError("AMP training requires CUDA")
    seed = training["seed"] + context.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    samples = scan_pairs(resolve_split(config, "train"), config["data"]["layout"])
    validate_split_sources(samples, "train")
    data_snapshot = snapshot(samples)
    heldout_snapshots = {}
    for split in ("val", "calibration"):
        if split == "calibration" and initial_payload.get("calibration"):
            heldout_snapshots[split] = initial_payload["calibration"]["data_snapshot"]
            validate_split_sources(heldout_snapshots[split], split)
        elif split in initial_payload.get("heldout_snapshots", {}):
            heldout_snapshots[split] = initial_payload["heldout_snapshots"][split]
            validate_split_sources(heldout_snapshots[split], split)
        folder = resolve_split(config, split)
        if folder.exists():
            scanner = scan_calibration if split == "calibration" else scan_pairs
            heldout_samples = scanner(folder, config["data"]["layout"])
            validate_split_sources(heldout_samples, split)
            heldout_snapshots[split] = snapshot(heldout_samples)
    check_leakage(data_snapshot, *heldout_snapshots.values())
    inherited_training_snapshots = [
        saved for saved in training_snapshots(initial_payload) if saved != data_snapshot
    ]
    for saved in inherited_training_snapshots:
        check_leakage(saved, *heldout_snapshots.values())
    original_calibration = initial_payload.get("calibration", {}).get("data_snapshot")
    if original_calibration:
        for saved in [data_snapshot, *inherited_training_snapshots]:
            check_leakage(saved, original_calibration)
        if heldout_snapshots.get("val"):
            check_leakage(heldout_snapshots["val"], original_calibration)
    previous_snapshot = initial_payload.get("data_snapshot")
    dataset = PairedDataset(samples, crop_size=training["crop_size"], seed=seed)
    generator = torch.Generator().manual_seed(seed)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=training["seed"],
            drop_last=True,
        )
        if context.distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=training["num_workers"],
        persistent_workers=training["persistent_workers"],
        drop_last=training["drop_last"],
        generator=generator,
        worker_init_fn=worker_init_fn,
        pin_memory=device.type == "cuda",
        multiprocessing_context=(
            "spawn" if context.distributed and training["num_workers"] > 0 else None
        ),
    )
    if not len(loader):
        raise ValueError("The training loader has no batches; check batch_size/drop_last")
    model = model.to(device).train()
    # Save the original module so checkpoints retain single-device key names and
    # ModelOpt receives the model whose quantizer structure it owns.
    prepared_model = model
    if context.distributed:
        model = DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training["lr"],
        betas=tuple(training["betas"]),
        weight_decay=training["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=training["amp"])
    loss_function = DerainLoss().to(device)
    contract = training_contract(config, kind, max_steps)
    epoch, step = 0, 0
    if training["resume"]:
        state = initial_payload.get("training_state")
        if not state or not state.get("resume_allowed"):
            raise ValueError("Checkpoint cannot resume; use --checkpoint for weights only")
        if state["contract"] != contract or previous_snapshot != data_snapshot:
            raise ValueError("Resume contract or data fingerprint changed")
        if training["num_workers"] != 0:
            raise ValueError("Exact epoch-boundary resume requires num_workers=0")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        epoch, step = state["epoch"], state["step"]
        if state["scheduler"] != {"epoch": epoch, "step": step}:
            raise ValueError("Resume schedule state is inconsistent with the saved epoch/step")
        if context.distributed:
            rank_states = state.get("rng_by_rank")
            if not isinstance(rank_states, list) or len(rank_states) != context.world_size:
                raise ValueError("Resume requires RNG states for every saved rank")
            saved_rng = rank_states[context.rank]
        else:
            saved_rng = state["rng"]
        _restore_rng(saved_rng, dataset, generator, context)
    if epoch >= training["epochs"] or (max_steps is not None and step >= max_steps):
        raise ValueError("Saved training already reached the configured epoch/step limit")
    run_dir = Path(
        context.primary_call(
            lambda: str(create_run_dir(config, "qat" if kind == "qat" else "train"))
        )
    )
    checkpoint_source = training["resume"] or training["checkpoint"]
    metadata = context.primary_call(lambda: run_metadata(config, checkpoint_source))
    metadata["environment"]["torch_runtime"] = {
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
    }
    metadata.update(
        {
            "data_snapshot": data_snapshot,
            "inherited_training_snapshots": inherited_training_snapshots,
            "heldout_snapshots": heldout_snapshots,
            "contract": contract,
            "resume_scope": "epoch boundary, num_workers=0",
            "kind": kind,
        }
    )
    if context.distributed:
        devices = context.gather_objects(
            {
                "rank": context.rank,
                "local_rank": context.local_rank,
                "device": str(device),
                "name": torch.cuda.get_device_name(device),
            }
        )
        metadata["environment"]["torch_runtime"]["distributed"] = {
            **contract["distributed"],
            "devices": devices,
        }
        metadata["sampler"] = {
            "samples_per_rank": len(sampler),
            "samples_per_epoch": len(sampler) * context.world_size,
            "excluded_sampler_tail": len(dataset) - len(sampler) * context.world_size,
            "local_drop_last": training["drop_last"],
        }
    context.primary_call(lambda: write_json(run_dir / "metadata.json", metadata))
    console = Console(stderr=True)
    checkpoint_path = None
    losses = []

    def checkpoint_payload(resume_allowed, rank_rng=None):
        state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": {"epoch": epoch, "step": step},
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "rng": rank_rng[0] if rank_rng else _rng_state(dataset, generator, context),
            "contract": contract,
            "resume_allowed": resume_allowed,
        }
        if rank_rng is not None:
            state["rng_by_rank"] = rank_rng
        payload = {
            "format_version": 1,
            "kind": kind,
            "state_dict": prepared_model.state_dict(),
            "config": {
                "model": deepcopy(config["model"]),
                "preprocessing": DEFAULT_PREPROCESSING,
                "quantization": deepcopy(config["quantization"]) if kind == "qat" else None,
            },
            "data_snapshot": data_snapshot,
            "inherited_training_snapshots": inherited_training_snapshots,
            "heldout_snapshots": heldout_snapshots,
            "training_state": state,
            "provenance": {
                "initial": initial_payload.get("provenance", {}),
                "checkpoint": str(checkpoint_source),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            },
        }
        if "calibration" in initial_payload:
            payload["calibration"] = initial_payload["calibration"]
        if modelopt_state_fn is not None:
            payload["modelopt_state"] = modelopt_state_fn(prepared_model)
        return payload

    try:
        with Progress(
            console=console,
            disable=not console.is_terminal or not context.is_primary,
            redirect_stdout=False,
        ) as progress:
            task = progress.add_task(
                "Training", total=training["epochs"] * len(loader), completed=step
            )
            while epoch < training["epochs"]:
                if sampler is not None:
                    sampler.set_epoch(epoch)
                epoch_loss = 0.0
                batches = 0
                completed_epoch = True
                for batch_index, (rainy, clean) in enumerate(loader):
                    rainy, clean = rainy.to(device), clean.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=device.type, enabled=training["amp"]):
                        prediction = model(rainy)
                    loss = loss_function(prediction.float(), clean.float())
                    if context.any_true(not bool(torch.isfinite(loss))):
                        raise ValueError(f"Nonfinite loss at step {step}")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    invalid_gradient = any(
                        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    )
                    if context.any_true(invalid_gradient):
                        raise ValueError(f"Nonfinite gradient at step {step}")
                    scaler.step(optimizer)
                    scaler.update()
                    step += 1
                    _adjust_lr(optimizer, training, epoch + 1)
                    epoch_loss += float(loss.detach())
                    batches += 1
                    progress.update(task, advance=1)
                    if max_steps is not None and step >= max_steps:
                        completed_epoch = batch_index + 1 == len(loader)
                        break
                if completed_epoch:
                    epoch += 1
                epoch_loss, batches = context.sum_values([epoch_loss, batches])
                losses.append({"epoch": epoch, "step": step, "loss": epoch_loss / batches})
                checkpoint_path = run_dir / f"epoch-{epoch:04d}-step-{step:08d}.pt"
                rank_rng = (
                    context.gather_objects(_rng_state(dataset, generator, context))
                    if context.distributed
                    else None
                )
                context.primary_call(
                    lambda: save_checkpoint(
                        checkpoint_path,
                        checkpoint_payload(
                            completed_epoch and training["num_workers"] == 0, rank_rng
                        ),
                    )
                )
                if max_steps is not None and step >= max_steps:
                    break
    except KeyboardInterrupt:
        checkpoint_path = run_dir / "interrupted.pt"
        # Other ranks may already be gone after a launcher interrupt. Do not
        # attempt collectives; this checkpoint only permits inference/export.
        if context.is_primary:
            payload = checkpoint_payload(False)
            payload["interrupted"] = True
            save_checkpoint(checkpoint_path, payload)
            console.print(
                f"Interrupted weights saved to {checkpoint_path}; exact resume uses an epoch checkpoint"
            )
        raise
    if not context.is_primary:
        return None
    result = {
        "command": "qat" if kind == "qat" else "train",
        "kind": kind,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "epoch": epoch,
        "step": step,
        "losses": losses,
        "artifacts": str(run_dir),
    }
    write_json(run_dir / "result.json", result)
    return result


def run_train(config: dict) -> dict | None:
    from modern_derain.models.assets import resolve_checkpoint
    from modern_derain.models.kpn import load_checkpoint

    training = config["training"]
    with distributed_session(training["device"]) as context:
        path = context.primary_call(
            lambda: str(resolve_checkpoint(training["resume"] or training["checkpoint"]))
        )
        model, payload = load_checkpoint(
            path, config_model=config["model"], preprocessing=DEFAULT_PREPROCESSING
        )
        if "modelopt_state" in payload:
            raise ValueError("Quantized checkpoints must be trained with qat")
        return train_model(model, config, payload)

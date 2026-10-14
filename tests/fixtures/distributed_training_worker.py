"""Small real-CUDA training scenarios invoked by torchrun in the regression tests."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler

from modern_derain.config import DEFAULTS, deep_merge
from modern_derain.training.distributed import distributed_session
from modern_derain.training.loop import train_model


class ScaleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([0.75, 1.0, 1.25]).view(1, 3, 1, 1))
        self.fail_loss = False

    def forward(self, image):
        output = image * self.scale
        return output * float("nan") if self.fail_loss else output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=("gradient", "resume", "loss", "gradient_nan"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = deep_merge(
        DEFAULTS,
        {
            "data": {"path": str(args.root / "data")},
            "training": {
                "checkpoint": str(args.root / "initial.pt"),
                "epochs": 2 if args.scenario == "resume" else 1,
                "decay_start_epoch": 1,
                "crop_size": 16,
                "batch_size": 2,
                "output_dir": str(args.root / "complete"),
            },
        },
    )
    sampler_epochs, sampler_indices = [], []
    original_set_epoch, original_iter = DistributedSampler.set_epoch, DistributedSampler.__iter__

    def record_epoch(sampler, epoch):
        sampler_epochs.append(epoch)
        return original_set_epoch(sampler, epoch)

    def record_indices(sampler):
        indices = list(original_iter(sampler))
        sampler_indices.append(indices)
        return iter(indices)

    DistributedSampler.set_epoch = record_epoch
    DistributedSampler.__iter__ = record_indices
    with distributed_session("cuda") as distributed:
        model = ScaleModel()
        if distributed.rank == 1:
            if args.scenario == "loss":
                model.fail_loss = True
            elif args.scenario == "gradient_nan":
                model.scale.register_hook(lambda gradient: gradient * float("nan"))
        observations = {"rank": distributed.rank, "device": str(distributed.device)}
        if args.scenario in ("loss", "gradient_nan"):
            try:
                train_model(model, config, {})
            except ValueError as error:
                observations["error"] = str(error)
            else:
                raise AssertionError("Nonfinite training must fail on every rank")
        else:
            complete = train_model(model, config, {})
            observations["is_primary_result"] = complete is not None
            complete = distributed.broadcast_object(complete)
            observations["complete"] = complete
            torch.save(
                {"scale": model.scale.detach().cpu(), "gradient": model.scale.grad.cpu()},
                args.root / "observations" / f"rank-{distributed.rank}.pt",
            )
            if args.scenario == "resume":
                first_epoch = sorted((args.root / "complete").glob("epoch-0001-*.pt"))[0]
                payload = torch.load(first_epoch, weights_only=True)
                restored = ScaleModel()
                restored.load_state_dict(payload["state_dict"], strict=True)
                resume_config = deep_merge(
                    config,
                    {
                        "training": {
                            "checkpoint": None,
                            "resume": str(first_epoch),
                            "output_dir": str(args.root / "resumed"),
                        }
                    },
                )
                resumed = train_model(restored, resume_config, payload)
                observations["resumed"] = distributed.broadcast_object(resumed)
                changed = deepcopy(resume_config)
                changed["training"]["batch_size"] = 1
                try:
                    train_model(restored, changed, payload)
                except ValueError as error:
                    observations["contract_error"] = str(error)
                else:
                    raise AssertionError("A changed per-GPU batch must reject resume")
        observations.update(sampler_epochs=sampler_epochs, sampler_indices=sampler_indices)
        (args.root / "observations" / f"rank-{distributed.rank}.json").write_text(
            json.dumps(observations), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

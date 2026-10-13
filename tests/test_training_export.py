from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from modern_derain.config import DEFAULTS, deep_merge
from modern_derain.data.sources import Sample, snapshot
from modern_derain.deployment.export import _runtime_comparison, export_model, inspect_graph
from modern_derain.models.quantized import INT8_BACKEND, _SignedInt8QDQ
from modern_derain.training.loop import _adjust_lr, save_checkpoint, train_model
from modern_derain.training.quantize import calibrate_model, restore_quantized_model, run_qat


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 3, 1))


def write_pairs(root):
    rng = np.random.default_rng(7)
    for side in ("rainy", "clean"):
        (root / "train" / side).mkdir(parents=True)
    for index in range(2):
        image = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        for side in ("rainy", "clean"):
            assert cv2.imwrite(str(root / "train" / side / f"{index}.png"), image)


def test_epoch_resume_restores_data_order_and_crop_rng(tmp_path):
    root = tmp_path / "data"
    write_pairs(root)
    torch.manual_seed(10)
    initial = tiny_model()
    weights = tmp_path / "weights.pt"
    save_checkpoint(weights, {"state_dict": initial.state_dict()})
    config = deep_merge(
        DEFAULTS,
        {
            "data": {"path": str(root)},
            "training": {
                "checkpoint": str(weights),
                "device": "cpu",
                "epochs": 2,
                "decay_start_epoch": 1,
                "crop_size": 16,
                "batch_size": 1,
                "output_dir": str(tmp_path / "complete"),
            },
        },
    )
    complete = train_model(deepcopy(initial), config, {})
    first_epoch = sorted((tmp_path / "complete").glob("epoch-0001-*.pt"))[0]
    payload = torch.load(first_epoch, weights_only=True)
    restored = tiny_model()
    restored.load_state_dict(payload["state_dict"], strict=True)
    resume_config = deep_merge(
        config,
        {
            "training": {
                "checkpoint": None,
                "resume": str(first_epoch),
                "output_dir": str(tmp_path / "resumed"),
            }
        },
    )
    resumed = train_model(restored, resume_config, payload)
    final = torch.load(complete["checkpoint"], weights_only=True)
    restarted = torch.load(resumed["checkpoint"], weights_only=True)
    for name, tensor in final["state_dict"].items():
        torch.testing.assert_close(tensor, restarted["state_dict"][name], rtol=0, atol=0)
    assert resumed["losses"] == complete["losses"][1:]
    (root / "train" / "rainy" / "0.png").touch()
    with pytest.raises(ValueError, match="fingerprint changed"):
        train_model(restored, resume_config, payload)


def test_schedule_updates_after_first_batch_of_epoch():
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter], lr=0.0002)
    training = {"lr": 0.0002, "epochs": 250, "decay_start_epoch": 50}
    _adjust_lr(optimizer, training, 49)
    assert optimizer.param_groups[0]["lr"] == 0.0002
    _adjust_lr(optimizer, training, 50)
    assert optimizer.param_groups[0]["lr"] == 0.0002
    _adjust_lr(optimizer, training, 51)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0002 * 199 / 200)
    _adjust_lr(optimizer, training, 250)
    assert optimizer.param_groups[0]["lr"] == 0


def test_checkpoint_never_overwrites(tmp_path):
    file = tmp_path / "checkpoint.pt"
    save_checkpoint(file, {"x": torch.ones(1)})
    original = file.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(file, {"x": torch.zeros(1)})
    assert file.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_interrupt_saves_weights_without_claiming_exact_resume(tmp_path):
    root = tmp_path / "data"
    write_pairs(root)
    model = tiny_model()
    checkpoint = tmp_path / "initial.pt"
    save_checkpoint(checkpoint, {"state_dict": model.state_dict()})
    forward = model.forward
    calls = 0

    def interrupt_after_one_step(image):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return forward(image)

    model.forward = interrupt_after_one_step
    config = deep_merge(
        DEFAULTS,
        {
            "data": {"path": str(root)},
            "training": {
                "checkpoint": str(checkpoint),
                "device": "cpu",
                "epochs": 2,
                "decay_start_epoch": 1,
                "crop_size": 16,
                "batch_size": 1,
                "output_dir": str(tmp_path / "interrupted"),
            },
        },
    )
    with pytest.raises(KeyboardInterrupt):
        train_model(model, config, {})
    interrupted = tmp_path / "interrupted" / "interrupted.pt"
    payload = torch.load(interrupted, weights_only=True)
    assert payload["interrupted"] and payload["training_state"]["step"] == 1
    assert not payload["training_state"]["resume_allowed"]
    restored = tiny_model()
    restored.load_state_dict(payload["state_dict"], strict=True)
    assert torch.isfinite(restored(torch.rand(1, 3, 16, 16))).all()
    config["training"].update(resume=str(interrupted), checkpoint=None)
    with pytest.raises(ValueError, match="cannot resume"):
        train_model(restored, config, payload)


def quantized_tiny_model():
    model = tiny_model()
    sample = torch.rand(1, 3, 16, 16)

    def forward_loop(prepared):
        with torch.no_grad():
            prepared(sample)

    model, policy = calibrate_model(model, forward_loop, [])
    return model, sample, policy


def test_modelopt_restore_scales_and_backward(tmp_path):
    import modelopt.torch.opt as mto

    model, sample, policy = quantized_tiny_model()
    assert all(row["axis"] in (0, None) for row in policy["quantizers"] if row["enabled"])
    assert policy["numeric_backend"] == INT8_BACKEND
    payload = {"modelopt_state": mto.modelopt_state(model), "state_dict": model.state_dict()}
    checkpoint = tmp_path / "qat.pt"
    save_checkpoint(checkpoint, payload)
    restored = restore_quantized_model(tiny_model(), torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(model(sample), restored(sample), rtol=0, atol=0)
    restored(sample).square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in restored.parameters()
    )
    corrupted = deepcopy(payload)
    scale_name = next(name for name in corrupted["state_dict"] if name.endswith("._amax"))
    corrupted["state_dict"][scale_name].fill_(float("nan"))
    with pytest.raises(ValueError, match="calibration scale"):
        restore_quantized_model(tiny_model(), corrupted)


def test_qdq_half_bin_rounding_and_clipped_gradient():
    values = torch.tensor(
        [-40, -32, -0.625, -0.375, -0.125, 0, 0.125, 0.375, 0.625, 31.75, 40],
        dtype=torch.float32,
        requires_grad=True,
    )
    result = _SignedInt8QDQ.apply(values, torch.tensor(0.25), -1)
    expected = torch.tensor([-32, -32, -0.5, -0.5, 0, 0, 0, 0.5, 0.5, 31.75, 31.75])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0]))
    weights = torch.tensor([[[[0.375]]], [[[0.625]]]], requires_grad=True)
    per_channel = _SignedInt8QDQ.apply(weights, torch.tensor([0.25, 0.5]), 0)
    torch.testing.assert_close(per_channel, torch.tensor([[[[0.5]]], [[[0.5]]]]))
    half = torch.tensor([-0.375, 0.125, 0.375], dtype=torch.float16, requires_grad=True)
    half_output = _SignedInt8QDQ.apply(half, torch.tensor(0.25), -1)
    assert half_output.dtype == torch.float16 and torch.isfinite(half_output).all()
    torch.testing.assert_close(half_output, torch.tensor([-0.5, 0, 0.5], dtype=torch.float16))
    half_output.sum().backward()
    assert half.grad.dtype == torch.float16 and torch.isfinite(half.grad).all()


@pytest.mark.parametrize("quantized", [False, True])
def test_fp_and_qdq_export_preserve_output(tmp_path, quantized):
    if quantized:
        model, sample, _ = quantized_tiny_model()
    else:
        model, sample = tiny_model(), torch.rand(1, 3, 16, 16)
    output = tmp_path / "model.onnx"
    result = export_model(model, sample, output, quantized=quantized)
    assert result["onnx_checker"] == "passed"
    assert result["output_comparison"]["status"] == "passed"
    assert result["exporter"] == ("torchscript" if quantized else "dynamo")
    if quantized:
        assert result["qdq_arithmetic"]["status"] == "passed"
        assert result["graph"]["operator_counts"]["QuantizeLinear"] > 0
        assert result["graph"]["operator_counts"]["DequantizeLinear"] > 0
    with pytest.raises(FileExistsError):
        export_model(model, sample, output, quantized=quantized)
    with pytest.raises(ValueError, match=".onnx extension"):
        export_model(model, sample, tmp_path / "bad.json", quantized=quantized)
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("scale", [np.array([], np.float32), np.array(0.25, np.float64)])
def test_graph_rejects_empty_and_wrong_dtype_scales(scale):
    from onnx import helper, numpy_helper

    nodes = [
        helper.make_node("QuantizeLinear", ["rainy", "scale", "zero"], ["quantized"]),
        helper.make_node("DequantizeLinear", ["quantized", "scale", "zero"], ["clean"]),
    ]
    graph = helper.make_model(
        helper.make_graph(
            nodes,
            "invalid-scale",
            [],
            [],
            initializer=[
                numpy_helper.from_array(scale, "scale"),
                numpy_helper.from_array(np.array(0, np.int8), "zero"),
            ],
        )
    )
    with pytest.raises(ValueError, match="scale"):
        inspect_graph(graph, quantized=True)


def test_quantized_full_graph_drift_is_reported_without_relaxing_fp_tolerance(tmp_path):
    import onnx
    from onnx import helper

    path = tmp_path / "identity.onnx"
    shape = [1, 3, 16, 16]
    graph = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["rainy"], ["clean"])],
            "identity",
            [helper.make_tensor_value_info("rainy", onnx.TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("clean", onnx.TensorProto.FLOAT, shape)],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.save(graph, path)
    sample = torch.ones(shape)
    result = _runtime_comparison(path, sample, sample + 0.01, quantized=True)
    assert result["status"] == "differs" and result["quality_pending"]
    assert result["absolute_tolerance"] == 3e-5
    with pytest.raises(ValueError, match="output mismatch"):
        _runtime_comparison(path, sample, sample + 0.01, quantized=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scale export regression")
def test_cuda_export_retains_exact_calibrated_scale_bits(tmp_path):
    import onnx
    from onnx import numpy_helper

    model, sample, _ = quantized_tiny_model()
    model = model.cuda()
    # This calibrated amax previously lost one scale ULP in legacy constant folding.
    value = np.array(1056325733, np.uint32).view(np.float32)
    model[0].input_quantizer.amax = torch.tensor(value, device="cuda")
    result = export_model(model, sample.cuda(), tmp_path / "cuda.onnx", quantized=True)
    assert result["qdq_arithmetic"]["status"] == "passed"
    graph = onnx.load(result["onnx"])
    node = next(
        node
        for node in graph.graph.node
        if node.op_type == "QuantizeLinear" and "/0/input_quantizer/" in node.name
    )
    values = {value.name: numpy_helper.to_array(value) for value in graph.graph.initializer}
    for constant in graph.graph.node:
        if constant.op_type == "Constant":
            values[constant.output[0]] = numpy_helper.to_array(constant.attribute[0].t)
    assert values[node.input[1]].view(np.uint32) == 997732551


def test_calibration_only_does_not_require_train(tmp_path, monkeypatch):
    from modern_derain.models import kpn

    calibration = tmp_path / "data" / "calibration"
    calibration.mkdir(parents=True)
    assert cv2.imwrite(str(calibration / "daytime.png"), np.full((20, 20, 3), 90, np.uint8))
    checkpoint = tmp_path / "initial.pt"
    model = tiny_model()
    save_checkpoint(checkpoint, {"state_dict": model.state_dict()})
    prior_rainy, prior_clean = tmp_path / "prior-rainy.png", tmp_path / "prior-clean.png"
    for image in (prior_rainy, prior_clean):
        assert cv2.imwrite(str(image), np.full((20, 20, 3), 80, np.uint8))
    training_snapshot = snapshot([Sample("previous.png", prior_rainy, prior_clean, "previous")])
    monkeypatch.setattr(
        kpn,
        "load_checkpoint",
        lambda *args, **kwargs: (model, {"data_snapshot": training_snapshot}),
    )
    config = deep_merge(
        DEFAULTS,
        {
            "data": {"path": str(tmp_path / "data")},
            "training": {
                "checkpoint": str(checkpoint),
                "device": "cpu",
                "output_dir": str(tmp_path / "ptq"),
            },
            "quantization": {"steps": 0, "calibration_samples": 1, "exclude_modules": []},
            "export": {"shape": [1, 3, 16, 16]},
        },
    )
    result = run_qat(config)
    assert result["kind"] == "ptq" and result["steps"] == 0
    payload = torch.load(result["checkpoint"], weights_only=True)
    assert payload["calibration"]["samples"] == 1
    assert len(payload["calibration"]["data_snapshot"]) == 1
    assert payload["data_snapshot"] == training_snapshot
    assert payload["config"]["quantization"] == config["quantization"]
    assert "training_state" not in payload
    for side in ("rainy", "clean"):
        folder = tmp_path / "data" / "val" / side
        folder.mkdir(parents=True)
        assert cv2.imwrite(str(folder / "daytime.png"), np.full((20, 20, 3), 90, np.uint8))
    with pytest.raises(ValueError, match="leakage"):
        run_qat(config)

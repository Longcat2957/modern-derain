"""Small CPU checks against byte-preserved upstream oracles."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from modern_derain.models.kernel import KernelConv
from modern_derain.models.kpn import KPN, MODEL_CONFIG, UPSTREAM_COMMIT, load_checkpoint
from modern_derain.training.losses import SSIM, DerainLoss

REFERENCE = Path(__file__).parent / "reference" / "EfficientDeRain" / UPSTREAM_COMMIT


def oracle(filename):
    spec = importlib.util.spec_from_file_location(
        "derain_oracle_" + filename.replace("/", "_"), REFERENCE / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def strict_cuda_fp32():
    convolution = torch.backends.cudnn.allow_tf32
    matrix = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32 = convolution
    torch.backends.cuda.matmul.allow_tf32 = matrix


def test_oracle_bytes():
    provenance = json.loads((REFERENCE / "PROVENANCE.json").read_text())
    for name, metadata in provenance["files"].items():
        content = (REFERENCE / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]
        header = f"blob {len(content)}\0".encode()
        assert hashlib.sha1(header + content).hexdigest() == metadata["git_blob"]


@pytest.mark.reference
@pytest.mark.parametrize("rate", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(1, 3, 2, 3), (2, 3, 5, 7)])
def test_kernel_forward_input_and_core_gradient(rate, shape):
    generator = torch.Generator().manual_seed(23)
    frames = torch.randn(shape, generator=generator, requires_grad=True)
    core = torch.randn((shape[0], 27, *shape[-2:]), generator=generator, requires_grad=True)
    original_frames = frames.detach().clone().requires_grad_()
    original_core = core.detach().clone().requires_grad_()
    actual = KernelConv()(frames, core, white_level=2.0, rate=rate)
    expected = oracle("network.py").KernelConv([3])(original_frames, original_core, 2.0, rate)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn(shape, generator=generator)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(frames.grad, original_frames.grad, atol=0, rtol=0)
    torch.testing.assert_close(core.grad, original_core.grad, atol=0, rtol=0)
    assert torch.isfinite(actual).all()


def test_kernel_color_order_and_zero_boundary():
    frames = torch.stack([torch.full((3, 4), float(channel + 1)) for channel in range(3)])[None]
    core = torch.zeros(1, 27, 3, 4)
    core[:, 4] = 2
    core[:, 13] = 3
    core[:, 22] = 4
    torch.testing.assert_close(
        KernelConv()(frames, core), frames * torch.tensor([2, 3, 4])[None, :, None, None]
    )
    core.zero_()
    core[:, (0, 9, 18)] = 1
    shifted = KernelConv()(frames, core, rate=4)
    assert shifted.count_nonzero() == 0


@pytest.mark.reference
def test_kpn_forward_and_all_parameter_gradients():
    torch.manual_seed(7)
    rebuilt = KPN()
    original = oracle("network.py").KPN(**MODEL_CONFIG)
    original.load_state_dict(rebuilt.state_dict(), strict=True)
    assert set(original.state_dict()) == set(rebuilt.state_dict())
    image = torch.rand(1, 3, 16, 32, requires_grad=True)
    original_image = image.detach().clone().requires_grad_()
    actual = rebuilt(image)
    expected = original(original_image, original_image, 1.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(image.grad, original_image.grad, atol=2e-7, rtol=2e-5)
    for name, parameter in rebuilt.named_parameters():
        reference = dict(original.named_parameters())[name]
        torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-7, rtol=2e-5)
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ssim_window_loss_and_gradients(dtype):
    torch.manual_seed(9)
    first = torch.rand(2, 3, 13, 17, dtype=dtype, requires_grad=True)
    second = torch.rand_like(first, requires_grad=True)
    first_reference = first.detach().clone().requires_grad_()
    second_reference = second.detach().clone().requires_grad_()
    reference = oracle("pytorch_ssim/__init__.py")
    current = SSIM()
    torch.testing.assert_close(current.window, reference.create_window(11, 1), atol=0, rtol=0)
    assert current.state_dict() == {}
    actual = DerainLoss()(first, second)
    expected = torch.nn.functional.l1_loss(
        first_reference, second_reference
    ) - 0.2 * reference.SSIM()(second_reference, first_reference)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(first.grad, first_reference.grad, atol=0, rtol=0)
    torch.testing.assert_close(second.grad, second_reference.grad, atol=0, rtol=0)


def test_low_precision_dynamic_filter_and_loss_accumulate_fp32():
    frames = torch.rand(1, 3, 16, 16, dtype=torch.bfloat16, requires_grad=True)
    core = torch.rand(1, 27, 16, 16, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = KernelConv()(frames, core)
        loss = DerainLoss()(output, frames)
    assert output.dtype == loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(frames.grad).all() and torch.isfinite(core.grad).all()


def test_checkpoint_strictness_and_metadata(tmp_path):
    model = KPN()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"model": MODEL_CONFIG, "preprocessing": {"protocol": "official-ceil16"}},
        },
        checkpoint,
    )
    restored, payload = load_checkpoint(checkpoint, MODEL_CONFIG, "official-ceil16")
    assert all(
        torch.equal(tensor, restored.state_dict()[key])
        for key, tensor in model.state_dict().items()
    )
    with pytest.raises(ValueError, match="preprocessing"):
        load_checkpoint(checkpoint, preprocessing="fixed-resize")
    payload["state_dict"]["outc.bias"] = payload["state_dict"]["outc.bias"].half()
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="implicit casts"):
        load_checkpoint(checkpoint)
    payload["state_dict"].pop("outc.bias")
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(checkpoint)


@pytest.mark.parametrize("shape", [(1, 1, 16, 16), (1, 3, 15, 16), (1, 3, 17, 32)])
def test_kpn_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        KPN()(torch.rand(shape))


@pytest.mark.reference
def test_verified_official_checkpoint_when_present():
    source = Path("weights/upstream-v3-rain1400.pth")
    packaged = Path("weights/efficientderain-rain1400-v3.pt")
    if not source.exists() or not packaged.exists():
        pytest.skip("official large assets are local/Release files; tests never download them")
    state = torch.load(source, map_location="cpu", weights_only=True)
    modern, payload = load_checkpoint(packaged)
    reference = oracle("network.py").KPN(**MODEL_CONFIG)
    reference.load_state_dict(state, strict=True)
    for name, tensor in state.items():
        assert tensor.dtype == payload["state_dict"][name].dtype
        assert torch.equal(tensor, payload["state_dict"][name])
    modern.eval()
    reference.eval()
    sample = torch.linspace(0, 1, 3 * 16 * 32).reshape(1, 3, 16, 32)
    with torch.inference_mode():
        torch.testing.assert_close(modern(sample), reference(sample, sample), atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.reference
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is required")
def test_cuda_official_checkpoint_gradients_and_amp(strict_cuda_fp32):
    source = Path("weights/upstream-v3-rain1400.pth")
    packaged = Path("weights/efficientderain-rain1400-v3.pt")
    if not source.exists() or not packaged.exists():
        pytest.skip("official large assets are local/Release files")
    torch.manual_seed(47)
    model, _ = load_checkpoint(packaged, device="cuda")
    reference = oracle("network.py").KPN(**MODEL_CONFIG).cuda()
    reference.load_state_dict(
        torch.load(source, map_location="cpu", weights_only=True), strict=True
    )
    for shape in ((1, 3, 16, 32), (1, 3, 256, 256)):
        sample = torch.rand(shape, device="cuda")
        with torch.inference_mode():
            torch.testing.assert_close(model(sample), reference(sample, sample), atol=0, rtol=0)
    image = torch.rand(1, 3, 16, 32, device="cuda", requires_grad=True)
    other = image.detach().clone().requires_grad_()
    probe = torch.randn_like(image)
    (model(image) * probe).sum().backward()
    (reference(other, other) * probe).sum().backward()
    torch.testing.assert_close(image.grad, other.grad, atol=1e-5, rtol=1e-4)
    for name, parameter in model.named_parameters():
        expected = dict(reference.named_parameters())[name].grad
        torch.testing.assert_close(parameter.grad, expected, atol=1e-5, rtol=1e-4)
    model.zero_grad(set_to_none=True)
    image = torch.rand(1, 3, 16, 32, device="cuda", requires_grad=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(image)
        loss = DerainLoss().cuda()(output, torch.rand_like(image))
    assert output.dtype == torch.float16 and loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(output).all() and torch.isfinite(image.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())

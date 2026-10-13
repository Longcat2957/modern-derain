"""Fixed-shape FP32 and ModelOpt explicit-Q/DQ ONNX delivery bundles."""

import os
import sys
import tempfile
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from modern_derain.artifacts import run_metadata, sha256, write_json
from modern_derain.config import DEFAULT_PREPROCESSING


def inspect_graph(graph, *, quantized: bool) -> dict:
    counts = Counter(node.op_type for node in graph.graph.node)
    if quantized and (not counts["QuantizeLinear"] or not counts["DequantizeLinear"]):
        raise ValueError("INT8 export did not preserve QuantizeLinear/DequantizeLinear")
    values = {value.name: numpy_helper.to_array(value) for value in graph.graph.initializer}
    for node in graph.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    values[node.output[0]] = numpy_helper.to_array(attribute.t)
    rows = []
    for node in graph.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        scale = values.get(node.input[1])
        zero_point = values.get(node.input[2]) if len(node.input) > 2 else None
        if (
            scale is None
            or scale.dtype != np.float32
            or not scale.size
            or not np.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError(f"Q/DQ node {node.name} has invalid/nonconstant scale")
        if zero_point is None or zero_point.dtype != np.int8 or np.any(zero_point):
            raise ValueError(f"Q/DQ node {node.name} is not symmetric signed INT8")
        axis = next((attribute.i for attribute in node.attribute if attribute.name == "axis"), None)
        if scale.ndim > 1 or (scale.size > 1 and axis != 0):
            raise ValueError(f"Q/DQ node {node.name} has unsupported weight axis")
        rows.append(
            {
                "name": node.name,
                "op": node.op_type,
                "scale_shape": list(scale.shape),
                "scale_dtype": str(scale.dtype),
                "zero_point_dtype": str(zero_point.dtype),
                "axis": axis,
            }
        )
    return {"operator_counts": dict(counts), "qdq": rows}


def _runtime_comparison(path: Path, example, expected, *, quantized: bool) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"status": "not performed", "reason": "onnxruntime is available in the dev group"}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    if quantized:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"rainy": example.detach().cpu().numpy()})[0]
    reference = expected.detach().cpu().numpy()
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError("ONNX Runtime returned a nonfinite output or incorrect shape")
    error = np.abs(actual - reference)
    equivalent = np.allclose(actual, reference, rtol=3e-4, atol=3e-5)
    if not quantized and not equivalent:
        raise ValueError(f"ONNX output mismatch: maximum absolute error {error.max()}")
    result = {
        "status": "passed" if equivalent else "differs",
        "provider": "CPUExecutionProvider",
        "graph_optimization": "disabled" if quantized else "default",
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "max_relative_error": float((error / np.maximum(np.abs(reference), 1e-7)).max()),
        "absolute_tolerance": 3e-5,
        "relative_tolerance": 3e-4,
        "finite": True,
        "reference_device": str(expected.device),
    }
    if quantized:
        result["quality_pending"] = True
        result["difference_reason"] = (
            "Floating convolution reductions and device FP32 division rounding can "
            "cross INT8 half-bin thresholds; full-graph equivalence and deployment quality "
            "require separate validation"
        )
        optimized_options = ort.SessionOptions()
        optimized_options.intra_op_num_threads = 1
        optimized = ort.InferenceSession(
            str(path), sess_options=optimized_options, providers=["CPUExecutionProvider"]
        )
        optimized_output = optimized.run(None, {"rainy": example.detach().cpu().numpy()})[0]
        if optimized_output.shape != reference.shape or not np.isfinite(optimized_output).all():
            raise ValueError("Optimized ONNX Runtime returned an invalid output")
        optimized_error = np.abs(optimized_output - reference)
        result["optimized_execution"] = {
            "status": "passed"
            if np.allclose(optimized_output, reference, rtol=3e-4, atol=3e-5)
            else "differs",
            "max_absolute_error": float(optimized_error.max()),
            "mean_absolute_error": float(optimized_error.mean()),
            "quality_evaluation": "not performed",
            "tensorrt_comparison": "not performed",
        }
    return result


def _verify_qdq_arithmetic(model, example, graph) -> dict:
    """Check Q/DQ with identical inputs, independently of convolution reductions."""
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "Q/DQ arithmetic verification requires onnxruntime; include the project's dev "
            "dependency group when running export"
        ) from error
    from modelopt.torch.quantization.nn import TensorQuantizer

    captured, handles = [], []
    original_device = next(model.parameters()).device
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and module.is_enabled:

            def capture(quantizer, arguments, output, name=name):
                captured.append(
                    (
                        name,
                        arguments[0].detach().numpy().copy(),
                        output.detach().numpy().copy(),
                        (quantizer.amax.detach().double() / 127).float().numpy().flatten(),
                    )
                )

            handles.append(module.register_forward_hook(capture))
    try:
        model.to("cpu")
        with torch.no_grad():
            model(example.detach().cpu())
    finally:
        for handle in handles:
            handle.remove()
        model.to(original_device)
    quantizers = [node for node in graph.graph.node if node.op_type == "QuantizeLinear"]
    if not captured or len(captured) != len(quantizers):
        raise ValueError("Exported Q/DQ count differs from the executed calibrated quantizers")
    constants = {value.name: value for value in graph.graph.initializer}
    for node in graph.graph.node:
        if node.op_type == "Constant":
            value = next(attribute.t for attribute in node.attribute if attribute.name == "value")
            constants[node.output[0]] = numpy_helper.from_array(
                numpy_helper.to_array(value), node.output[0]
            )
    nodes, inputs, outputs, initializers, feed, references = [], [], [], {}, {}, []
    for node, (name, value, reference, scale) in zip(quantizers, captured, strict=True):
        scope = node.name.removesuffix("/QuantizeLinear").strip("/").split("/")
        scope = [
            item
            for index, item in enumerate(scope)
            if index + 1 == len(scope) or not scope[index + 1].startswith(item + ".")
        ]
        if ".".join(scope) != name:
            raise ValueError(f"Exported Q/DQ mapping differs from the calibrated module: {name}")
        exported_scale = numpy_helper.to_array(constants[node.input[1]]).flatten()
        if not np.array_equal(exported_scale, scale):
            raise ValueError(f"Exported Q/DQ scale bits differ from the calibrated module: {name}")
        dequantizer = next(
            item
            for item in graph.graph.node
            if item.op_type == "DequantizeLinear" and item.input[0] == node.output[0]
        )
        nodes.extend([node, dequantizer])
        for identifier in node.input[1:]:
            initializers[identifier] = constants[identifier]
        feed[node.input[0]] = value
        inputs.append(
            onnx.helper.make_tensor_value_info(node.input[0], onnx.TensorProto.FLOAT, value.shape)
        )
        outputs.append(
            onnx.helper.make_tensor_value_info(
                dequantizer.output[0], onnx.TensorProto.FLOAT, reference.shape
            )
        )
        references.append((name, reference))
    standalone = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes, "same-input-qdq", inputs, outputs, initializer=list(initializers.values())
        ),
        opset_imports=graph.opset_import,
        ir_version=graph.ir_version,
    )
    onnx.checker.check_model(standalone)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        standalone.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, feed)
    for output, (name, reference) in zip(actual, references, strict=True):
        if not np.array_equal(output, reference) or not np.isfinite(output).all():
            raise ValueError(f"Exported Q/DQ arithmetic differs for identical CPU inputs: {name}")
    return {
        "status": "passed",
        "provider": "CPUExecutionProvider",
        "reference_device": "cpu",
        "quantizers": len(captured),
        "comparison": "bit-for-bit identical captured inputs and FP32 scales",
        "full_graph_equivalence": "reported separately",
    }


def export_model(model, example, output: Path, *, quantized: bool = False, opset: int = 18) -> dict:
    output = Path(output).resolve()
    if output.suffix.lower() != ".onnx":
        raise ValueError("ONNX export output must use the .onnx extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    external_path = output.with_name(output.name + ".data")
    if output.exists() or external_path.exists():
        raise FileExistsError(f"Export destination already exists: {output}")
    model.eval()
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        expected = model(example)
    if not torch.isfinite(expected).all():
        raise ValueError("The PyTorch export model produced nonfinite output")
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix=".onnx-", dir=output.parent) as folder:
            staging = Path(folder)
            raw = staging / "raw.onnx"
            # The fixed ModelOpt backend supplies explicit symbolic Q/DQ functions.
            # Floating export uses the current torch.export/dynamo path.
            with redirect_stdout(sys.stderr):
                torch.onnx.export(
                    model,
                    (example,),
                    str(raw),
                    input_names=["rainy"],
                    output_names=["clean"],
                    opset_version=opset,
                    dynamo=not quantized,
                    external_data=True,
                )
            graph = onnx.load(raw, load_external_data=True)
            actual_opset = next(item.version for item in graph.opset_import if not item.domain)
            if actual_opset != opset:
                raise ValueError(
                    f"Exporter produced opset {actual_opset}, requested {opset}; select a supported opset"
                )
            details = inspect_graph(graph, quantized=quantized)
            arithmetic = _verify_qdq_arithmetic(model, example, graph) if quantized else None
            staged_output = staging / output.name
            onnx.save_model(
                graph,
                str(staged_output),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_path.name,
                size_threshold=1024,
                convert_attribute=False,
            )
            onnx.checker.check_model(str(staged_output), full_check=True)
            comparison = _runtime_comparison(staged_output, example, expected, quantized=quantized)
            for file in [staged_output, staging / external_path.name]:
                if file.exists():
                    destination = output.parent / file.name
                    os.link(file, destination)
                    published.append(destination)
    except BaseException:
        for file in published:
            file.unlink()
        raise
    return {
        "onnx": str(output),
        "onnx_sha256": sha256(output),
        "external_data": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in published
            if path != output
        ],
        "opset": next(item.version for item in graph.opset_import if not item.domain),
        "exporter": "torchscript" if quantized else "dynamo",
        "exporter_reason": "fixed-scale ModelOpt backend symbolic Q/DQ support"
        if quantized
        else "current PyTorch exporter",
        "onnx_checker": "passed",
        "reference_policy": "FP32 with CUDA TF32 disabled",
        "output_comparison": comparison,
        "qdq_arithmetic": arithmetic,
        "graph": details,
        "tensorrt_target_validation": "not performed",
    }


def run_export(config: dict) -> dict:
    from modern_derain.models.assets import resolve_checkpoint
    from modern_derain.models.kpn import load_checkpoint

    export = config["export"]
    shape = export["shape"]
    if shape is None:
        raise ValueError(
            "Export requires export.shape=[1,3,H,W]; target resolution is not inferred"
        )
    output = Path(export["output"]).resolve()
    if output.suffix.lower() != ".onnx":
        raise ValueError("ONNX export output must use the .onnx extension")
    manifest_path = output.with_suffix(".json")
    if manifest_path.exists():
        raise FileExistsError(f"Export manifest already exists: {manifest_path}")
    path = resolve_checkpoint(export["checkpoint"])
    model, payload = load_checkpoint(
        path, config_model=config["model"], preprocessing=DEFAULT_PREPROCESSING
    )
    quantized = export["precision"] == "int8"
    if quantized != ("modelopt_state" in payload):
        raise ValueError("Export precision does not match checkpoint kind/modelopt_state")
    if quantized:
        calibration = payload.get("calibration")
        if not calibration or calibration["shape"] != shape:
            raise ValueError("INT8 export shape must match the checkpoint calibration shape")
    device = torch.device(export["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select export.device='cpu'")
    model = model.to(device)
    generator = torch.Generator().manual_seed(1234)
    example = torch.rand(shape, generator=generator).to(device)
    result = export_model(model, example, output, quantized=quantized, opset=export["opset"])
    metadata = run_metadata(config, path)
    metadata["environment"]["torch_runtime"] = {
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
    }
    metadata.update(result)
    metadata.update(
        {
            "input": {
                "name": "rainy",
                "shape": shape,
                "dtype": "float32",
                "color": "RGB",
                "range": [0.0, 1.0],
            },
            "output": {"name": "clean", "shape": shape, "dtype": "float32", "model_clip": False},
            "preprocessing": {
                **DEFAULT_PREPROCESSING,
                "protocol": "fixed",
                "resize": "opencv-bilinear-to-fixed-HW",
                "padding": None,
                "restore": "official recovery to original H/W",
            },
            "precision": export["precision"],
            "typing_policy": "explicit-QDQ-FP32-floating" if quantized else "FP32",
            "calibration": payload.get("calibration"),
        }
    )
    write_json(manifest_path, metadata)
    return {"command": "export", "manifest": str(manifest_path), **result}

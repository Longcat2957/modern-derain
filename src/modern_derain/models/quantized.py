"""A fixed-scale ModelOpt backend matching ONNX's signed INT8 Q/DQ formula."""

import torch

INT8_BACKEND = "modern_derain_int8_qdq_divide_v2"


class _SignedInt8QDQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale, axis):
        broadcast = scale.view(-1, *([1] * (inputs.ndim - 1))) if axis == 0 else scale
        normalized = inputs.float() / broadcast
        ctx.save_for_backward((normalized >= -128) & (normalized <= 127))
        return (normalized.round().clamp(-128, 127) * broadcast).to(inputs.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (mask,) = ctx.saved_tensors
        return gradient * mask, None, None

    @staticmethod
    def symbolic(graph, inputs, scale, axis):
        shape = (inputs.type().sizes()[0],) if axis == 0 else ()
        zero = graph.op("Constant", value_t=torch.zeros(shape, dtype=torch.int8))
        attributes = {"axis_i": 0} if axis == 0 else {}
        dtype = inputs.type().scalarType()
        source = graph.op("Cast", inputs, to_i=1) if dtype != "Float" else inputs
        quantized = graph.op("QuantizeLinear", source, scale, zero, **attributes)
        result = graph.op("DequantizeLinear", quantized, scale, zero, **attributes)
        if dtype != "Float":
            result = graph.op("Cast", result, to_i={"Half": 10, "BFloat16": 16}[dtype])
        return result


def int8_qdq(inputs, quantizer):
    # Reciprocal multiply in ModelOpt/native fake quant changes rounding at INT8 ties.
    # Round the fixed scale once to FP32 identically on CPU/CUDA and during export.
    scale = (quantizer.amax.detach().double() / 127.0).float()
    if quantizer.axis is not None:
        scale = scale.flatten()
    if torch.onnx.is_in_onnx_export():
        # Legacy ONNX folding can round the double division as FP32. Embed the
        # calibrated scale bits directly; eager and traced arithmetic stay equal.
        scale = torch.tensor(
            scale.detach().cpu().tolist(), dtype=torch.float32, device=inputs.device
        )
    return _SignedInt8QDQ.apply(inputs, scale, -1 if quantizer.axis is None else 0)


def register_int8_backend():
    from modelopt.torch.quantization.nn import is_registered_quant_backend, register_quant_backend

    if not is_registered_quant_backend(INT8_BACKEND):
        register_quant_backend(INT8_BACKEND, int8_qdq)

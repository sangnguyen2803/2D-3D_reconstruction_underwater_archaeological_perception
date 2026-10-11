"""Actual trained model export, dynamic-batch parity and local GPU timing."""

import time

import numpy as np
import torch

from underwater_vision.matchability.model import load_model
from underwater_vision.matchability.protocol import digest, new_run
from underwater_vision.utils.io import write_json


def export_matchability(
    checkpoint, output, batch=1, input_hw=(720, 960), warmup=5, repeats=20, measure=True
):
    import onnx
    import onnxruntime as ort

    root = new_run(output)
    torch.set_num_threads(4)
    model, saved = load_model(checkpoint)
    if saved["model_config"]["kind"] != "segformer_b0":
        raise ValueError("Main deployment model is the SegFormer-B0 matchability network")
    h, w = input_hw
    torch.manual_seed(42)
    image = torch.rand(batch, 3, h, w)
    path = root / "matchability_v3.onnx"
    torch.onnx.export(
        model,
        image,
        str(path),
        input_names=["rgb"],
        output_names=["logits"],
        dynamic_axes={"rgb": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(image).numpy()
    actual = session.run(None, {"rgb": image.numpy()})[0]
    parity = bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-3))
    if not parity:
        raise ValueError(
            f"ONNX parity failed, max absolute error={np.max(np.abs(actual - expected))}"
        )
    if session.run(None, {"rgb": np.repeat(image.numpy(), 2, 0)})[0].shape[0] != batch * 2:
        raise ValueError("ONNX dynamic batch test failed")
    timing = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) if measure else []
    for device in devices:
        for precision in (["fp32", "fp16"] if device == "cuda" else ["fp32"]):
            dtype = torch.float16 if precision == "fp16" else torch.float32
            model = model.to(device=device, dtype=dtype)
            x = image.to(device=device, dtype=dtype)
            with torch.inference_mode():
                for _ in range(warmup):
                    model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                latency = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    out = model(x)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    latency.append((time.perf_counter() - start) * 1000)
                result = out.float().cpu().numpy()
            timing.append(
                {
                    "device": torch.cuda.get_device_name() if device == "cuda" else "CPU",
                    "precision": precision,
                    "batch": batch,
                    "input_hw": list(input_hw),
                    "warmup": warmup,
                    "repeats": repeats,
                    "latency_mean_ms": float(np.mean(latency)),
                    "latency_p95_ms": float(np.quantile(latency, 0.95)),
                    "throughput_images_per_second": batch * 1000 / float(np.mean(latency)),
                    "peak_vram_bytes": (
                        torch.cuda.max_memory_allocated() if device == "cuda" else None
                    ),
                    "max_absolute_logit_difference_from_cpu_fp32": float(
                        np.max(np.abs(result - expected))
                    ),
                    "scope": "PyTorch model only; excludes image decoding/resizing, transfers, map sampling, API",
                }
            )
    report = {
        "status": "trained",
        "feature_spec": "dense_matchability_v3",
        "checkpoint_hash": digest(checkpoint),
        "onnx_sha256": digest(path),
        "temperature": saved["temperature"],
        "input_hw": list(input_hw),
        "input": "NCHW float RGB [0,1]; normalization embedded; fixed spatial size, dynamic batch",
        "parity": parity,
        "max_absolute_logit_error": float(np.max(np.abs(actual - expected))),
        "tolerance": {"atol": 1e-4, "rtol": 1e-3},
        "dynamic_batch_checked": True,
        "timing": timing,
        "training_images": saved["training_images"],
        "validation_images": saved["validation_images"],
    }
    write_json(root / "deployment_v3.json", report)
    return report

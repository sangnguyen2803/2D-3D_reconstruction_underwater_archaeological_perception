"""ONNX export, numerical parity and measured GPU deployment timing."""

import time

import numpy as np
import torch

from underwater_vision.semantic.model import load_model
from underwater_vision.semantic.protocol import digest_file, new_run
from underwater_vision.utils.io import write_json


def export_onnx(checkpoint, output, batch=1, warmup=5, repeats=20):
    import onnx
    import onnxruntime as ort

    root = new_run(output)
    model, saved = load_model(checkpoint)
    if saved["model_config"]["kind"] != "segformer":
        raise ValueError(
            "Deployment target is the SegFormer student; DINO remains a research baseline"
        )
    h, w = saved.get("size", [384, 512])
    torch.manual_seed(42)
    image = torch.rand(batch, 3, h, w)
    path = root / "segmentation_v2.onnx"
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
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        reference = model(image).numpy()
    actual = session.run(None, {"rgb": image.numpy()})[0]
    maximum = float(np.max(np.abs(actual - reference)))
    parity = bool(np.allclose(reference, actual, atol=1e-4, rtol=1e-3))
    if not parity:
        raise ValueError(f"ONNX parity failed: max abs error {maximum}")
    # Exercise the exported dynamic batch independently of the traced batch.
    doubled = np.concatenate([image.numpy(), image.numpy()])
    if session.run(None, {"rgb": doubled})[0].shape[0] != 2 * batch:
        raise ValueError("ONNX dynamic batch failed")
    timing = []
    devices = ["cpu"] + ["cuda"] if torch.cuda.is_available() else ["cpu"]
    for device in devices:
        model = model.to(device)
        for precision in (["fp32", "fp16"] if device == "cuda" else ["fp32"]):
            model = model.half() if precision == "fp16" else model.float()
            x = image.to(
                device=device, dtype=torch.float16 if precision == "fp16" else torch.float32
            )
            with torch.inference_mode():
                for _ in range(warmup):
                    model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                latencies = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    model(x)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - start) * 1000)
            latency = float(np.mean(latencies))
            timing.append(
                {
                    "device": torch.cuda.get_device_name() if device == "cuda" else "CPU",
                    "precision": precision,
                    "batch": batch,
                    "input_size": [h, w],
                    "warmup": warmup,
                    "repeats": repeats,
                    "mean_latency_ms": latency,
                    "p95_latency_ms": float(np.quantile(latencies, 0.95)),
                    "throughput_images_per_second": batch * 1000 / latency,
                    "peak_vram_bytes": (
                        torch.cuda.max_memory_allocated() if device == "cuda" else None
                    ),
                    "scope": "PyTorch model only; excludes decoding, resizing, API and transfer",
                }
            )
    result = {
        "checkpoint_sha256": digest_file(checkpoint),
        "checkpoint_status": saved.get("status", "trained"),
        "onnx_sha256": digest_file(path),
        "max_absolute_logit_error": maximum,
        "parity": parity,
        "tolerance": {"atol": 1e-4, "rtol": 1e-3},
        "dynamic_batch_checked": True,
        "timing": timing,
        "onnx_provider": "CPUExecutionProvider",
        "temperature": saved.get("temperature", 1),
        "size": [h, w],
        "dynamic_threshold": saved.get("dynamic_threshold", 0.5),
        "training_images": saved.get("training_images", []),
        "validation_images": saved.get("validation_images", []),
        "input": "float RGB [0,1], NCHW; normalization embedded; fixed H/W",
        "classes": 6,
    }
    write_json(root / "deployment_v2.json", result)
    return result

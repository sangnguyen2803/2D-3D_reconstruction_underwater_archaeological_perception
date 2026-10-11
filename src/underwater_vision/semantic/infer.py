"""Model-independent inference/cache interface for the pipeline and API."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from underwater_vision.semantic.model import load_model
from underwater_vision.semantic.protocol import digest_file
from underwater_vision.utils.io import atomic_npz, fingerprint, read_rgb


class SemanticPredictor:
    def __init__(
        self, checkpoint, device="auto", cache_root="cache/semantic_v2", require_trained=True
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.checkpoint_hash = digest_file(checkpoint)
        self.model, self.metadata = load_model(checkpoint, self.device)
        if require_trained and self.metadata.get("status") != "trained":
            raise ValueError("Untrained smoke checkpoint cannot be used for semantic experiments")
        self.temperature = float(self.metadata.get("temperature", 1))
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.size = tuple(self.metadata.get("size", [384, 512]))
        self.cache_root = Path(cache_root)
        self.backend = "pytorch"

    @torch.inference_mode()
    def logits(self, rgb):
        image = cv2.resize(rgb, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()[None].to(self.device) / 255
        logits = self.model(tensor)
        return (
            F.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)[0]
            .float()
            .cpu()
            .numpy()
        )

    def predict(self, rgb):
        logits = torch.from_numpy(self.logits(rgb)) / self.temperature
        return torch.softmax(logits, dim=0).numpy()

    def cached(self, image_path, max_size=960):
        key = fingerprint(
            Path(image_path),
            {
                "semantic": "v2",
                "checkpoint": self.checkpoint_hash,
                "temperature": self.temperature,
                "input_size": self.size,
                "output_max_size": max_size,
                "backend": self.backend,
            },
        )
        self.cache_root.mkdir(parents=True, exist_ok=True)
        path = self.cache_root / (key + ".npz")
        if path.exists():
            with np.load(path) as z:
                probabilities = z["probabilities"].astype(np.float32)
                return probabilities / np.maximum(probabilities.sum(0, keepdims=True), 1e-12)
        from underwater_vision.preprocessing.images import resize_rgb

        image, _ = resize_rgb(read_rgb(image_path), max_size)
        probabilities = self.predict(image)
        atomic_npz(
            path,
            probabilities=probabilities.astype(np.float16),
            checkpoint_hash=np.asarray(self.checkpoint_hash),
            temperature=np.asarray(self.temperature),
        )
        probabilities = probabilities.astype(np.float16).astype(np.float32)
        return probabilities / np.maximum(probabilities.sum(0, keepdims=True), 1e-12)


class OnnxSemanticPredictor(SemanticPredictor):
    """Same inference/cache contract with ONNX Runtime as the model backend."""

    def __init__(
        self, onnx_path, metadata_path, cache_root="cache/semantic_v2", require_trained=True
    ):
        import json

        import onnxruntime as ort

        self.metadata = json.loads(Path(metadata_path).read_text())
        if digest_file(onnx_path) != self.metadata["onnx_sha256"]:
            raise ValueError("ONNX file hash differs from export metadata")
        if require_trained and self.metadata.get("checkpoint_status") != "trained":
            raise ValueError("Untrained smoke ONNX cannot be used for semantic experiments")
        self.checkpoint_hash = self.metadata["checkpoint_sha256"]
        self.temperature = float(self.metadata["temperature"])
        self.size = tuple(self.metadata.get("size") or self.metadata["timing"][0]["input_size"])
        self.cache_root = Path(cache_root)
        self.backend = "onnx_" + self.metadata["onnx_sha256"]
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def logits(self, rgb):
        image = cv2.resize(rgb, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        tensor = np.transpose(image.astype(np.float32) / 255, (2, 0, 1))[None]
        logits = self.session.run(None, {"rgb": tensor})[0][0]
        return cv2.resize(
            logits.transpose(1, 2, 0), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)


class TeacherCache:
    """Explicitly uncalibrated teacher masks for the mask-only ablation."""

    def __init__(self, root):
        self.root = Path(root)
        self.checkpoint_hash = digest_file(self.root / "teacher_provenance.json")

    def cached(self, image_path, max_size=960):
        from underwater_vision.preprocessing.images import resize_rgb

        image, _ = resize_rgb(read_rgb(image_path), max_size)
        with np.load(self.root / (Path(image_path).stem + ".npz")) as z:
            mask = cv2.resize(
                z["mask"], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        p = np.full((6, *mask.shape), 1 / 6, np.float32)
        valid = mask != 255
        p[:, valid] = 0
        for c in range(6):
            p[c, mask == c] = 1
        return p

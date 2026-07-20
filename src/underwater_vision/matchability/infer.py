"""Full-image map inference, aligned sampling and temperature-aware caching."""

from pathlib import Path

import cv2
import numpy as np
import torch

from underwater_vision.matchability.calibration import calibrated_probabilities, probabilities
from underwater_vision.matchability.loss import sample_logits
from underwater_vision.matchability.model import load_model
from underwater_vision.matchability.protocol import digest
from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.utils.io import atomic_npz, fingerprint, read_rgb


class MatchabilityPredictor:
    def __init__(self, checkpoint, device="auto", calibration=None):
        self.device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        self.model, self.metadata = load_model(checkpoint, self.device)
        self.checkpoint_hash = digest(checkpoint)
        self.temperature = self.metadata["temperature"]
        self._set_calibration(calibration)

    def _set_calibration(self, path):
        from underwater_vision.matchability.protocol import read_json

        self.calibration = read_json(path) if path is not None else None
        self.calibration_hash = digest(path) if path is not None else None
        if self.calibration is not None:
            if self.calibration.get("checkpoint_hash") != self.checkpoint_hash:
                raise ValueError("Calibration was fitted for another checkpoint")
            if self.calibration.get("fitted_on") != "validation_only":
                raise ValueError("Calibration must be fitted on validation only")
            calibrated_probabilities([0.0], self.calibration)

    @torch.inference_mode()
    def logits(self, rgb):
        tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            .float()[None]
            .to(self.device)
            / 255
        )
        device_type = torch.device(self.device).type
        with torch.autocast(device_type, enabled=device_type == "cuda"):
            logits = self.model(tensor)
        return logits[0, 0].float().cpu().numpy()

    def predict(self, rgb):
        logits = self.logits(rgb)
        if self.calibration is not None:
            return calibrated_probabilities(logits, self.calibration).astype(np.float32)
        return probabilities(logits, self.temperature)

    def cached(self, path, max_size=960):
        root = Path("cache/matchability_v3")
        root.mkdir(parents=True, exist_ok=True)
        configuration = {
            "spec": "dense_matchability_v3",
            "hash": self.checkpoint_hash,
            "temperature": self.temperature,
            "max_size": max_size,
            "backend": type(self).__name__,
            "device": self.device,
            "precision": "float16_autocast" if str(self.device).startswith("cuda") else "fp32",
        }
        if self.calibration_hash is not None:
            configuration["calibration_hash"] = self.calibration_hash
        key = fingerprint(path, configuration)
        cache = root / (key + ".npz")
        if cache.exists():
            with np.load(cache) as z:
                return z["matchability"]
        rgb, _ = resize_rgb(read_rgb(path), max_size)
        result = self.predict(rgb)
        atomic_npz(cache, matchability=result)
        return result


def sample_probability(values, xy):
    tensor = torch.from_numpy(np.ascontiguousarray(values)).float()[None, None]
    return sample_logits(tensor, torch.as_tensor(xy, dtype=torch.float32))[0].numpy()


def attach_matchability(record, values, checkpoint_hash):
    if tuple(record["shape"]) != tuple(values.shape):
        raise ValueError("Matchability image/keypoint resolution mismatch")
    result = dict(record)
    result["matchability"] = sample_probability(values, record["xy"])
    result["matchability_hash"] = np.asarray(checkpoint_hash)
    result.setdefault("source_indices", np.arange(len(record["xy"]), dtype=np.int32))
    return result


def select_keypoints(record, count, source="matchability", seed=42):
    if count < 1 or source not in {"matchability", "response", "random"}:
        raise ValueError("Invalid keypoint budget/source")
    size = len(record["xy"])
    if count > size:
        raise ValueError("Exact keypoint budget exceeds detected keypoints")
    order = (
        np.random.default_rng(seed).permutation(size)
        if source == "random"
        else np.argsort(-record[source], kind="stable")
    )
    selected = np.sort(order[:count])
    result = {
        k: (
            v[selected]
            if k not in {"shape", "matchability_hash", "segmentation_hash"}
            and isinstance(v, np.ndarray)
            and v.ndim > 0
            and len(v) == size
            else v
        )
        for k, v in record.items()
    }
    result["source_indices"] = record.get("source_indices", np.arange(size))[selected]
    return result


class OnnxMatchabilityPredictor(MatchabilityPredictor):
    def __init__(self, model_path, metadata_path, calibration=None):
        import onnxruntime as ort

        from underwater_vision.matchability.protocol import read_json

        self.metadata = read_json(metadata_path)
        if (
            self.metadata.get("feature_spec") != "dense_matchability_v3"
            or self.metadata.get("status") != "trained"
        ):
            raise ValueError("Trained matchability ONNX metadata required")
        if digest(model_path) != self.metadata["onnx_sha256"]:
            raise ValueError("ONNX model hash mismatch")
        self.checkpoint_hash = self.metadata["checkpoint_hash"]
        self.temperature = self.metadata["temperature"]
        self._set_calibration(calibration)
        self.device = "onnx_cpu_fp32"
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def logits(self, rgb):
        original = rgb.shape[:2]
        h, w = self.metadata["input_hw"]
        image = cv2.resize(rgb, (w, h)).transpose(2, 0, 1)[None].astype(np.float32) / 255
        result = self.session.run(None, {"rgb": image})[0][0, 0]
        return cv2.resize(result, (original[1], original[0]), interpolation=cv2.INTER_LINEAR)

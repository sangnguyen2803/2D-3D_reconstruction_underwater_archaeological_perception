"""Lossless, bounded cache for frozen DINO evaluation; never used for augmented RGB."""

from pathlib import Path

import numpy as np
import torch

from underwater_vision.utils.io import atomic_npz, fingerprint


class FrozenEvaluationCache:
    def __init__(self, maximum_bytes=5_000_000_000):
        self.root = Path("cache/matchability_dino_eval_fp32_v3")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.resolve().parent != Path("cache").resolve():
            raise ValueError("Frozen cache must stay inside the workspace cache")
        self.maximum_bytes = maximum_bytes
        self.bytes = sum(p.stat().st_size for p in self.root.glob("*.npz"))

    def grid(self, model, rgb, source_path):
        key = fingerprint(
            source_path,
            {
                "spec": "frozen_dino_lossless_fp32_v3",
                "backbone": model.specification.get("backbone", "dinov2_vitb14"),
                "revision": model.specification["revision"],
                "amp": (
                    str(torch.get_autocast_dtype(rgb.device.type))
                    if torch.is_autocast_enabled(rgb.device.type)
                    else "disabled"
                ),
                "encoder_output": "float32_lossless_original_strides",
                "preprocessing": "RGB960_noCLAHE_ImageNet_pad14",
            },
        )
        path = self.root / (key + ".npz")
        if path.exists():
            with np.load(path) as z:
                array, strides = z["grid"], tuple(map(int, z["strides"]))
            result = torch.empty_strided(
                array.shape, strides, dtype=torch.float32, device=rgb.device
            )
            result.copy_(torch.from_numpy(array).to(rgb.device))
            path.touch()
            return result
        result = model.frozen_grid(rgb)
        atomic_npz(
            path, grid=result.float().cpu().numpy(), strides=np.asarray(result.stride(), np.int64)
        )
        self.bytes += path.stat().st_size
        if self.bytes > self.maximum_bytes:
            for old in sorted(self.root.glob("*.npz"), key=lambda p: p.stat().st_mtime_ns):
                if old == path or old.resolve().parent != self.root.resolve():
                    continue
                self.bytes -= old.stat().st_size
                old.unlink()
                if self.bytes <= self.maximum_bytes:
                    break
        return result

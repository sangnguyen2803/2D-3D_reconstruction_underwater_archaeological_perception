"""Check lossless frozen-grid cache against a trained DINO head on real RGB."""

import argparse
from pathlib import Path

import numpy as np
import torch

from underwater_vision.matchability.frozen_cache import FrozenEvaluationCache
from underwater_vision.matchability.model import load_model
from underwater_vision.matchability.protocol import digest, new_run
from underwater_vision.utils.io import read_rgb, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = new_run(a.output)
    torch.set_num_threads(2)
    model, _ = load_model(a.checkpoint, "cuda")
    image = read_rgb(a.image)
    rgb = (
        torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()[None].cuda() / 255
    )
    cache = FrozenEvaluationCache()
    with torch.inference_mode(), torch.autocast("cuda"):
        initial_grid = cache.grid(model, rgb, a.image)
        initial = model.decode_grid(initial_grid, image.shape[:2]).float()
        restored_grid = cache.grid(model, rgb, a.image)
        restored = model.decode_grid(restored_grid, image.shape[:2]).float()
        fresh = model(rgb).float()
    cache_error = float((initial - restored).abs().max())
    fresh_error = float((fresh - restored).abs().max())
    result = {
        "checkpoint_sha256": digest(a.checkpoint),
        "image_sha256": digest(a.image),
        "cache_dtype": "float32 lossless; original tensor strides restored",
        "encoder_dtype": str(initial_grid.dtype),
        "original_strides": list(initial_grid.stride()),
        "restored_strides": list(restored_grid.stride()),
        "grid_exact": bool(torch.equal(initial_grid, restored_grid)),
        "maximum_absolute_cache_logit_error": cache_error,
        "maximum_absolute_fresh_forward_error": fresh_error,
        "atol": 1e-6,
        "passed": cache_error <= 1e-6 and fresh_error <= 1e-6,
    }
    write_json(root / "parity_v3.json", result)
    print(result)
    if not result["passed"]:
        raise AssertionError("Lossless cache differs from the original inference path")


if __name__ == "__main__":
    main()
